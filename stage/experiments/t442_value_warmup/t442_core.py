"""Value-only controlled warmup; no Actor, simulator or navigation imports."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import copy,gzip,hashlib,io,json,math,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t442_value_warmup';DOC=ROOT/'stage/docs/research_rescue_20261109'
sys.path.insert(0,str(ROOT/'stage/experiments/joint_recurrent_actor_critic'))
from physical_gae import physical_gae


def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def read(f):return json.loads(Path(f).read_text())
def save(f,v):Path(f).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def cpu(v):
    if torch.is_tensor(v):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
    if isinstance(v,list):return [cpu(x) for x in v]
    if isinstance(v,tuple):return tuple(cpu(x) for x in v)
    return copy.deepcopy(v)


def digest(value):
    def encode(v):
        if torch.is_tensor(v):
            a=v.detach().cpu().numpy();return ['tensor',str(a.dtype),list(a.shape),hashlib.sha256(a.tobytes()).hexdigest()]
        if isinstance(v,np.ndarray):return ['array',str(v.dtype),list(v.shape),hashlib.sha256(v.tobytes()).hexdigest()]
        if isinstance(v,dict):return ['dict',[[str(k),encode(x)] for k,x in sorted(v.items(),key=lambda x:str(x[0]))]]
        if isinstance(v,(tuple,list)):return ['sequence',[encode(x) for x in v]]
        if isinstance(v,np.generic):return v.item()
        return v
    return hashlib.sha256(json.dumps(encode(value),sort_keys=True,allow_nan=False).encode()).hexdigest()


def arrays(f):
    with np.load(f) as z:a={k:z[k].copy() for k in z.files}
    if 'last_next_state'in a:a['next_states']=np.concatenate([a['states'][1:],a.pop('last_next_state')])
    return a


def source_initial(p):
    x=torch.load(io.BytesIO(gzip.decompress(Path(p['initial_checkpoint']).read_bytes())),map_location='cpu',weights_only=False)
    return x['joint_state']['value']


class Value(torch.nn.Module):
    """Exactly T441's independent V topology; no policy module is constructed."""
    def __init__(self,state):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261109442)
            self.register_buffer('mean',state['mean'].clone());self.register_buffer('scale',state['scale'].clone())
            self.encoder=torch.nn.Sequential(torch.nn.Linear(120,64),torch.nn.Tanh()).double()
            self.core=torch.nn.GRU(64,32).double();self.head=torch.nn.Linear(32,1).double()
        self.load_state_dict(state,strict=True)
    def forward(self,x):
        x=((x.double()-self.mean)/self.scale).clamp(-10,10)
        h,_=self.core(self.encoder(x)[:,None]);return self.head(h[:,0])[:,0]


def make(p,device='cuda'):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    v=Value(source_initial(p)).to(device).train()
    opt=torch.optim.Adam(v.parameters(),lr=p['lr'],foreach=False,fused=False)
    return v,opt


def state(v,opt):return cpu(dict(value=v.state_dict(),optimizer=opt.state_dict()))
def restore(objects,s):
    v,opt=objects;v.load_state_dict(s['value']);opt.load_state_dict(copy.deepcopy(s['optimizer']))


def export(objects,path,limit):
    s=state(*objects);buf=io.BytesIO();torch.save(dict(state=s,state_hash=digest(s)),buf)
    b=gzip.compress(buf.getvalue(),mtime=0);assert len(b)<=limit,(len(b),limit);Path(path).write_bytes(b)
    return dict(path=str(path),sha256=sha(path),bytes=len(b),state_hash=digest(s))


def data(p,device='cuda'):
    result={}
    for r in p['records']:
        assert sha(r['path'])==r['sha256'];a=arrays(r['path'])
        assert len(a['states'])==len(a['rewards']) and a['states'].shape[1]==120
        assert np.array_equal(a['states'][1:],a['next_states'][:-1])
        flags=np.asarray(a['end_flags']);assert flags[-1].sum()==1 and not flags[:-1].any()
        assert np.asarray(a['stochastic']).all() and not np.asarray(a['ownership']).any()
        key=hashlib.sha256(b''.join(a[k].tobytes() for k in ['initial_native_integration','task_goal_xy','scene_sha256'])).hexdigest()
        assert key==r['reset_group']
        mc=np.zeros(len(a['rewards']));g=0.
        for k in range(len(mc)-1,-1,-1):g=float(a['rewards'][k])+p['gamma_native']**int(a['native_counts'][k])*g;mc[k]=g
        x=torch.tensor(a['states'],device=device)
        result[r['key']]=dict(x=x,extended=torch.tensor(np.concatenate([a['states'],a['next_states'][-1:]]),device=device),
            mc=torch.tensor(mc,device=device),mc_numpy=mc,counts=a['native_counts'].astype(np.int64),
            rewards=a['rewards'],record=r)
    return result


def targets(v,item,arm,p):
    if arm=='MC':return item['mc'].detach()
    assert arm=='GAE'
    with torch.no_grad():vv=v(item['extended']).cpu().numpy()
    mask=np.ones((len(vv)-1,1),bool);mask[-1]=False
    _,ret=physical_gae(np.asarray(item['rewards'])[:,None],vv[:-1,None],vv[1:,None],item['counts'][:,None],mask,mask,
        gamma_native=p['gamma_native'],lambda_native=p['lambda_native'])
    return item['mc'].new_tensor(ret[:,0])


def fit(objects,dataset,p,arm,guard,event,blocks=None):
    v,opt=objects;trace=[];steps=0
    schedule=p['batch_schedule'] if blocks is None else p['batch_schedule'][:blocks]
    for block,keys in enumerate(schedule):
        guard();batch=[dataset[k] for k in keys]
        # As in T441, hold lambda targets fixed within four updates; refresh at
        # the next batch using that head's current detached recurrent values.
        ys=[targets(v,item,arm,p) for item in batch]
        for epoch in range(p['epochs_per_batch']):
            guard();opt.zero_grad(set_to_none=True);loss_value=0.
            for item,y in zip(batch,ys):
                loss=.5*(v(item['x'])-y).square().mean()/len(batch)
                assert torch.isfinite(loss);loss.backward();loss_value+=float(loss.detach())
            norm=float(torch.nn.utils.clip_grad_norm_(v.parameters(),1.,error_if_nonfinite=True))
            event('BEFORE',steps);opt.step();event('AFTER',steps)
            trace.append(dict(step=steps,block=block,epoch=epoch,loss=loss_value,gradient_norm=norm));steps+=1
    return dict(arm=arm,steps=steps,trace=trace,state_hash=digest(state(v,opt)))


def metrics(pred,target):
    e=pred-target;variance=float(np.var(target))
    return dict(mse=float(np.mean(e*e)),bias=float(np.mean(e)),start_squared_error=float(e[0]**2),
                target_variance=variance,explained_variance=None if variance<1e-15 else 1-float(np.var(e))/variance)


def evaluate(v,dataset,p):
    rows=[];predictions={}
    with torch.inference_mode():
        for key,item in dataset.items():
            pred=v(item['x']).cpu().numpy();r=item['record'];truth=item['mc_numpy']
            rows.append(dict(key=key,job_id=r['job_id'],category=r['category'],split=r['split'],
                             reset_group=r['reset_group'],**metrics(pred,truth)))
            predictions[key]=pred
    groups={}
    for split in ['train','validation']:
        groups[split]={}
        for cat in p['categories']:
            rs=[r for r in rows if r['split']==split and r['category']==cat];assert rs
            groups[split][cat]=dict(episodes=len(rs),reset_groups=len({r['reset_group'] for r in rs}),
                **{k:float(np.mean([r[k] for r in rs])) for k in ['mse','bias','start_squared_error']})
        groups[split]['category_equal_overall']={k:float(np.mean([groups[split][cat][k] for cat in p['categories']]))
                                                for k in ['mse','bias','start_squared_error']}
    return dict(rows=rows,groups=groups),predictions


def screen(zero,a,b,p):
    z=zero['groups']['validation'];aa=a['groups']['validation'];bb=b['groups']['validation'];overall='category_equal_overall'
    tests=dict(MC_overall_MSE_vs_GAE_75percent=bb[overall]['mse']<=p['overall_ratio']*aa[overall]['mse'],
               MC_overall_MSE_vs_zero_75percent=bb[overall]['mse']<=p['overall_ratio']*z[overall]['mse'],
               MC_start_MSE_not_worse_GAE=bb[overall]['start_squared_error']<=aa[overall]['start_squared_error']+1e-12,
               MC_start_MSE_not_worse_zero=bb[overall]['start_squared_error']<=z[overall]['start_squared_error']+1e-12)
    for cat in p['categories']:
        tests[cat+'_MC_MSE_not_worse_GAE102percent']=bb[cat]['mse']<=p['category_ratio']*aa[cat]['mse']+1e-12
        tests[cat+'_MC_MSE_not_worse_zero']=bb[cat]['mse']<=z[cat]['mse']+1e-12
    return dict(all_pass=all(tests.values()),tests=tests,
                navigation_success_NOT_measured=True,eligible_only_for_separate_future_online_test=all(tests.values()))
