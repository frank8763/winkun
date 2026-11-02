"""Conditional relative future-position density ratio, not first-hit Q."""
from pathlib import Path
import sys,hashlib,io,json,math,copy,gzip
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t275_future_target_contract'))
import future_data as F
sys.path.insert(0,str(ROOT/'stage/experiments/t290_exact_duplicate_store'))
from store import resolve


def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def digest(x):
    if isinstance(x,torch.Tensor):x=x.detach().cpu().numpy()
    if isinstance(x,np.ndarray):return hashlib.sha256(str((x.dtype.str,x.shape)).encode()+x.tobytes()).hexdigest()
    if isinstance(x,dict):return hashlib.sha256(''.join(str(k)+digest(v) for k,v in sorted(x.items(),key=lambda kv:str(kv[0]))).encode()).hexdigest()
    if isinstance(x,(tuple,list)):return hashlib.sha256(''.join(digest(v) for v in x).encode()).hexdigest()
    return hashlib.sha256(repr(x).encode()).hexdigest()
def clean(x):
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
def save(f,x):Path(f).write_text(json.dumps(x,indent=2,allow_nan=False,default=clean)+'\n')


class Head(torch.nn.Module):
    def __init__(self):
        super().__init__();self.state=torch.nn.Sequential(torch.nn.Linear(123,64),torch.nn.Tanh(),torch.nn.Linear(64,32))
        self.goal=torch.nn.Sequential(torch.nn.Linear(2,32),torch.nn.Tanh(),torch.nn.Linear(32,32))
    def forward(self,x,g,blind=False):
        if blind:x=torch.cat([x[:,:120],torch.zeros_like(x[:,120:122]),x[:,122:]],dim=1)
        return self.state(x)@self.goal(g).T/math.sqrt(32)


def load(p):
    old=json.loads(Path(p['data_protocol']).read_text())
    with np.load(p['pairs']) as z:pairs={k:z[k].copy() for k in z.files}
    x=np.empty((len(pairs['episode']),123),np.float32);g=pairs['goal_body_m'].astype(np.float64)/10
    for e,row in enumerate(old['records']):
        f=resolve(row['path']);assert sha(f)==row['sha256'];idx=np.flatnonzero(pairs['episode']==e)
        with np.load(f) as z:
            k=pairs['anchor'][idx].astype(np.int64);j=pairs['endpoint'][idx].astype(np.int64);assert not np.any(z['end_flags'][j,0])
            assert np.allclose(F.relative(z['poses'][k],z['next_poses'][j,:2]),pairs['goal_body_m'][idx],atol=5e-7,rtol=0)
            starts=np.r_[0,np.cumsum(z['native_counts'].astype(np.int64))[:-1]];ends=np.cumsum(z['native_counts'].astype(np.int64))
            assert np.array_equal(ends[j]-starts[k],pairs['delta_steps'][idx])
            x[idx]=np.c_[z['states'][k],z['actions'][k],pairs['delta_steps'][idx].astype(np.float64)*.002/32]
    tr=np.flatnonzero(pairs['train']).astype(np.int64);te=np.flatnonzero(1-pairs['train']).astype(np.int64)
    assert (len(tr),len(te))==(21836,4559)
    train_epi=sorted(set(pairs['episode'][tr].tolist()));test_epi=sorted(set(pairs['episode'][te].tolist()));assert len(train_epi)==80 and len(test_epi)==16 and not set(train_epi)&set(test_epi)
    pools=[np.flatnonzero((pairs['episode']==e)&pairs['train']) for e in train_epi]
    w=np.zeros(len(x),np.float64)
    for pool in pools:w[pool]=1/(80*len(pool))
    mu=(x*w[:,None]).sum(0);sd=np.sqrt(np.maximum(((x-mu)**2*w[:,None]).sum(0),1e-6));gm=(g*w[:,None]).sum(0);gs=np.sqrt(np.maximum(((g-gm)**2*w[:,None]).sum(0),1e-6))
    xx=np.clip((x-mu)/sd,-10,10).astype(np.float32);gg=np.clip((g-gm)/gs,-10,10).astype(np.float32)
    return dict(x=xx,g=gg,train=tr,test=te,pairs=pairs,records=old['records'],pools=pools,normalization=dict(mu=mu,sd=sd,goal_mu=gm,goal_sd=gs))


def sample_batches(data,p):
    rng=np.random.default_rng(p['seed']);out=[]
    for _ in range(p['steps']):
        ep=rng.integers(0,80,p['batch']);out.append(np.array([data['pools'][e][rng.integers(len(data['pools'][e]))] for e in ep]))
    return out


def train(data,p,guard,event,holder=None):
    torch.manual_seed(p['seed']);initial=Head();models={'action':copy.deepcopy(initial),'blind':copy.deepcopy(initial)}
    opt={n:torch.optim.Adam(q.parameters(),lr=p['learning_rate']) for n,q in models.items()};x=torch.tensor(data['x']);g=torch.tensor(data['g']);losses=[];count={n:0 for n in models};calls=0
    if holder is not None:holder.update(models=models,opt=opt,initial=digest(initial.state_dict()))
    for step,idx in enumerate(sample_batches(data,p)):
        guard('TRAIN_'+str(step));ix=torch.tensor(idx,dtype=torch.long);labels=torch.arange(len(ix));row=[]
        for name,q in models.items():
            o=opt[name];o.zero_grad();loss=torch.nn.functional.cross_entropy(q(x[ix],g[ix],name=='blind'),labels)
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(q.parameters(),5,error_if_nonfinite=True)
            event(name,step);o.step();calls+=1;count[name]+=1;row.append(float(loss.detach()))
        losses.append(row)
    assert count==dict(action=500,blind=500) and calls==1000
    assert torch.equal(models['blind'].state[0].weight[:,120:122],initial.state[0].weight[:,120:122])
    return models,opt,np.array(losses,dtype=np.float64),digest(initial.state_dict())


def score_arrays(q,data,p,blind):
    rng=np.random.default_rng(p['eval_seed']);neg=data['train'][rng.integers(len(data['train']),size=(len(data['test']),31))]
    assert not np.any(np.isin(neg,data['test']));allg=np.column_stack([data['test'],neg]);out=[]
    with torch.no_grad():
        for k in range(0,len(allg),128):
            idx=data['test'][k:k+128];gi=allg[k:k+128];x=torch.tensor(data['x'][idx]);g=torch.tensor(data['g'][gi])
            if blind:x=torch.cat([x[:,:120],torch.zeros_like(x[:,120:122]),x[:,122:]],dim=1)
            a=q.state(x);b=q.goal(g);out.append((a[:,None,:]*b).sum(-1).div(math.sqrt(32)).numpy())
    return np.concatenate(out),neg


def metrics(scores,data):
    t=torch.tensor(scores,dtype=torch.float64);nll=(-t[:,0]+torch.logsumexp(t,dim=1)).numpy();mx=scores.max(1);ties=(scores==mx[:,None]);acc=ties[:,0]/ties.sum(1);epi=data['pairs']['episode'][data['test']]
    episode=[]
    for e in sorted(set(epi.tolist())):
        mask=epi==e;r=data['records'][e];episode.append(dict(episode=int(e),id=r['id'],group=str(r['distance_m'])+'_'+r['side'],n=int(mask.sum()),NLL=float(nll[mask].mean()),tie_aware_top1=float(acc[mask].mean())))
    groups={}
    for group in sorted({r['group'] for r in episode}):
        v=[r for r in episode if r['group']==group];groups[group]=dict(NLL=float(np.mean([r['NLL'] for r in v])),top1=float(np.mean([r['tie_aware_top1'] for r in v])))
    return dict(equal_episode_NLL=float(np.mean([r['NLL'] for r in episode])),equal_episode_top1=float(np.mean([r['tie_aware_top1'] for r in episode])),episodes=episode,groups=groups)


def screen(a,b):
    checks=dict(heldout_NLL_5pct_better=a['equal_episode_NLL']<=.95*b['equal_episode_NLL'],
        heldout_top1_2percentage_points_better=a['equal_episode_top1']>=b['equal_episode_top1']+.02,
        each_group_NLL_not_over102pct=all(a['groups'][k]['NLL']<=1.02*v['NLL'] for k,v in b['groups'].items()))
    return dict(checks=checks,passed=all(checks.values()))
