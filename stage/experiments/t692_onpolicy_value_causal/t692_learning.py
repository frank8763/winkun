"""Same-behavior terminal MC projected value; never guides the frozen Actor."""
from pathlib import Path
import copy,gzip,io,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t673_all_task_signal_gate'))
import t673_analysis as B
M,T,torch,np=B.M,B.T,B.torch,B.np
read,sha,digest=B.read,B.sha,B.digest

class Value(torch.nn.Module):
    def __init__(self,blind,seed):
        super().__init__();self.blind=blind
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.net=torch.nn.Sequential(torch.nn.Linear(186,64),torch.nn.Tanh(),torch.nn.Linear(64,64),torch.nn.Tanh(),torch.nn.Linear(64,1),torch.nn.Tanh()).double()
            torch.nn.init.zeros_(self.net[-2].weight);torch.nn.init.zeros_(self.net[-2].bias)
        assert sum(v.numel() for v in self.parameters())==16193
    def forward(self,x):
        if self.blind:x=torch.cat([x[...,:184],torch.zeros_like(x[...,184:])],-1)
        return self.net(x)[...,0]

def make(p,arm):
    q=Value(arm=='blind',p['value_seed']);opt=torch.optim.Adam(q.parameters(),lr=1e-3,eps=1e-8,foreach=False,fused=False)
    return q,opt
def state(obj):return copy.deepcopy(dict(value=obj[0].state_dict(),optimizer=obj[1].state_dict()))
def load(obj,s):
    obj[0].load_state_dict(copy.deepcopy(s['value']));obj[1].load_state_dict(copy.deepcopy(s['optimizer']))

def data(p,guard):
    q=M.L.frozen(p,p['initial']['path']);actor_before=digest(q.state_dict());groups=B.dataset(p)
    chunks=[];episodes=[];by_task=[];offset=0
    with torch.no_grad():
        for task,rows in groups.items():
            train=[];held=[]
            for rep,row in enumerate(rows):
                guard('features',task);assert sha(row['path'])==row['sha256'] and row['policy_hash']==q.phash()
                a=T.R.E.N.S.arrays(row['path']);assert all(digest(v)==row['array_hashes'][k] for k,v in a.items())
                assert not a['ownership'].any() and a['stochastic'].all() and np.array_equal(a['actions'],a['raw_actions'].clip(-1,1).astype(np.float32))
                g=M.features(q,a,task,row['sha256']);mu=M.location(q,g)
                assert float((mu-torch.tensor(a['locations'])).abs().max())<1e-10
                assert float((M.P.L.density(mu,torch.tensor(a['raw_actions']),torch.tensor(a['atom']),p['raw_sigma'])-torch.tensor(a['logp'])).abs().max())<1e-9
                inputs=torch.cat([g['f'],torch.tensor(a['actions']).double()],-1);n=len(inputs);assert 0<n<=1024
                test=rep>=(6 if task.startswith('t502_') else 3)
                entry=dict(task=task,offset=offset,length=n,label=M.outcome(row,'collision_cost'),heldout=test,source=row['path'],source_sha256=row['sha256'],rep=rep)
                (held if test else train).append(len(episodes));episodes.append(entry);chunks.append(inputs);offset+=n
            by_task.append(dict(task=task,train=train,heldout=held))
    inputs=torch.cat(chunks);del chunks
    assert len(episodes)==1724 and sum(e['heldout'] for e in episodes)==431 and len(by_task)==407
    assert digest(q.state_dict())==actor_before and all(v.grad is None for v in q.parameters())
    manifest=dict(feature_order='owncarry64_then_original_fixed_normalized120_then_current_actual_command2',no_train_fitted_normalization=True,
        episodes=episodes,feature_sha256=digest(inputs),initial_actor_sha256=actor_before,train_episodes=1293,heldout_episodes=431)
    return dict(x=inputs,episodes=episodes,tasks=by_task,manifest=manifest,protected384=p['before_success_ids'])

def batch(dataset,rng,size=1024):
    tasks=rng.integers(0,407,size=size);slots=rng.integers(0,1024,size=size)
    # Different per-ID episode counts require uniform 0..n-1 draws, rather
    # than modulo a common count (which would bias three versus six).
    counts=np.array([len(dataset['tasks'][k]['train']) for k in tasks]);pick=rng.integers(np.zeros(size,dtype=np.int64),counts)
    rows=np.array([dataset['tasks'][int(k)]['train'][int(j)] for k,j in zip(tasks,pick)])
    ep=[dataset['episodes'][int(i)] for i in rows];length=np.array([e['length'] for e in ep]);valid=slots<length
    indices=np.array([e['offset'] for e in ep])+np.minimum(slots,length-1)
    x=dataset['x'][torch.tensor(indices)].clone();x[torch.tensor(~valid)]=0
    y=torch.tensor([e['label'] for e in ep],dtype=torch.float64);mask=torch.tensor(valid,dtype=torch.float64)
    return x,y,mask

def loss(q,pack):
    x,y,mask=pack
    return (2*(q(x)-y).square()*mask).mean()

def fit(p,obj,dataset,guard,event):
    q,opt=obj;assert not opt.state;rng=np.random.default_rng(p['batch_seed']);trace=[]
    for step in range(500):
        guard('fit',str(step));pack=batch(dataset,rng);opt.zero_grad(set_to_none=True);value=loss(q,pack);value.backward()
        assert bool(torch.isfinite(value)) and all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in q.parameters())
        event('BEFORE',step);opt.step();event('AFTER',step)
        assert all(bool(torch.isfinite(v).all()) for v in q.parameters())
        trace.append(float(value.detach()))
    return dict(steps=500,trace=trace,state_sha256=digest(state(obj)),fixed_batch_stream_seed=p['batch_seed'])

def metrics(obj,dataset,heldout):
    q=obj[0];per={};initial=[]
    with torch.no_grad():
        for task in dataset['tasks']:
            indices=task['heldout' if heldout else 'train'];total=0.;starts=[];labels=[]
            for i in indices:
                e=dataset['episodes'][i];x=dataset['x'][e['offset']:e['offset']+e['length']];prediction=q(x);err=(prediction-e['label']).square()
                total+=float(err.sum())/(len(indices)*512);starts.append(float(err[0]));initial.append(float(err[0]));labels.append(e['label'])
            per[task['task']]=dict(fixed_decision_loss=total,initial_decision_MSE=sum(starts)/len(starts),episodes=len(indices),labels=labels)
    old=[v['fixed_decision_loss'] for k,v in per.items() if not k.startswith('t502_')];field=[v['fixed_decision_loss'] for k,v in per.items() if k.startswith('t502_')]
    protected=[v['fixed_decision_loss'] for k,v in per.items() if k in dataset['protected384']];assert len(protected)==384
    return dict(fixed_decision_loss=sum(v['fixed_decision_loss'] for v in per.values())/407,initial_decision_MSE=sum(initial)/len(initial),
        old384_fixed_loss=sum(protected)/384,nonfield_fixed_loss=sum(old)/len(old),field_fixed_loss=sum(field)/len(field),per_task=per)

def regression_gate(results):
    a,b=results['aware']['heldout'],results['blind']['heldout']
    checks=dict(global_fixed=a['fixed_decision_loss']<=.9*b['fixed_decision_loss'],global_initial=a['initial_decision_MSE']<=.9*b['initial_decision_MSE'],
        old384_no_over2pct=a['old384_fixed_loss']<=1.02*b['old384_fixed_loss'],field_no_over2pct=a['field_fixed_loss']<=1.02*b['field_fixed_loss'])
    return dict(passed=all(checks.values()),checks=checks,engineering_not_statistical_gate=True,no_Actor_guidance=True)

def export(obj,p,arm,folder):
    s=state(obj);buf=io.BytesIO();torch.save(dict(state=s,state_sha256=digest(s),arm=arm,architecture='T692_MC186_64_64_1'),buf)
    blob=gzip.compress(buf.getvalue(),mtime=0);assert len(blob)<p['model_bytes'];path=Path(folder)/f'{arm}.pt.gz'
    with path.open('xb') as f:f.write(blob)
    return dict(path=str(path),sha256=sha(path),state_sha256=digest(s),bytes=len(blob),critic_only=True)

def branch_predictions(p,objects,dataset):
    answer={}
    for j in p['branch_jobs']:
        e=next(e for e in dataset['episodes'] if e['source']==j['source_path']);assert e['heldout'] and j['branch_decision']==20
        x=dataset['x'][e['offset']+20].clone();key=j['id'];answer[key]={}
        a=T.R.E.N.S.arrays(j['source_path']);mu=a['locations'][20];cmds=[]
        for action_index,delta in enumerate(p['action_deltas']):
            command=np.clip(mu+np.asarray(delta),-1,1).astype(np.float32);cmds.append(command)
            inp=x.clone();inp[184:]=torch.tensor(command).double()
            with torch.no_grad():values={arm:float(obj[0](inp[None])[0]) for arm,obj in objects.items()}
            answer[key][str(action_index)]=dict(command=command.tolist(),values=values,source_sha256=j['source_sha256'])
        assert len({u.tobytes() for u in cmds})==4
    return answer
