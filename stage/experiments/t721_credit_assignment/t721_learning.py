"""Finite iid censored-Gaussian PPO on fresh conditional suffixes only."""
from pathlib import Path
import copy,gzip,io,json,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t703_projected_local'));import t703_actor as A
sys.path.insert(0,str(ROOT/'stage/experiments/t699_matched_value'));import t699_actor as B
torch,np,digest=A.torch,A.np,A.digest
sys.path.insert(0,str(ROOT/'stage/experiments/t441_joint_actor_critic'));import t441_learning as Density
DENSITY=Density.density

def checkpoint(path):return B.Existing.Base.P.checkpoint(path)
def fixed(q):return digest({k:v for k,v in q.state_dict().items() if not k.startswith('replay_head.')})
def make(p):
    q=A.frozen(p,p['learner_initial']['path']);q.requires_grad_(False);q.replay_head.requires_grad_(True)
    assert sum(v.numel() for v in q.parameters() if v.requires_grad)==20198
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(p['seed']);v=torch.nn.Sequential(torch.nn.Linear(184,64),torch.nn.Tanh(),torch.nn.Linear(64,1)).double();torch.nn.init.zeros_(v[-1].weight);torch.nn.init.zeros_(v[-1].bias)
    ao=torch.optim.Adam(q.replay_head.parameters(),lr=p['actor_lr'],foreach=False,fused=False)
    vo=torch.optim.Adam(v.parameters(),lr=p['value_lr'],foreach=False,fused=False)
    return q,v,ao,vo

def state(obj):return copy.deepcopy({k:v.state_dict() for k,v in zip(['actor','value','actor_opt','value_opt'],obj)})
def load(obj,s):
    for k,v in zip(['actor','value','actor_opt','value_opt'],obj):v.load_state_dict(copy.deepcopy(s[k]))
def write_state(path,s):
    b=io.BytesIO();torch.save(s,b);blob=gzip.compress(b.getvalue(),mtime=0);assert len(blob)<4*2**20
    with Path(path).open('xb') as f:f.write(blob)
    return dict(path=str(path),state_hash=digest(s))
def read_state(row):
    s=torch.load(io.BytesIO(gzip.decompress(Path(row['path']).read_bytes())),map_location='cpu',weights_only=False);assert digest(s)==row['state_hash'];return s

def features(q,a):
    q.sample=False;q.rng=None;q.reset_noise();h=None;f=[];base=[];locations=[]
    with torch.no_grad():
        for x in a['states']:
            xx=torch.tensor(x[None]);_,mu,h,_=q.step(xx,h);n=q.normalized(xx)
            f.append(torch.cat([h[0,0],n[0]],-1));base.append(B.Parent.Actor.location(q,h[0],n)[0]);locations.append(mu[0])
    q.reset_noise();g=dict(f=torch.stack(f),base=torch.stack(base));mu=location(q,g)
    assert float((mu-torch.stack(locations)).abs().max())<1e-10,'Batch1/cached locations differ'
    assert torch.equal(torch.stack(f)[:,:64],torch.tensor(np.vstack([a['hidden_before'][1:],a['last_hidden_after'][None]])))
    return g

def location(q,g):return g['base']+q.replay_head(g['f'])
def gae(reward,discount,value,lam):
    assert len(reward)==len(discount)==len(value);out=torch.zeros_like(value);last=value.new_zeros(())
    for i in range(len(value)-1,-1,-1):
        nxt=value[i+1] if i+1<len(value) else value.new_zeros(())
        last=reward[i]+discount[i]*nxt-value[i]+discount[i]*lam*last;out[i]=last
    return out,out+value

def dataset(obj,values,p):
    q,v,*_=obj;data=[]
    for a,r in values:
        k=int(a.get('recovery_first',0));assert r['complete'] and not r['administrative_censor'] and a['end_flags'][-1].sum()==1 and not a['end_flags'][:-1].any()
        assert not a['ownership'][k:].any() and a['stochastic'][k:].all() and not a['stochastic'][:k].any()
        assert k<len(a['states']);g=features(q,a)
        with torch.no_grad():
            raw=torch.tensor(a['raw_actions']);atoms=torch.tensor(a['atom']);old=torch.tensor(a['logp'])
            assert float((DENSITY(location(q,g)[k:],raw[k:],atoms[k:],p['raw_sigma'])-old[k:]).abs().max())<1e-8
            vv=v(g['f'])[:,0];rw=torch.tensor(a['rewards']);dis=torch.tensor(p['gamma_native']**a['native_counts']);dis[-1]=0
            adv,_=gae(rw[k:],dis[k:],vv[k:],p['gae_lambda']);_,ret=gae(rw[k:],dis[k:],torch.zeros_like(vv[k:]),1.0)
        data.append(dict(f=g['f'][k:],base=g['base'][k:],raw=raw[k:],atoms=atoms[k:],old=old[k:],adv=adv,target=ret,first=k,id=r['job_id']))
    adv=torch.cat([g['adv'] for g in data]);mean=adv.mean();std=adv.std(unbiased=False).clamp_min(1e-8)
    for g in data:g['raw_adv_mean']=float(mean);g['raw_adv_std']=float(std)
    for g in data:g['adv']=(g['adv']-mean)/std
    return data

def fit(obj,values,p,guard,event):
    q,v,ao,vo=obj;body=fixed(q);data=dataset(obj,values,p);n=sum(len(g['adv']) for g in data);trace=[]
    assert n>0
    for epoch in range(p['epochs']):
        guard();ao.zero_grad(set_to_none=True);total=0.
        for g in data:
            lp=DENSITY(location(q,g),g['raw'],g['atoms'],p['raw_sigma']);ratio=(lp-g['old']).exp()
            loss=-torch.minimum(ratio*g['adv'],ratio.clamp(1-p['clip'],1+p['clip'])*g['adv']).sum()/n
            loss.backward();total+=float(loss.detach())
        norm=float(torch.nn.utils.clip_grad_norm_(q.replay_head.parameters(),.5,error_if_nonfinite=True));event('actor','BEFORE',epoch);ao.step();event('actor','AFTER',epoch)
        assert fixed(q)==body and all(torch.isfinite(x).all() for x in q.parameters())
        trace.append(dict(epoch=epoch,policy_loss=total,gradient_norm=norm))
    allf=torch.cat([g['f'] for g in data]);targets=torch.cat([g['target'] for g in data]);vt=[]
    for epoch in range(p['epochs']):
        guard();vo.zero_grad(set_to_none=True);loss=.5*(v(allf)[:,0]-targets).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(v.parameters(),.5,error_if_nonfinite=True);event('value','BEFORE',epoch);vo.step();event('value','AFTER',epoch);vt.append(float(loss.detach()))
    with torch.no_grad():
        delta=torch.cat([DENSITY(location(q,g),g['raw'],g['atoms'],p['raw_sigma'])-g['old'] for g in data]);ratios=delta.exp();pred=v(allf)[:,0]
        diagnostic=dict(raw_adv_mean=data[0]['raw_adv_mean'],raw_adv_std=data[0]['raw_adv_std'],return_mean=float(targets.mean()),return_std=float(targets.std(unbiased=False)),value_MSE=float((pred-targets).square().mean()),approx_KL=float(((ratios-1)-delta).mean()),clip_fraction=float(((ratios-1).abs()>p['clip']).double().mean()))
    return dict(diagnostic=diagnostic,advantage_lambda=p['gae_lambda'],common_value_target='complete_discounted_return',trace=trace,value_trace=vt,Actor_Adam=p['epochs'],Value_Adam=p['epochs'],suffix_decisions=n,prefix_excluded=True,frozen_body=body)

def export(q,p,path):
    joint=dict(actor=copy.deepcopy(q.state_dict()));b=io.BytesIO();torch.save(dict(joint_state=joint,architecture=B.ARCH,policy_hash=q.phash(),joint_hash=digest(joint),new_RL=True,optimizer_separate=True,runtime_teacher=False),b)
    blob=gzip.compress(b.getvalue(),mtime=0);assert len(blob)<3*2**20
    with Path(path).open('xb') as f:f.write(blob)
    return dict(path=str(path),sha256=B.sha(path),bytes=len(blob),policy_hash=q.phash(),joint_hash=digest(joint))
