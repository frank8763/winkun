"""Own-policy clipped policy gradients with physical-time peer RTG baseline.

Critic-free finite pilot. Full episodes are sampled from the actual Actor;
no successful-history supervision is passed as on-policy actions.
"""
from pathlib import Path
import math,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t501_joint_command_repair'));import t501_learning as A
G,P,torch,np=A.G,A.P,A.torch,A.np
read,sha,digest=G.read,G.sha,G.digest

def make(p):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);old=P.checkpoint(p['initial']['path'])['joint_state'];q=A.Actor(p);q.load_state_dict(old['actor']);assert q.phash()==p['initial']['policy_hash']
    q.requires_grad_(False);q.drive_adapter.requires_grad_(True);q.steer_adapter.requires_grad_(True)
    v=P.L.Value(q,p['value_seed']);v.load_state_dict(old['value']);v.requires_grad_(False)
    ao=torch.optim.Adam(A.variables(q),lr=p['actor_lr'],foreach=False,fused=False);ao.load_state_dict(old['actor_opt'])
    for g in ao.param_groups:g['lr']=p['actor_lr']
    vo=torch.optim.Adam(v.parameters(),lr=p['value_lr'],foreach=False,fused=False);vo.load_state_dict(old['value_opt'])
    assert all(int(ao.state[a]['step'])==896 for a in q.drive_adapter.parameters()) and all(int(ao.state[a]['step'])==256 for a in q.steer_adapter.parameters())
    if p.get('initial_joint_locked'):assert digest(P.state(q,v,ao,vo))==p['initial']['joint_hash']
    return q,v,ao,vo

def peer_batch(obj,arrays,p):
    assert len(arrays)==8;q=obj[0];items=[];errors=[]
    for i,a in enumerate(arrays):
        assert not a['ownership'].any() and a['stochastic'].all() and not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1
        assert np.array_equal(a['states'][1:],a['next_states'][:-1])
        x=torch.tensor(a['states']);raw=torch.tensor(a['raw_actions']);atom=torch.tensor(a['atom']);old=torch.tensor(a['logp'])
        with torch.no_grad():
            loc=q.sequence(x);me=float((loc-torch.tensor(a['locations'])).abs().max());le=float((P.L.density(loc,raw,atom,p['raw_sigma'])-old).abs().max());assert me<1e-10 and le<1e-9,(me,le)
        dt=a['native_counts'].astype(np.int64);time0=np.r_[0,np.cumsum(dt)[:-1]];rtg=np.empty(len(dt),float);v=0.
        for k in reversed(range(len(dt))):v=float(a['rewards'][k])+p['gamma_native']**int(dt[k])*v;rtg[k]=v
        items.append(dict(x=x,raw=raw,atom=atom,old=old,time=time0,end=int(time0[-1]+dt[-1]),rtg=rtg,group=i//4,terminal=a['end_flags'][-1].tolist(),density_error=[me,le]));errors.append([me,le])
    diagnostics=[]
    for group in [0,1]:
        pool=[b for b in items if b['group']==group]
        for b in pool:
            others=[o for o in pool if o is not b];baselines=[]
            for o in others:
                index=np.searchsorted(o['time'],b['time'],side='right')-1;valid=b['time']<o['end'];base=np.zeros(len(b['time']))
                base[valid]=o['rtg'][index[valid]];baselines.append(base)
            baseline=np.mean(baselines,axis=0);b['adv_unscaled']=(p['gamma_native']**b['time'])*(b['rtg']-baseline)
        rms=math.sqrt(sum(float(np.mean(b['adv_unscaled']**2)) for b in pool)/4)+1e-8
        for b in pool:
            b['adv']=b['old'].new_tensor(b['adv_unscaled']/rms)
            diagnostics.append(dict(group=group,G0=float(b['rtg'][0]),peer_RMS=rms,advantage_mean=float(b['adv'].mean()),terminal=b['terminal'],density_error=b['density_error']))
    return items,diagnostics

def references(p):
    q=make(p)[0].requires_grad_(False);refs=[]
    with torch.no_grad():
        for row in p['keep_rows']:
            assert row['success'] and not row['collision'] and sha(row['path'])==row['sha256'];a=G.E.N.S.arrays(row['path']);x=torch.tensor(a['states']);n=q.normalized(x);h,_=q.core(q.encoder(n)[:,None]);h=h[:,0];target=torch.tensor(a['locations'])
            assert float((q.location(h,n)-target).abs().max())<1e-10
            key=row['job_id'];family=3 if key.startswith('t502_') else 2 if key.startswith('wide117_8m_') else 1 if key.startswith('wide117_80_') else 0
            refs.append(dict(id=key,kind='keep',family=family,features=torch.cat([h,n],-1),logits=q.head(h),target=target,source_sha256=row['sha256']))
    assert [sum(g['family']==i for g in refs) for i in range(4)]==[271,54,54,5]
    return refs

def preserve(q,refs,p):
    counts=[sum(g['family']==i for g in refs) for i in range(4)];total=0.
    for g in refs:
        loc=A.location(q,g);err=.5*((loc-g['target'])/loc.new_tensor(p['raw_sigma'])).square().mean(-1)
        loss=(.5*err.mean()+.5*err.topk(max(1,math.ceil(.2*len(err)))).values.mean())/(4*counts[g['family']]);(p['keep_weight']*loss).backward();total+=float(loss.detach())
    return total

def fit(obj,arrays,p,refs,guard,event,steps=None):
    q,v,ao,vo=obj;fixed=A.fixed_hash(q);vh=digest(v.state_dict());voh=digest(vo.state_dict());items,diagnostic=peer_batch(obj,arrays,p);targets=[]
    with torch.no_grad():targets=[q.sequence(b['x']).detach().clone() for b in items]
    trace=[]
    for k in range(p['epochs'] if steps is None else steps):
        guard();ao.zero_grad(set_to_none=True);pl=own=0.
        for b,target in zip(items,targets):
            loc=q.sequence(b['x']);ratio=(P.L.density(loc,b['raw'],b['atom'],p['raw_sigma'])-b['old']).exp();loss=-torch.minimum(ratio*b['adv'],ratio.clamp(.8,1.2)*b['adv']).mean()/8
            keep=.5*((loc-target)/loc.new_tensor(p['raw_sigma'])).square().mean()/8;(loss+p['keep_weight']*keep).backward();pl+=float(loss.detach());own+=float(keep.detach())
        keep=preserve(q,refs,p);norm=float(torch.nn.utils.clip_grad_norm_(A.variables(q),1.,error_if_nonfinite=True));event('BEFORE',k);ao.step();event('AFTER',k)
        with torch.no_grad():kl=sum(float(((z:=(P.L.density(q.sequence(b['x']),b['raw'],b['atom'],p['raw_sigma'])-b['old'])).exp()-1-z).mean()) for b in items)/8
        trace.append(dict(epoch=k,policy_loss=pl,own_keep=own,successful_history_risk_keep=keep,gradient_norm=norm,KL=kl))
        assert A.fixed_hash(q)==fixed and digest(v.state_dict())==vh and digest(vo.state_dict())==voh
        if kl>p['target_kl']:break
    n=len(trace);assert all(int(ao.state[a]['step'])==896+n for a in q.drive_adapter.parameters()) and all(int(ao.state[a]['step'])==256+n for a in q.steer_adapter.parameters())
    return dict(Actor_Adam=n,Value_Adam=0,trace=trace,peer_batch=diagnostic,trainable_parameters=11906,physical_discount_and_three_peer_time_baseline=True,not_claimed_unbiased_or_variance_reduced=True,teacher_at_deployment=False)
