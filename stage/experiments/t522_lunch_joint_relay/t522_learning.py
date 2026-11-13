"""Own-policy clipped policy gradients with physical-time peer RTG baseline.

Critic-free finite pilot. Full episodes are sampled from the actual Actor;
no successful-history supervision is passed as on-policy actions.
"""
from pathlib import Path
import math,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t501_joint_command_repair'));import t501_learning as A
sys.path.insert(0,str(ROOT/'stage/experiments/t504_success_replay'));import t504_learning as B
G,P,torch,np=A.G,A.P,A.torch,A.np
read,sha,digest=G.read,G.sha,G.digest

def variables(q):return list(q.parameters())
def make(p):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    old=P.checkpoint(p['initial']['path'])['joint_state'];q=A.Actor(p);q.load_state_dict(old['actor']);assert q.phash()==p['initial']['policy_hash'];q.requires_grad_(True)
    v=P.L.Value(q,p['value_seed']);v.load_state_dict(old['value']);v.requires_grad_(False)
    ao=torch.optim.Adam(A.variables(q),lr=p['actor_lr'],foreach=False,fused=False)
    expanded=len(old['actor_opt']['param_groups'])==2
    if not expanded:ao.load_state_dict(old['actor_opt'])
    adapter_ids={id(z) for z in A.variables(q)};new=[z for z in q.parameters() if id(z) not in adapter_ids]
    ao.add_param_group(dict(params=new,lr=p['actor_lr']))
    if expanded:ao.load_state_dict(old['actor_opt'])
    for g in ao.param_groups:g['lr']=p['actor_lr']
    vo=torch.optim.Adam(v.parameters(),lr=p['value_lr'],foreach=False,fused=False);vo.load_state_dict(old['value_opt'])
    assert not ({id(z) for z in v.parameters()}&{id(z) for z in q.parameters()})
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
    for group in range(len(items)//4):
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
            assert row['success'] and not row['collision'] and sha(row['path'])==row['sha256']
            a=G.E.N.S.arrays(row['path']);assert all(digest(v)==row['array_hashes'][k] for k,v in a.items())
            g=B.cached(q,a);target=g['target'].detach().clone()
            g.update(x=torch.tensor(a['states']),id=row['job_id'],kind='keep',family=int(row['job_id'].startswith('t502_')),source_sha256=row['sha256'],target=target,reference_is_current_function_on_historical_inputs_NOT_actual_new_success=True);refs.append(g)
    assert [sum(g['family']==i for g in refs) for i in range(2)]==[8,11]
    return refs

def preserve(q,refs,p):
    counts=[sum(g['family']==i for g in refs) for i in range(2)];total=0.
    for g in refs:
        loc=q.sequence(g['x']);err=.5*((loc-g['target'])/loc.new_tensor(p['raw_sigma'])).square().mean(-1)
        loss=(.5*err.mean()+.5*err.topk(max(1,math.ceil(.2*len(err)))).values.mean())/(2*counts[g['family']]);(p['keep_weight']*loss).backward();total+=float(loss.detach())
    return total

def pg_fit(obj,arrays,p,refs,guard,event,steps=None):
    q,v,ao,vo=obj;vh=digest(v.state_dict());voh=digest(vo.state_dict());items,diagnostic=peer_batch(obj,arrays,p);targets=[]
    with torch.no_grad():targets=[q.sequence(b['x']).detach().clone() for b in items]
    trace=[];previous_steps=[int(ao.state[z]['step']) if z in ao.state else 0 for z in variables(q)]
    for k in range(p['epochs'] if steps is None else steps):
        guard();ao.zero_grad(set_to_none=True);pl=own=0.
        for b,target in zip(items,targets):
            loc=q.sequence(b['x']);ratio=(P.L.density(loc,b['raw'],b['atom'],p['raw_sigma'])-b['old']).exp();loss=-torch.minimum(ratio*b['adv'],ratio.clamp(.8,1.2)*b['adv']).mean()/len(items)
            keep=.5*((loc-target)/loc.new_tensor(p['raw_sigma'])).square().mean()/len(items);(loss+p['keep_weight']*keep).backward();pl+=float(loss.detach());own+=float(keep.detach())
        keep=preserve(q,refs,p);norm=float(torch.nn.utils.clip_grad_norm_(variables(q),1.,error_if_nonfinite=True));event('BEFORE',k);ao.step();event('AFTER',k)
        with torch.no_grad():kl=sum(float(((z:=(P.L.density(q.sequence(b['x']),b['raw'],b['atom'],p['raw_sigma'])-b['old'])).exp()-1-z).mean()) for b in items)/len(items)
        trace.append(dict(epoch=k,policy_loss=pl,own_keep=own,successful_history_risk_keep=keep,gradient_norm=norm,KL=kl))
        assert digest(v.state_dict())==vh and digest(vo.state_dict())==voh
        if kl>p['target_kl']:break
    n=len(trace);assert [int(ao.state[z]['step']) for z in variables(q)]==[s+n for s in previous_steps]
    return dict(Actor_Adam=n,Value_Adam=0,trace=trace,peer_batch=diagnostic,trainable_parameters=sum(z.numel() for z in variables(q)),full_actor_trainable=True,physical_discount_and_three_peer_time_baseline=True,not_claimed_unbiased_or_variance_reduced=True,teacher_at_deployment=False)


def fit(obj,arrays,p,refs,guard,event,steps=None):
    if p['mechanism']=='joint_pg':return pg_fit(obj,arrays,p,refs,guard,event,steps)
    assert p['mechanism']=='success_bc' and len(arrays)==2
    q,v,ao,vo=obj;vh=digest(v.state_dict());voh=digest(vo.state_dict());trace=[]
    data=[]
    for a in arrays:
        assert not a['ownership'].any() and not a['stochastic'].any() and a['end_flags'][-1].tolist()==[0,1,0]
        data.append((torch.tensor(a['states']),torch.tensor(a['locations'])))
    previous=[int(ao.state[z]['step']) for z in variables(q)]
    for k in range(p['epochs'] if steps is None else steps):
        guard();ao.zero_grad(set_to_none=True);total=0.
        for x,target in data:
            loc=q.sequence(x);err=.5*((loc-target)/loc.new_tensor(p['raw_sigma'])).square().mean(-1)
            loss=(.5*err.mean()+.5*err.topk(max(1,math.ceil(.2*len(err)))).values.mean())/2
            loss.backward();total+=float(loss.detach())
        keep=preserve(q,refs,p);norm=float(torch.nn.utils.clip_grad_norm_(variables(q),1.,error_if_nonfinite=True))
        event('BEFORE',k);ao.step();event('AFTER',k)
        trace.append(dict(epoch=k,balanced_success_location_loss=total,history_keep=keep,gradient_norm=norm))
        assert digest(v.state_dict())==vh and digest(vo.state_dict())==voh
    n=len(trace);assert [int(ao.state[z]['step']) for z in variables(q)]==[s+n for s in previous]
    return dict(Actor_Adam=n,Value_Adam=0,new_policy_gradient_updates=0,trace=trace,offpolicy_two_actual_autonomous_successes=True,full_history_BPTT=True,teacher_at_deployment=False)
