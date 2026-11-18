"""Conditional fresh-suffix clipped PG, no score term on forced prefix."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t629_offline_night_relay'))
import t629_learning as M
torch,np=M.torch,M.np


def batch(q,arrays,p):
    assert len(arrays)==4
    parts=[]
    for a in arrays:
        start=int(a['student_takeover_first']);n=a['native_counts'].astype(np.int64)
        assert 0<start<len(n) and a['ownership'][:start].all() and not a['ownership'][start:].any()
        assert not a['stochastic'][:start].any() and a['stochastic'][start:].all()
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1
        x=torch.tensor(a['states'],device='cuda');raw=torch.tensor(a['raw_actions'],device='cuda')
        atom=torch.tensor(a['atom'],device='cuda');old=torch.tensor(a['logp'],device='cuda')
        with torch.no_grad():
            loc=q.sequence(x);mask=torch.arange(len(x),device=x.device)>=start
            assert float((loc-torch.tensor(a['locations'],device=x.device)).abs().max())<1e-10
            lp=M.L.P.L.density(loc[mask],raw[mask],atom[mask],p['raw_sigma'])
            assert float((lp-old[mask]).abs().max())<1e-9
        v=0.;ret=np.zeros(len(n))
        for k in range(len(n)-1,start-1,-1):
            v=float(a['rewards'][k])+p['gamma_native']**int(n[k])*v;ret[k]=v
        t=np.r_[0,np.cumsum(n[start:])[:-1]]
        parts.append(dict(x=x,mask=mask,raw=raw[mask],atom=atom[mask],old=old[mask],
                          initial=loc[mask].detach(),time=t,end=int(n[start:].sum()),
                          rtg=ret[start:],start=start))
    for i,b in enumerate(parts):
        peers=[]
        for j,other in enumerate(parts):
            if i==j:continue
            idx=np.searchsorted(other['time'],b['time'],side='right')-1
            valid=b['time']<other['end'];values=np.zeros(len(b['time']))
            values[valid]=other['rtg'][idx[valid]];peers.append(values)
        b['adv0']=p['gamma_native']**b['time']*(b['rtg']-np.mean(peers,axis=0))
    scale=np.sqrt(np.mean([np.mean(b['adv0']**2) for b in parts]))+1e-8
    for b in parts:b['adv']=b['old'].new_tensor(b['adv0']/scale)
    return parts,dict(suffix_returns=[float(b['rtg'][0]) for b in parts],peer_RMS=float(scale),
                      forced_prefix_scores=0,full_history_memory_reconstructed=True)


def references(q,p,anchor,nominal):
    # Historical states are anchors, not asserted to be current-policy rollouts.
    refs=M.Prior.references(q,p)
    with torch.no_grad():
        for a in [anchor,nominal]:
            x=torch.tensor(a['states'],device=next(q.parameters()).device)
            refs.append(dict(x=x,mask=torch.ones(len(x),dtype=torch.bool,device=x.device),
                             target=q.sequence(x).detach()))
    assert len(refs)==35
    return refs


def objective(q,parts,refs,p):
    loss=next(q.branch.parameters()).new_zeros(());pg=0.;keep=0.
    for b in parts:
        loc=q.sequence(b['x'])[b['mask']]
        ratio=(M.L.P.L.density(loc,b['raw'],b['atom'],p['raw_sigma'])-b['old']).exp()
        value=-torch.minimum(ratio*b['adv'],ratio.clamp(.8,1.2)*b['adv']).mean()/4
        loss=loss+value;pg+=float(value.detach())
    for r in refs:
        loc=q.sequence(r['x'])[r['mask']]
        value=.5*((loc-r['target'][r['mask']])/loc.new_tensor(p['retention_sigma'])).square().mean()/len(refs)
        loss=loss+p['keep_weight']*value;keep+=float(value.detach())
    return loss,dict(policy_loss=pg,keep=keep)


def fit(obj,p,arrays,anchor,nominal,guard,event):
    q,opt=obj;parts,diag=batch(q,arrays,p);refs=references(q,p,anchor,nominal)
    body=M.L.fixed(q);trace=[]
    for k in range(p['actor_steps']):
        guard();before=M.state(obj);opt.zero_grad(set_to_none=True)
        loss,detail=objective(q,parts,refs,p);loss.backward()
        norm=float(torch.nn.utils.clip_grad_norm_(M.L.variables(q),1.,error_if_nonfinite=True))
        event('BEFORE',k);opt.step();event('AFTER',k)
        with torch.no_grad():
            kl=[float(.5*((q.sequence(b['x'])[b['mask']]-b['initial'])/b['initial'].new_tensor(p['raw_sigma'])).square().sum(-1).mean()) for b in parts]
        okay=np.isfinite(kl).all() and max(kl)<=p['max_task_latent_KL']
        if not okay:M.load(obj,before)
        trace.append(dict(step=k,loss=float(loss.detach()),gradient_norm=norm,task_latent_KL=kl,
                          accepted=bool(okay),**detail))
        if not okay:break
    assert M.L.fixed(q)==body
    return dict(attempted_optimizer=len(trace),applied=sum(t['accepted'] for t in trace),trace=trace,
                diagnostics=diag,conditional_suffix_only=True,forced_prefix_not_original_start_onpolicy=True,
                no_Critic=True,full_current_history_BPTT=True)
