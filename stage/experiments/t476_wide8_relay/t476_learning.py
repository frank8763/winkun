"""Recurrent PPO on complete own-policy episodes; physical time and exact atoms."""
from pathlib import Path
import sys,copy,math
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t449_offline_joint_recovery'))
import run as M
P,E,R,torch,np=M.P,M.E,M.R,M.torch,M.np
read,save,sha,digest=P.read,P.save,P.sha,P.digest

def make(p):
    obj=P.make(p);saved=P.checkpoint(p['initial']['path'])['joint_state']
    # Same Actor and V in all arms; fresh identical optimizers explicitly.
    obj[0].load_state_dict(saved['actor']);obj[1].load_state_dict(saved['value'])
    for g in obj[2].param_groups:g['lr']=p['actor_lr']
    return obj

def fit(obj,arrays,p,guard,event):
    q,v,ao,vo=obj;device=next(q.parameters()).device;batch=[]
    assert len(arrays)==2
    for group,a in enumerate(arrays):
        assert not a['ownership'].any() and a['stochastic'].all()
        assert np.array_equal(a['states'][1:],a['next_states'][:-1])
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1
        x=torch.tensor(a['states'],device=device);raw=torch.tensor(a['raw_actions'],device=device);atom=torch.tensor(a['atom'],device=device);old=torch.tensor(a['logp'],device=device)
        with torch.no_grad():
            loc=q.sequence(x);err=float((loc-loc.new_tensor(a['locations'])).abs().max());lp=float((P.L.density(loc,raw,atom,p['raw_sigma'])-old).abs().max())
            assert err<1e-10 and lp<1e-9,(err,lp)
            vv=v(torch.tensor(np.concatenate([a['states'],a['next_states'][-1:]]),device=device)).cpu().numpy()
        mask=np.ones((len(x),1),bool);mask[-1]=False
        adv,target=P.L.physical_gae(a['rewards'][:,None],vv[:-1,None],vv[1:,None],a['native_counts'].astype(np.int64)[:,None],mask,mask,gamma_native=p['gamma_native'],lambda_native=p['lambda_native'])
        z=adv[:,0];z=(z-z.mean())/(z.std()+1e-8)
        batch.append(dict(x=x,raw=raw,atom=atom,old=old,adv=old.new_tensor(z),target=old.new_tensor(target[:,0]),density_error=[err,lp]))
    trace=[]
    for k in range(p['epochs']):
        guard();ao.zero_grad(set_to_none=True);vo.zero_grad(set_to_none=True);pl=vl=0.
        for b in batch:
            ratio=(P.L.density(q.sequence(b['x']),b['raw'],b['atom'],p['raw_sigma'])-b['old']).exp()
            loss=-torch.minimum(ratio*b['adv'],ratio.clamp(.8,1.2)*b['adv']).mean()/len(batch);loss.backward();pl+=float(loss.detach())
            loss=.5*(v(b['x'])-b['target']).square().mean()/len(batch);loss.backward();vl+=float(loss.detach())
        an=float(torch.nn.utils.clip_grad_norm_(q.parameters(),1.,error_if_nonfinite=True));vn=float(torch.nn.utils.clip_grad_norm_(v.parameters(),1.,error_if_nonfinite=True))
        event('BEFORE',k);ao.step();vo.step();event('AFTER',k)
        with torch.no_grad():kl=sum(float(((z:=(P.L.density(q.sequence(b['x']),b['raw'],b['atom'],p['raw_sigma'])-b['old'])).exp()-1-z).mean()) for b in batch)/len(batch)
        trace.append(dict(epoch=k,policy_loss=pl,value_loss=vl,actor_norm=an,value_norm=vn,KL=kl))
        if kl>p['target_kl']:break
    return dict(trace=trace,Actor_Adam=len(trace),Value_Adam=len(trace),errors=[b['density_error'] for b in batch])
