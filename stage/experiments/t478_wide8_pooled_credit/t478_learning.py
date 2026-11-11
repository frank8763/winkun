"""Per-side pooled complete-episode advantages plus finite function preservation."""
from pathlib import Path
import sys,math
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t477_wide8_recovery'))
import t477_runtime as B
P,E,M,R,torch,np=B.P,B.E,B.M,B.R,B.torch,B.np
read,save,sha,digest=B.read,B.save,B.sha,B.digest

def make(p):
    obj=P.make(p);P.load(obj,P.checkpoint(p['initial']['path'])['joint_state']);assert digest(P.state(*obj))==p['initial']['joint_hash'];return obj

def references(p,probe_rows=()):
    obj=make(p);q=obj[0];device=next(q.parameters()).device;old=[];own=[]
    with np.load(p['old_states']) as z:
        assert len(z['case_ids'])==211
        for i,name in enumerate(z['case_ids']):
            x=torch.tensor(z['states'][z['offsets'][i]:z['offsets'][i+1]],device=device)
            with torch.no_grad():target=q.sequence(x).detach().clone()
            old.append(dict(id=str(name),task=int(z['task'][i]),x=x,target=target))
    for row in [*p['own_reference_rows'],*[r for r in probe_rows if r['success'] and not r['collision']]]:
        assert sha(row['path'])==row['sha256'];a=E.N.S.arrays(row['path']);assert a['end_flags'][-1,1] and not a['end_flags'][:,0].any()
        x=torch.tensor(a['states'],device=device)
        with torch.no_grad():target=q.sequence(x).detach().clone()
        own.append(dict(id=row['job_id'],x=x,target=target,source_sha256=row['sha256']))
    assert len(old)==211 and len(own)>=2
    return dict(old=old,own=own,provenance=dict(old211_context_sha256=sha(p['old_states']),own=[{k:g[k] for k in ['id','source_sha256']} for g in own],target_policy_hash=q.phash()))

def keep_loss(q,refs,step):
    pools=[[g for g in refs['old'] if g['task']==t] for t in range(8)];selected=[pool[step%len(pool)] for pool in pools]
    x=torch.nn.utils.rnn.pad_sequence([g['x'] for g in selected]);loc=q.sequence(x);sd=loc.new_tensor(q.p['raw_sigma']);old=sum(.5*((loc[:len(g['x']),i]-g['target'])/sd).square().mean() for i,g in enumerate(selected))/8
    own=sum(.5*((q.sequence(g['x'])-g['target'])/sd).square().mean() for g in refs['own'])/len(refs['own'])
    return .5*(old+own),old,own

def batch(obj,arrays,p):
    q,v,ao,vo=obj;device=next(q.parameters()).device;items=[];assert len(arrays)==4
    for i,a in enumerate(arrays):
        assert not a['ownership'].any() and a['stochastic'].all() and np.array_equal(a['states'][1:],a['next_states'][:-1]);assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1
        x=torch.tensor(a['states'],device=device);raw=torch.tensor(a['raw_actions'],device=device);atom=torch.tensor(a['atom'],device=device);old=torch.tensor(a['logp'],device=device)
        with torch.no_grad():
            loc=q.sequence(x);err=float((loc-loc.new_tensor(a['locations'])).abs().max());lp=float((P.L.density(loc,raw,atom,p['raw_sigma'])-old).abs().max());assert err<1e-10 and lp<1e-9,(err,lp)
            vv=v(torch.tensor(np.concatenate([a['states'],a['next_states'][-1:]]),device=device)).cpu().numpy()
        mask=np.ones((len(x),1),bool);mask[-1]=False
        adv,target=P.L.physical_gae(a['rewards'][:,None],vv[:-1,None],vv[1:,None],a['native_counts'].astype(np.int64)[:,None],mask,mask,gamma_native=p['gamma_native'],lambda_native=1.)
        items.append(dict(x=x,raw=raw,atom=atom,old=old,adv=adv[:,0],target=old.new_tensor(target[:,0]),group=i//2,raw_adv_mean=float(adv.mean()),density_error=[err,lp]))
    diagnostics=[]
    for group in [0,1]:
        selected=[b for b in items if b['group']==group];mean=float(np.mean([b['adv'].mean() for b in selected]));second=float(np.mean([np.mean(b['adv']**2) for b in selected]));sd=math.sqrt(max(0.,second-mean*mean))+1e-8
        for b in selected:
            z=(b['adv']-mean)/sd;diagnostics.append(dict(group=group,raw_mean=b['raw_adv_mean'],normalized_mean=float(z.mean()),shared_mean=mean,shared_sd=sd));b['adv']=b['old'].new_tensor(z)
    return items,diagnostics

def fit(obj,arrays,p,refs,round_index,guard,event,steps=None):
    q,v,ao,vo=obj;items,diagnostics=batch(obj,arrays,p);trace=[]
    for k in range(p['epochs'] if steps is None else steps):
        guard();ao.zero_grad(set_to_none=True);vo.zero_grad(set_to_none=True);pg=vl=0.
        for b in items:
            ratio=(P.L.density(q.sequence(b['x']),b['raw'],b['atom'],p['raw_sigma'])-b['old']).exp();loss=-torch.minimum(ratio*b['adv'],ratio.clamp(.8,1.2)*b['adv']).mean()/4;loss.backward();pg+=float(loss.detach())
            loss=.5*(v(b['x'])-b['target']).square().mean()/4;loss.backward();vl+=float(loss.detach())
        keep,ko,kn=keep_loss(q,refs,round_index*p['epochs']+k);(p['keep_weight']*keep).backward()
        an=float(torch.nn.utils.clip_grad_norm_(q.parameters(),1.,error_if_nonfinite=True));vn=float(torch.nn.utils.clip_grad_norm_(v.parameters(),1.,error_if_nonfinite=True));event('BEFORE',k);ao.step();vo.step();event('AFTER',k)
        with torch.no_grad():kl=sum(float(((z:=(P.L.density(q.sequence(b['x']),b['raw'],b['atom'],p['raw_sigma'])-b['old'])).exp()-1-z).mean()) for b in items)/4
        trace.append(dict(epoch=k,policy_loss=pg,value_loss=vl,keep=float(keep.detach()),old_keep=float(ko.detach()),own_keep=float(kn.detach()),actor_norm=an,value_norm=vn,KL=kl))
        if kl>p['target_kl']:break
    return dict(trace=trace,Actor_Adam=len(trace),Value_Adam=len(trace),pooled_advantage=diagnostics,errors=[b['density_error'] for b in items])
