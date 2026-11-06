"""Two finite current-student correction cycles; deployment remains one Actor."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t395_frontier_recovery'))
from t395_learning import *
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t396_iterative_recovery'

def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    N.capacity(OUT,p['final_bytes'])
    assert N.K.S.used(ROOT/'stage/runs')-N.K.S.used(OUT)+p['final_bytes']<=1024**3
    return N.science(p)

def groups(q,p,corrections):
    gs=datasets(q)
    for g in gs:g.update(weight=.25,stop=len(g['x']))
    for correction in corrections:
        a=N.S.arrays(correction);first=p['teacher_start_hold'];stop=first+p['bridge_holds']
        assert len(a['states'])>=stop and a['end_flags'][-1,1] and not a['end_flags'][:,0].any()
        assert np.all(a['ownership'][first:stop]==1)
        x=torch.tensor(a['states']);gs.append(dict(x=x,b=q.base_actions(x),y=torch.tensor(a['actions'],dtype=torch.float64),first=first,stop=stop,weight=.5/len(corrections)))
    return gs

def fit(q,opt,items,first,p,refs,guard,event):
    batch,returns,errors=blocks(q,items,first,p);trace=[];components=[]
    params=[v for v in q.parameters() if v.requires_grad]
    fixed=digest({k:v for k,v in q.state_dict().items() if k.startswith(('base.','future_head.'))})
    for k in range(p['epochs']):
        guard();opt.zero_grad(set_to_none=True);pg=aux=0.;maxlog=0.
        for x,b,raw,atom,old,adv in batch:
            lr=(logp(q.sequence(x,b),raw,atom,p)-old)[first:];ratio=lr.exp()
            c=adv/(4*400);loss=-torch.minimum(ratio*c,ratio.clamp(1-p['ppo_clip'],1+p['ppo_clip'])*c).sum()
            loss.backward();pg+=float(loss.detach());maxlog=max(maxlog,float(lr.detach().abs().max()))
        pggrad=[v.grad.detach().clone() for v in params]
        for g in refs:
            pred=q.sequence(g['x'],g['b']);sl=slice(g['first'],g['stop'])
            loss=p['demo_weight']*g['weight']*(pred[sl]-g['y'][sl]).square().mean()
            loss.backward();aux+=float(loss.detach())
        dg=[v.grad.detach()-a for v,a in zip(params,pggrad)]
        pn=float(sum(a.square().sum() for a in pggrad).sqrt());dn=float(sum(a.square().sum() for a in dg).sqrt())
        dot=float(sum((a*b).sum() for a,b in zip(pggrad,dg)))
        components.append(dict(pg_norm=pn,aux_increment_norm=dn,cosine=None if pn*dn==0 else dot/(pn*dn)))
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
        event('BEFORE',k);opt.step();event('AFTER',k)
        assert digest({n:v for n,v in q.state_dict().items() if n.startswith(('base.','future_head.'))})==fixed
        trace.append([pg,aux,norm,maxlog])
    with torch.no_grad():mx=max(float((logp(q.sequence(x,b),raw,atom,p)-old)[first:].abs().max()) for x,b,raw,atom,old,_ in batch)
    return dict(trace=trace,returns=returns.tolist(),density_errors=errors,gradient_components=components,post_max_abs_logratio=mx,numeric_accept=bool(np.isfinite(mx) and mx<=p['max_logratio']))

def teacher_controls(theta,reference,p):
    h=400-p['teacher_start_hold'];axis=np.arange(h);knots=np.linspace(0,h-1,16)
    v=np.stack([np.interp(axis,knots,theta[:,j]) for j in range(2)],1)
    phase=np.clip(np.maximum.accumulate(p['teacher_start_hold']+axis+5*np.clip(v[:,1],-4,4)),0,len(reference)-1)
    u=np.stack([np.interp(phase,np.arange(len(reference)),reference[:,j]) for j in range(2)],1);u[:,0]+=v[:,0]
    return np.clip(u,-1,1).astype(np.float32)

def teacher_candidates(mean,sd,best,cycle,generation,p):
    z=np.random.default_rng(np.random.SeedSequence([p['seed'],cycle,generation,777])).normal(size=(5,16,2))
    choices=[mean.copy(),np.zeros((16,2)) if best is None else best.copy()]
    for v in z:choices.extend([mean+v*sd,mean-v*sd])
    return np.asarray(choices)

def objective(u,n,flags,remaining):
    contact,goal,timeout=map(bool,flags)
    return 1000.*contact+500.*timeout-1000.*(goal and not contact)+10.*remaining+.01*float(np.square(np.diff(u[:n].astype(float),axis=0)).sum())

def teacher_buffer(p):
    n,h=24,400-p['teacher_start_hold']
    return dict(coefficients=np.zeros((n,16,2)),controls=np.zeros((n,h,2),np.float32),next_poses=np.full((n,h,3),np.nan),counts=np.zeros((n,h),np.uint8),flags=np.zeros((n,h,3),np.uint8),hold_sha256=np.zeros((n,h,32),np.uint8),lengths=np.zeros(n,np.uint16),costs=np.full(n,np.nan),remaining_m=np.full(n,np.nan),final_integration=np.full((n,203),np.nan))
