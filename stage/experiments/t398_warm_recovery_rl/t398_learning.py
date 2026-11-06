"""Finite short-recovery warmup, then separately collected on-policy PPO."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t396_iterative_recovery'))
from t396_learning import *
import t396_learning as ORIGINAL
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t398_warm_recovery_rl'
KNOWN_BRIDGE=ROOT/'stage/runs/t397_handoff_duration/episodes/duration20.npz'

def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    N.capacity(OUT,p['final_bytes'])
    assert N.K.S.used(ROOT/'stage/runs')-N.K.S.used(OUT)+p['final_bytes']<=1024**3
    return N.science(p)

def old_references(q,base,guard):
    # Historical T255 visited observations are not a recovery expert. Preserve
    # the selected T393 parent's raw location, rather than clipped targets.
    old=N.old_data(q,base,guard)
    with torch.no_grad():
        for g in old:g.update(target=q.sequence(g['x'],g['b']).detach().clone(),weight=1.)
    assert len(old)==211
    return old

def groups(q,p,corrections):return ORIGINAL.groups(q,p,corrections)

def warm_fit(q,opt,refs,old,p,guard,event,steps=None):
    params=[v for v in q.parameters() if v.requires_grad];trace=[]
    fixed=digest({k:v for k,v in q.state_dict().items() if k.startswith(('base.','future_head.'))})
    for k in range(p['warm_steps'] if steps is None else steps):
        guard();opt.zero_grad(set_to_none=True);losses=[]
        for g in refs:
            pred=q.sequence(g['x'],g['b']);sl=slice(g['first'],g['stop']);loss=p['demo_weight']*g['weight']*(pred[sl]-g['y'][sl]).square().mean()
            loss.backward();losses.append(float(loss.detach()))
        g=old[k%len(old)];loss=p['old_weight']*(q.sequence(g['x'],g['b'])-g['target']).square().mean();loss.backward();losses.append(float(loss.detach()))
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));event('BEFORE',k);opt.step();event('AFTER',k)
        assert digest({n:v for n,v in q.state_dict().items() if n.startswith(('base.','future_head.'))})==fixed
        trace.append([*losses,norm])
    return dict(trace=trace,steps=len(trace),optimizer_updates=len(trace),old_reference_schedule='fixed round robin, first200 of211 during full warmup',new_RL=0)

def fit(q,opt,items,first,p,refs,old,guard,event):
    batch,returns,errors=blocks(q,items,first,p);trace=[];components=[]
    params=[v for v in q.parameters() if v.requires_grad]
    fixed=digest({k:v for k,v in q.state_dict().items() if k.startswith(('base.','future_head.'))})
    for k in range(p['epochs']):
        guard();opt.zero_grad(set_to_none=True);pg=aux=0.;maxlog=0.
        for x,b,raw,atom,oldlog,adv in batch:
            lr=(logp(q.sequence(x,b),raw,atom,p)-oldlog)[first:];ratio=lr.exp();c=adv/(4*400)
            loss=-torch.minimum(ratio*c,ratio.clamp(1-p['ppo_clip'],1+p['ppo_clip'])*c).sum();loss.backward();pg+=float(loss.detach());maxlog=max(maxlog,float(lr.detach().abs().max()))
        pggrad=[v.grad.detach().clone() for v in params]
        for g in refs:
            pred=q.sequence(g['x'],g['b']);sl=slice(g['first'],g['stop']);loss=p['demo_weight']*g['weight']*(pred[sl]-g['y'][sl]).square().mean();loss.backward();aux+=float(loss.detach())
        g=old[k%len(old)];keep=p['old_weight']*(q.sequence(g['x'],g['b'])-g['target']).square().mean();keep.backward()
        dg=[v.grad.detach()-a for v,a in zip(params,pggrad)]
        pn=float(sum(a.square().sum() for a in pggrad).sqrt());dn=float(sum(a.square().sum() for a in dg).sqrt())
        dot=float(sum((a*b).sum() for a,b in zip(pggrad,dg)))
        components.append(dict(pg_norm=pn,combined_aux_old_increment_norm=dn,cosine=None if pn*dn==0 else dot/(pn*dn)))
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));event('BEFORE',k);opt.step();event('AFTER',k)
        assert digest({n:v for n,v in q.state_dict().items() if n.startswith(('base.','future_head.'))})==fixed
        trace.append([pg,aux,float(keep.detach()),norm,maxlog])
    with torch.no_grad():mx=max(float((logp(q.sequence(x,b),raw,atom,p)-oldlog)[first:].abs().max()) for x,b,raw,atom,oldlog,_ in batch)
    return dict(trace=trace,returns=returns.tolist(),density_errors=errors,gradient_components=components,post_max_abs_logratio=mx,numeric_accept=bool(np.isfinite(mx) and mx<=p['max_logratio']))
