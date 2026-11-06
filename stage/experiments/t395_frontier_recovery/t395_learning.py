"""Same recurrent estimator, fixed one-second successful-frontier curriculum."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t393_safe_demo_pg'))
from t393_learning import *
import t393_learning as C
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t395_frontier_recovery'
PRIOR=ROOT/'stage/runs/t393_safe_demo_pg';PARENT=PRIOR/'candidate.pt.gz'

class Actor(C.Actor):
    def __init__(self,base,p):
        super().__init__(base,p);self.load_learned(N.S.read_model(PARENT));self.reset_noise()

def learner(base,p):
    q=Actor(base,p).eval();params=N.trainable(q);assert sum(v.numel() for v in params)==6864
    return q,torch.optim.Adam(params,lr=p['lr'])

def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    N.capacity(OUT,p['final_bytes'])
    assert N.K.S.used(ROOT/'stage/runs')-N.K.S.used(OUT)+p['final_bytes']<=1024**3
    return N.science(p)

def check_stop():C.check_stop()
def install_stop_handlers():return C.install_stop_handlers()
def clear_stop():C.STOP_REQUESTED=None

def fit(q,opt,items,first,p,refs,guard,event):
    batch,returns,errors=blocks(q,items,first,p);trace=[];components=[]
    params=[v for v in q.parameters() if v.requires_grad]
    fixed=digest({k:v for k,v in q.state_dict().items() if k.startswith(('base.','future_head.'))})
    for k in range(p['epochs']):
        guard();opt.zero_grad(set_to_none=True);pg=aux=0.;maxlog=0.
        for x,b,raw,atom,old,adv in batch:
            lr=(logp(q.sequence(x,b),raw,atom,p)-old)[first:];ratio=lr.exp();assert torch.isfinite(ratio).all()
            c=adv/(4*400)
            loss=-torch.minimum(ratio*c,ratio.clamp(1-p['ppo_clip'],1+p['ppo_clip'])*c).sum()
            loss.backward();pg+=float(loss.detach());maxlog=max(maxlog,float(lr.detach().abs().max()))
        pggrad=[v.grad.detach().clone() for v in params]
        for g in refs:
            pred=q.sequence(g['x'],g['b']);loss=.5*p['demo_weight']*(pred[g['first']:]-g['y'][g['first']:]).square().mean()
            loss.backward();aux+=float(loss.detach())
        # These are the actually accumulated raw gradient components before
        # norm clipping and Adam, not loss magnitudes or causal step attribution.
        demograd=[v.grad.detach()-g for v,g in zip(params,pggrad)]
        pnorm=float(sum(g.square().sum() for g in pggrad).sqrt());dnorm=float(sum(g.square().sum() for g in demograd).sqrt())
        dot=float(sum((a*b).sum() for a,b in zip(pggrad,demograd)))
        components.append(dict(pg_norm=pnorm,demo_accumulation_increment_norm=dnorm,
            cosine=None if pnorm*dnorm==0 else dot/(pnorm*dnorm)))
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
        event('BEFORE',k);opt.step();event('AFTER',k)
        assert digest({n:v for n,v in q.state_dict().items() if n.startswith(('base.','future_head.'))})==fixed
        trace.append([pg,aux,norm,maxlog])
    with torch.no_grad():
        maxlog=max(float((logp(q.sequence(x,b),raw,atom,p)-old)[first:].abs().max()) for x,b,raw,atom,old,_ in batch)
    return dict(trace=trace,returns=returns.tolist(),density_errors=errors,gradient_components=components,
        post_max_abs_logratio=maxlog,numeric_accept=bool(np.isfinite(maxlog) and maxlog<=p['max_logratio']))
