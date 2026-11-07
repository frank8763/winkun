"""Wide117/80 route expansion from an exactly embedded historical wide actor."""
from pathlib import Path
import sys,os,json,time,copy,shutil,signal,subprocess,traceback,fcntl,multiprocessing as mp,math
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t404_last_narrow'))
import t404_core as C
from t404_core import *
import t404_core as STAGE404_NATIVE
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t406_wide_full_route';DOC=ROOT/'stage/docs/review_20261107'
LEGACY=ROOT/'stage/runs/t353_temporal_exploration/iid_final.pt.gz'
def plan():return read(OUT/'protocol.json')
def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    N.capacity(OUT,p['final_bytes']);return N.science(p)
def embedded(base,p):
    q=L.Actor(base,p).eval();legacy=N.S.read_model(LEGACY);q.load_learned(legacy)
    # Legacy loader merges keys. Explicitly erase the later adapter's action
    # contribution; otherwise the inherited T393 adapter silently survives.
    with torch.no_grad():q.adapter[-1].weight.zero_();q.adapter[-1].bias.zero_()
    for k,v in legacy.items():assert equal(q.learned()[k],v),k
    q.sample=False;q.reset_noise();return q
def learner(base,p):
    q=embedded(base,p);return q,torch.optim.Adam(N.trainable(q),lr=p['lr'])
def supports(q,base,p,guard):
    old=N.old_data(q,base,guard)
    for g in old:g['target']=g['b'].clone()
    wide=[]
    for f in p['wide_history_files']:
        a=N.S.arrays(f);x=torch.tensor(a['states']);wide.append(dict(x=x,b=q.base_actions(x),target=torch.tensor(a['locations']),id=Path(f).stem))
    assert len(old)==211 and len(wide)==5;return old,wide
def fit(q,opt,items,p,old,wide,rnd,guard,event):
    batch,returns,errors=L.blocks(q,items,0,p);params=N.trainable(q);trace=[];components=[];bytask={t:[g for g in old if g['task']==t] for t in range(8)}
    immutable=digest({k:v for k,v in q.state_dict().items() if k.startswith(('base.','future_head.'))})
    for k in range(p['epochs']):
        guard();opt.zero_grad(set_to_none=True);pg=0.
        for x,b,raw,atom,oldlog,adv in batch:
            lr=L.logp(q.sequence(x,b),raw,atom,p)-oldlog;ratio=lr.exp();c=adv/(4*400);loss=-torch.minimum(ratio*c,ratio.clamp(1-p['ppo_clip'],1+p['ppo_clip'])*c).sum();loss.backward();pg+=float(loss.detach())
        pggrad=[v.grad.detach().clone() for v in params];task=k%8;g=bytask[task][(rnd*p['epochs']+k)//8%len(bytask[task])]
        ol=(q.sequence(g['x'],g['b'])-g['target']).square().mean()*p['old_weight'];ol.backward()
        w=wide[(rnd*p['epochs']+k)%5];wl=(q.sequence(w['x'],w['b'])-w['target']).square().mean()*p['wide_weight'];wl.backward()
        dg=[v.grad.detach()-a for v,a in zip(params,pggrad)];pn=float(sum(a.square().sum() for a in pggrad).sqrt());dn=float(sum(a.square().sum() for a in dg).sqrt());dot=float(sum((a*b).sum() for a,b in zip(pggrad,dg)))
        components.append(dict(pg_norm=pn,old_plus_wide_increment_norm=dn,cosine=None if not pn*dn else dot/(pn*dn),old_task=task,old_group=g['id'],wide_group=w['id']))
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));event('BEFORE',k);opt.step();event('AFTER',k);trace.append([pg,float(ol.detach()),float(wl.detach()),norm])
        assert digest({n:v for n,v in q.state_dict().items() if n.startswith(('base.','future_head.'))})==immutable
    with torch.no_grad():mx=max(float((L.logp(q.sequence(x,b),raw,atom,p)-oldlog).abs().max()) for x,b,raw,atom,oldlog,_ in batch)
    return dict(returns=returns.tolist(),density_errors=errors,trace=trace,gradient_components=components,post_max_abs_logratio=mx,numeric_accept=bool(np.isfinite(mx) and mx<=p['max_logratio']))
def native_worker(pipe,index,p,counts):
    STAGE404_NATIVE.OUT=OUT;STAGE404_NATIVE.native_worker(pipe,index,p,counts)
class Pool(STAGE404_NATIVE.Pool):
    def __init__(self,p,counts,guard):
        self.guard=guard;self.children=[];self.pipes=[];ctx=mp.get_context('spawn')
        try:
            for i in range(4):
                a,b=ctx.Pipe();child=ctx.Process(target=native_worker,args=(b,i,p,counts));child.start();b.close();self.children.append(child);self.pipes.append(a)
            save(OUT/'workers.json',dict(parent=os.getpid(),workers=[c.pid for c in self.children]));self.ready=[self.receive(i) for i in range(4)];assert all(x['state']=='READY' for x in self.ready)
        except BaseException:self.close();raise
def tally(rows):return dict(valid=len(rows),success=sum(r['success'] for r in rows),contact=sum(r['collision'] for r in rows),timeout=sum(r['timeout'] for r in rows),invalid=0,censored=0)
