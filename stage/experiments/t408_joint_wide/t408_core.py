"""Joint LEFT/RIGHT factual policy consolidation; single deployed recurrent actor."""
from pathlib import Path
import sys,os,json,time,copy,shutil,signal,subprocess,traceback,fcntl,multiprocessing as mp,math
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t406_wide_full_route'))
from t406_core import *
import t404_core as STAGE404_RUNTIME
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t408_joint_wide';DOC=ROOT/'stage/docs/review_20261107'
PARENT=ROOT/'stage/runs/t404_last_narrow/candidate.pt.gz'
def plan():return read(OUT/'protocol.json')
def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    N.capacity(OUT,p['final_bytes']);return N.science(p)
def learner(base,p):
    q=STAGE404_RUNTIME.actor(base,p,PARENT);return q,torch.optim.Adam(N.trainable(q),lr=p['lr'])
def groups(q,old_arrays,p,broad):
    old=[];left=[]
    for a in old_arrays:
        x=torch.tensor(a['states']);old.append(dict(x=x,b=q.base_actions(x),target=torch.tensor(a['locations'])))
    for f in p['left_history_files']:
        a=N.S.arrays(f);x=torch.tensor(a['states']);left.append(dict(x=x,b=q.base_actions(x),target=torch.tensor(a['locations'])))
    a=N.S.arrays(p['right_history_file']);x=torch.tensor(a['states']);right=dict(x=x,b=q.base_actions(x),target=torch.tensor(a['locations']))
    return dict(old=old,left=left,right=right,broad=broad)
def auxiliary(q,g,p,k):
    losses=[];params=N.trainable(q)
    old=g['old'][k%8];loss=(q.sequence(old['x'],old['b'])-old['target']).square().mean()*p['old_weight'];loss.backward();losses.append(float(loss.detach()))
    task=k%8;options=[x for x in g['broad'] if x['task']==task];b=options[k//8%len(options)];loss=(q.sequence(b['x'],b['b'])-b['b']).square().mean()*p['broad_old_weight'];loss.backward();losses.append(float(loss.detach()))
    before=[torch.zeros_like(v) if v.grad is None else v.grad.detach().clone() for v in params]
    left=g['left'][k%5];loss=(q.sequence(left['x'],left['b'])-left['target']).square().mean()*p['left_weight'];loss.backward();losses.append(float(loss.detach()));after=[v.grad.detach().clone() for v in params];lg=[a-b for a,b in zip(after,before)]
    right=g['right'];loss=(q.sequence(right['x'],right['b'])-right['target']).square().mean()*p['right_weight'];loss.backward();losses.append(float(loss.detach()));rg=[v.grad.detach()-a for v,a in zip(params,after)]
    ln=float(sum(v.square().sum() for v in lg).sqrt());rn=float(sum(v.square().sum() for v in rg).sqrt());dot=float(sum((a*b).sum() for a,b in zip(lg,rg)))
    return losses,dict(left_norm=ln,right_norm=rn,left_right_cosine=None if not ln*rn else dot/(ln*rn),left_history_index=k%5,old_task=task)
def consolidate(q,opt,g,p,guard,event):
    params=N.trainable(q);trace=[];components=[];fixed=digest({k:v for k,v in q.state_dict().items() if k.startswith(('base.','future_head.'))})
    for k in range(p['consolidation_steps']):
        guard();opt.zero_grad(set_to_none=True);loss,detail=auxiliary(q,g,p,k);norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));event('BEFORE',k);opt.step();event('AFTER',k);trace.append([*loss,norm]);components.append(detail)
        assert fixed==digest({n:v for n,v in q.state_dict().items() if n.startswith(('base.','future_head.'))})
    return dict(trace=trace,components=components,steps=p['consolidation_steps'],new_RL=0,left_total_weight=p['left_weight'],right_total_weight=p['right_weight'])
def rl_fit(q,opt,items,g,p,rnd,guard,event):
    batch,returns,errors=L.blocks(q,items,0,p);params=N.trainable(q);trace=[];components=[];fixed=digest({k:v for k,v in q.state_dict().items() if k.startswith(('base.','future_head.'))})
    for k in range(p['epochs']):
        guard();opt.zero_grad(set_to_none=True);pg=0.
        for x,b,raw,atom,oldlog,adv in batch:
            lr=L.logp(q.sequence(x,b),raw,atom,p)-oldlog;ratio=lr.exp();c=adv/(4*400);loss=-torch.minimum(ratio*c,ratio.clamp(1-p['ppo_clip'],1+p['ppo_clip'])*c).sum();loss.backward();pg+=float(loss.detach())
        pgvec=[v.grad.detach().clone() for v in params];pn=float(sum(v.square().sum() for v in pgvec).sqrt());loss,detail=auxiliary(q,g,p,rnd*p['epochs']+k);dg=[v.grad.detach()-a for v,a in zip(params,pgvec)];an=float(sum(v.square().sum() for v in dg).sqrt());dot=float(sum((a*b).sum() for a,b in zip(pgvec,dg)));detail.update(pg_norm=pn,aux_norm=an,pg_aux_cosine=None if not pn*an else dot/(pn*an));components.append(detail)
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));event('BEFORE',k);opt.step();event('AFTER',k);trace.append([pg,*loss,norm]);assert fixed==digest({n:v for n,v in q.state_dict().items() if n.startswith(('base.','future_head.'))})
    with torch.no_grad():mx=max(float((L.logp(q.sequence(x,b),raw,atom,p)-oldlog).abs().max()) for x,b,raw,atom,oldlog,_ in batch)
    return dict(trace=trace,gradient_components=components,returns=returns.tolist(),density_errors=errors,post_max_abs_logratio=mx,numeric_accept=bool(np.isfinite(mx) and mx<=p['max_logratio']))
def native_worker(pipe,index,p,counts):
    STAGE404_RUNTIME.OUT=OUT;STAGE404_RUNTIME.native_worker(pipe,index,p,counts)
class Pool(STAGE404_RUNTIME.Pool):
    def __init__(self,p,counts,guard):
        self.guard=guard;self.children=[];self.pipes=[];ctx=mp.get_context('spawn')
        try:
            for i in range(4):
                a,b=ctx.Pipe();child=ctx.Process(target=native_worker,args=(b,i,p,counts));child.start();b.close();self.children.append(child);self.pipes.append(a)
            save(OUT/'workers.json',dict(parent=os.getpid(),workers=[c.pid for c in self.children]));self.ready=[self.receive(i) for i in range(4)];assert all(x['state']=='READY' for x in self.ready)
        except BaseException:self.close();raise
