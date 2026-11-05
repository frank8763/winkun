"""Fresh on-policy centre PPO plus explicitly historical supervised rehearsal."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t365_correction_aggregation'))
from t365_core import *
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t367_factual_rehearsal_rl';DOC=ROOT/'stage/docs/narrow_20261105'


def references(p,q):
    old,_=P.C.dataset(dict(p,sequence_sources=[]),q,lambda:None);humans=[];wide=[];narrow=[]
    for name,path,first in [('T361',p['demonstration'],0),('T364',p['correction'],20)]:
        a=arrays(path);x=torch.tensor(a['states']);humans.append(dict(x=x,b=q.base_actions(x),y=torch.tensor(a['actions'],dtype=torch.float64),first_label=first,id=name))
    for r in p['factual_rehearsal']:
        assert K.C.sha(r['path'])==r['sha256'];a=arrays(r['path']);x=torch.tensor(a['states'])
        g=dict(x=x,b=q.base_actions(x),y=torch.tensor(a['locations'],dtype=torch.float64),id=r['job_id'],source_model=r['source_model'])
        (wide if r['job_id'].startswith('wide') else narrow).append(g)
    assert len(old)==211 and len(wide)==5 and len(narrow)==2 and len(humans[1]['x'])-humans[1]['first_label']==277
    return old,humans,narrow,wide


def optimize(p,obj,items,refs,rnd,guard,event):
    assert len(items)==4
    returns=[float(reward_to_go(it['arrays'])[0]) for it in items];blocks=[]
    for j,it in enumerate(items):
        a=it['arrays'];x=torch.tensor(a['states']);b=obj.actor.base_actions(x);raw=torch.tensor(a['raw_actions']);before=torch.tensor(a['logp'])
        with torch.no_grad():
            loc=obj.actor.sequence(x,b);lp=raw_logp(loc,raw,0.,p['raw_sigma'])
            assert float((loc-torch.tensor(a['locations'])).abs().max())<1e-10
            assert float((lp-before).abs().max())<1e-9
        advantage=returns[j]-(sum(returns)-returns[j])/3
        blocks.append((x,b,raw,before,advantage/(4*400)))
    old,humans,narrow,wide=refs;trace=[]
    for k in range(p['epochs']):
        guard();obj.opt.zero_grad(set_to_none=True);pg=0.;maxlog=0.;losses=dict(old=0.,human=0.,wide=0.,narrow=0.)
        for x,b,raw,before,c in blocks:
            delta=raw_logp(obj.actor.sequence(x,b),raw,0.,p['raw_sigma'])-before;ratio=delta.exp();assert torch.isfinite(ratio).all()
            loss=-torch.minimum(ratio*c,ratio.clamp(.9,1.1)*c).sum();assert torch.isfinite(loss);loss.backward();pg+=float(loss.detach());maxlog=max(maxlog,float(delta.detach().abs().max()))
        task=(rnd*p['epochs']+k)%8;og=next(g for g in old if g['task']==task)
        choices=[('old',og,1.)]+[('wide',g,.2) for g in wide]+[('narrow',g,.5) for g in narrow]
        for name,g,weight in choices:
            loss=weight*(obj.actor.sequence(g['x'],g['b'])-g['y']).square().mean();assert torch.isfinite(loss);loss.backward();losses[name]+=float(loss.detach())
        for g in humans:
            loss=.5*labeled_loss(obj.actor.sequence(g['x'],g['b']),g);assert torch.isfinite(loss);loss.backward();losses['human']+=float(loss.detach())
        norm=float(torch.nn.utils.clip_grad_norm_(obj.actor.trainable(),1.,error_if_nonfinite=True));event(dict(phase='BEFORE',step=k))
        obj.opt.step();obj.calls+=1;event(dict(phase='AFTER',step=k));assert all(v.grad is None for v in obj.actor.base.parameters())
        trace.append(dict(step=k,pg=pg,supervised_components=losses,gradient_norm=norm,max_abs_logratio=maxlog,old_task=task))
    return dict(trace=trace,start_returns=returns)
