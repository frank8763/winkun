"""Demo-supported on-policy acquisition; standard-method engineering pilot."""
from pathlib import Path
import sys
_ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(_ROOT/'stage/experiments/t339_success_to_control'))
from t339_core import *
ROOT=_ROOT;HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t340_demo_supported_recovery'

def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    assert K.S.used(OUT)<=p['final_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']

def demos(p,q,success_items=()):
    groups=[]
    for record in p['demo_sources']:
        a=arrays(record['path']);sl=slice(record['start'],record['stop'])
        groups.append((a['states'][sl],a['actions'][sl],record['name']))
    for item in success_items:
        assert item['row']['success'] and not item['row']['collision'] and item['row']['complete']
        a=item['arrays'];groups.append((a['states'],a['actions'],item['row']['id']))
    result=[]
    for x,y,name in groups:
        assert len(x)>0 and x.shape[1]==120 and y.shape==(len(x),2)
        xx=torch.tensor(x);result.append(dict(x=xx,y=torch.tensor(y,dtype=torch.float64),cache=q.cache(xx),name=name))
    return result

def demo_backward(q,groups,p,rng,weight):
    lossvalue=0.
    for data in groups:
        idx=rng.integers(len(data['x']),size=p['demo_rows_per_source'])
        loc,_=q.stats(data['x'][idx],tuple(v[idx] for v in data['cache']))
        # Regression in the unclipped location coordinates avoids zero gradient
        # outside an executed endpoint. Targets remain real executed actions.
        scale=torch.tensor(p['executed_std'],dtype=torch.float64)
        loss=((loc-data['y'][idx])/scale).square().mean()*weight/len(groups)
        assert torch.isfinite(loss);loss.backward();lossvalue+=float(loss.detach())
    return lossvalue

def warmup(p,q,opt,groups,guard,event,steps=None):
    rng=np.random.default_rng(p['seed']+4000);trace=[]
    for k in range(p['warm_steps'] if steps is None else steps):
        guard();opt.zero_grad(set_to_none=True);loss=demo_backward(q,groups,p,rng,1.)
        norm=float(torch.nn.utils.clip_grad_norm_(q.trainable(),1.,error_if_nonfinite=True))
        event(dict(stage='BEFORE_ADAM',step=k));opt.step()
        assert all(v.grad is None for v in q.fixed.parameters())
        row=dict(step=k,loss=loss,gradient_norm=norm,policy_hash=digest(q.state_dict()),optimizer_hash=digest(opt.state_dict()))
        trace.append(row);event(dict(stage='AFTER_ADAM',**row))
    return trace

def supported_update(p,learner,items,groups,rnd,guard,event,steps=None):
    assert len(items)==4;data=[v['arrays'] for v in items];ret=[returns(a) for a in data]
    horizon=max(map(len,ret));rms=max(.05,float(np.sqrt(np.mean(np.concatenate(ret)**2))))
    table=np.zeros((4,horizon))
    for j,r in enumerate(ret):table[j,:len(r)]=r
    blocks=[]
    for j,(a,r) in enumerate(zip(data,ret)):
        assert a['end_flags'][-1].any() and not a['end_flags'][:-1].any() and (a['native_counts'][:-1]==100).all()
        x=torch.tensor(a['states']);cache=learner.actor.cache(x);raw=torch.tensor(a['raw_actions']);atom=torch.tensor(a['atom']);old=torch.tensor(a['logp'])
        with torch.no_grad():
            loc,sd=learner.actor.stats(x,cache);lp=learner.actor.logp(loc,sd,raw,atom)
            assert torch.equal(lp,old) and np.array_equal(loc.numpy(),a['locations']) and np.array_equal(sd.numpy(),a['sigmas'])
        baseline=(table.sum(0)-table[j])/3
        discount=K.R.GAMMA**np.r_[0,np.cumsum(a['native_counts'].astype(np.int64))[:-1]]
        coeff=torch.tensor(discount*(r-baseline[:len(r)])/rms/(4*400))
        blocks.append((x,cache,raw,atom,old,coeff))
    weight=p['demo_rl_weight']*p['demo_decay']**rnd;rng=np.random.default_rng(p['seed']+5000+rnd);trace=[]
    for step in range(p['epochs'] if steps is None else steps):
        guard();learner.opt.zero_grad(set_to_none=True);pg=0.;maxratio=0.
        for x,cache,raw,atom,old,c in blocks:
            loc,sd=learner.actor.stats(x,cache);lr=learner.actor.logp(loc,sd,raw,atom)-old
            ratio=lr.exp();assert torch.isfinite(ratio).all()
            loss=-torch.minimum(ratio*c,ratio.clamp(1-p['ppo_clip'],1+p['ppo_clip'])*c).sum();loss.backward()
            pg+=float(loss.detach());maxratio=max(maxratio,float(lr.detach().abs().max()))
        imitation=demo_backward(learner.actor,groups,p,rng,weight)
        norm=float(torch.nn.utils.clip_grad_norm_(learner.actor.trainable(),1.,error_if_nonfinite=True))
        event(dict(stage='BEFORE_ADAM',step=step));learner.opt.step();learner.calls+=1
        assert all(v.grad is None for v in learner.actor.fixed.parameters())
        row=dict(step=step,pg_loss=pg,imitation_loss=imitation,demo_weight=weight,gradient_norm=norm,max_abs_logratio=maxratio,hashes=learner.hashes())
        trace.append(row);event(dict(stage='AFTER_ADAM',**row))
    return trace
