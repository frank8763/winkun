"""Aggregate real-human labels; retain but never supervise the learner prefix."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t358_direct_narrow'))
from t358_core import *
import t358_core as PREVIOUS
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t365_correction_aggregation';DOC=ROOT/'stage/docs/narrow_20261105'


def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    assert K.S.used(OUT)<=p['final_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']


def dataset(p,q,guard):
    old,_,supports=PREVIOUS.references(p,q);demos=[]
    for name,path,sha,n,first in [('prior_human',p['demonstration'],p['demonstration_sha256'],295,0),
                                ('learner_state_correction',p['correction'],p['correction_sha256'],297,20)]:
        assert K.C.sha(path)==sha
        a=arrays(path);x=torch.tensor(a['states']);b=q.base_actions(x)
        assert len(x)==n and a['end_flags'][-1,1] and not a['end_flags'][:,0].any() and not a['end_flags'][:-1].any()
        assert np.array_equal(a['states'][1:],a['next_states'][:-1]);guard()
        demos.append(dict(name=name,x=x,b=b,y=torch.tensor(a['actions'],dtype=torch.float64),first_label=first))
    assert np.array_equal(demos[0]['x'][0].numpy(),demos[1]['x'][0].numpy())
    assert sum(len(g['x'])-g['first_label'] for g in demos)==572
    return old,demos,supports


def labeled_loss(pred,g):
    first=g['first_label']
    return (pred[first:]-g['y'][first:]).square().mean()


def mask_contract(obj,data):
    g=data[1][1];assert g['first_label']==20 and len(g['x'])==297
    pred=obj.actor.sequence(g['x'],g['b']);loss=labeled_loss(pred,g)
    grads=torch.autograd.grad(loss,obj.actor.trainable())
    altered=dict(g,y=g['y'].clone());altered['y'][:20]=12345.
    pred2=obj.actor.sequence(g['x'],g['b']);loss2=labeled_loss(pred2,altered)
    grads2=torch.autograd.grad(loss2,obj.actor.trainable())
    assert torch.equal(loss,loss2) and all(torch.equal(a,b) for a,b in zip(grads,grads2))
    x=g['x'].clone().requires_grad_(True)
    loss3=labeled_loss(obj.actor.sequence(x,g['b']),g)
    dx=torch.autograd.grad(loss3,x)[0]
    prefix_gradient=float(dx[:20].abs().max());assert prefix_gradient>0. and torch.isfinite(dx).all()
    h=None;locations=[]
    with torch.no_grad():
        obj.actor.reset_noise();obj.actor.sample=False
        for row in g['x']:
            _,loc,h,_=obj.actor.step(row[None],h);locations.append(loc[0])
        step=torch.stack(locations);seq=obj.actor.sequence(g['x'],g['b'])
        error=float((step-seq).abs().max());assert error<1e-10
    return dict(state='PASS',mutated_prefix_targets_loss_and_grad_exact=True,prefix_action_labels=0,
        correction_human_labels=277,prior_human_labels=295,full297_history_unrolled=True,
        prefix_observation_gradient_max=prefix_gradient,sequence_step_max_error=error,new_native=0,new_Adam=0)


def schedule(p,old):
    rng=np.random.default_rng(p['fit_seed']);return [int(rng.choice([j for j,g in enumerate(old) if g['task']==k%8])) for k in range(p['fit_steps'])]


def fit(p,obj,data,guard,event,steps=None):
    old,demos,supports=data;seq=schedule(p,old);trace=[]
    for k,oi in enumerate(seq[:p['fit_steps'] if steps is None else steps]):
        guard();obj.opt.zero_grad(set_to_none=True);human=[];other=[]
        for g in demos:
            loss=.5*labeled_loss(obj.actor.sequence(g['x'],g['b']),g);assert torch.isfinite(loss)
            loss.backward();human.append(float(loss.detach()))
        for g,weight in [(old[oi],1.),(supports[k%2],.5)]:
            loss=weight*(obj.actor.sequence(g['x'],g['b'])-g['y']).square().mean();assert torch.isfinite(loss)
            loss.backward();other.append(float(loss.detach()))
        norm=float(torch.nn.utils.clip_grad_norm_(obj.actor.trainable(),1.,error_if_nonfinite=True));event(dict(phase='BEFORE',step=k))
        obj.opt.step();obj.calls+=1;assert all(v.grad is None for v in obj.actor.base.parameters());event(dict(phase='AFTER',step=k))
        trace.append(dict(step=k,old_episode=oi,prior_human_loss=human[0],correction_human_loss=human[1],
            human_loss=sum(human),old_loss=other[0],support_loss=other[1],gradient_norm=norm))
    return trace


def descriptive_error(obj,data):
    result={}
    with torch.no_grad():
        for g in data[1]:
            n=g['first_label'];pred=obj.actor.sequence(g['x'],g['b']);error=pred[n:]-g['y'][n:];executed=pred[n:].clamp(-1,1)-g['y'][n:]
            result[g['name']]=dict(raw_location_rmse=error.square().mean(0).sqrt().tolist(),
                clipped_action_rmse=executed.square().mean(0).sqrt().tolist(),labeled_decisions=len(error),masked_prefix=n)
    result['not_closed_loop_navigation']=True
    return result
