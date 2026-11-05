"""Full-history learning of one verified real-human narrow-scene route."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t358_direct_narrow'))
from t358_core import *
import t358_core as PREVIOUS
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t362_narrow_demonstration_learning';DOC=ROOT/'stage/docs/narrow_20261105'


def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    assert K.S.used(OUT)<=p['final_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']


def dataset(p,q,guard):
    old,_,supports=PREVIOUS.references(p,q);a=arrays(p['demonstration']);x=torch.tensor(a['states']);b=q.base_actions(x)
    assert len(x)==295 and a['end_flags'][-1,1] and not a['end_flags'][:,0].any() and not a['end_flags'][:-1].any()
    assert np.array_equal(a['states'][1:],a['next_states'][:-1]);guard()
    demo=dict(x=x,b=b,y=torch.tensor(a['actions'],dtype=torch.float64))
    return old,demo,supports


def schedule(p,old):
    rng=np.random.default_rng(p['fit_seed']);return [int(rng.choice([j for j,g in enumerate(old) if g['task']==k%8])) for k in range(p['fit_steps'])]


def fit(p,obj,data,guard,event,steps=None):
    old,demo,supports=data;seq=schedule(p,old);trace=[]
    for k,oi in enumerate(seq[:p['fit_steps'] if steps is None else steps]):
        guard();obj.opt.zero_grad(set_to_none=True);losses=[]
        for g,weight in [(demo,1.),(old[oi],1.),(supports[k%2],.5)]:
            loss=weight*(obj.actor.sequence(g['x'],g['b'])-g['y']).square().mean();assert torch.isfinite(loss)
            loss.backward();losses.append(float(loss.detach()))
        norm=float(torch.nn.utils.clip_grad_norm_(obj.actor.trainable(),1.,error_if_nonfinite=True));event(dict(phase='BEFORE',step=k))
        obj.opt.step();obj.calls+=1;assert all(v.grad is None for v in obj.actor.base.parameters());event(dict(phase='AFTER',step=k))
        trace.append(dict(step=k,old_episode=oi,human_loss=losses[0],old_loss=losses[1],support_loss=losses[2],gradient_norm=norm))
    return trace


def descriptive_error(obj,data):
    demo=data[1]
    with torch.no_grad():pred=obj.actor.sequence(demo['x'],demo['b']);error=pred-demo['y'];executed=pred.clamp(-1,1)-demo['y']
    return dict(raw_location_rmse=error.square().mean(0).sqrt().tolist(),clipped_action_rmse=executed.square().mean(0).sqrt().tolist(),not_closed_loop_navigation=True)
