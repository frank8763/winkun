"""Fixed73parameter repair with factual-parent replay and worst-state preservation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t372_nonlinear_terminal_repair'))
from t372_core import *
import t372_core as PREVIOUS
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t374_factual_worst_state_repair';DOC=ROOT/'stage/docs/narrow_20261105'


def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes'] and K.S.used(OUT)<=p['final_bytes']
    assert shutil.disk_usage(ROOT).free>=p['min_free_bytes']


def dataset(p,q,guard):
    old,g,_=PREVIOUS.dataset(p,q,guard);factual=[]
    for r in p['factual_parent_rows']:
        assert K.C.sha(r['path'])==r['sha256'];a=arrays(r['path']);assert {k:digest(v) for k,v in a.items()}==r['array_hashes']
        x=torch.tensor(a['states']);b=q.base_actions(x)
        with torch.no_grad():y=q.sequence(x,b).detach().clone()
        assert float((y-torch.tensor(a['locations'])).abs().max())<1e-10
        assert np.array_equal(y.clamp(-1,1).float().numpy(),a['actions'])
        factual.append(dict(id=r['job_id'],x=x,b=b,y=y,actual_actions=a['actions']));guard()
    assert len(factual)==14
    return old,g,factual


def fit(p,obj,data,guard,event,steps=None):
    old,g,factual=data;seq=schedule(p,old);trace=[];fixed=fixed_hash(obj.actor)
    for k,oi in enumerate(seq[:p['fit_steps'] if steps is None else steps]):
        guard();obj.opt.zero_grad(set_to_none=True);pred=obj.actor.sequence(g['x'],g['b'])[:,1]
        patch=(pred[g['mask']]-g['target'][g['mask']]).square().mean();route=(pred[~g['mask']]-g['y'][~g['mask'],1]).square().max()
        patch.backward(retain_graph=True);route.backward();losses=[float(patch.detach()),float(route.detach())]
        for ref,reduce in [(old[oi],'mean'),(factual[k%14],'max')]:
            error=(obj.actor.sequence(ref['x'],ref['b'])[:,1]-ref['y'][:,1]).square();loss=error.mean() if reduce=='mean' else error.max()
            loss.backward();losses.append(float(loss.detach()))
        assert all(np.isfinite(losses));norm=float(torch.nn.utils.clip_grad_norm_(obj.actor.adapter.parameters(),1.,error_if_nonfinite=True))
        event(dict(phase='BEFORE',step=k));obj.opt.step();obj.calls+=1;event(dict(phase='AFTER',step=k))
        assert fixed_hash(obj.actor)==fixed and all(v.grad is None for name,v in obj.actor.named_parameters() if not name.startswith('adapter.'))
        trace.append(dict(step=k,old_episode=oi,factual_history=k%14,patch_MSE=losses[0],route_max_squared_error=losses[1],old_MSE=losses[2],factual_max_squared_error=losses[3],gradient_norm=norm))
    return trace


def locality(q,data):
    _,g,factual=data;rows=[]
    with torch.no_grad():
        pred=q.sequence(g['x'],g['b']);action=pred.clamp(-1,1).float();reference=g['y'].clamp(-1,1).float()
        delta=(action-reference).abs();patch_rmse=float((pred[g['mask'],1]-g['target'][g['mask']]).square().mean().sqrt())
        for ref in factual:
            value=q.sequence(ref['x'],ref['b']);err=(value.clamp(-1,1).float()-ref['y'].clamp(-1,1).float()).abs()
            rows.append(dict(id=ref['id'],decisions=len(ref['x']),executed_steer_max=float(err[:,1].max()),executed_drive_max=float(err[:,0].max()),executed_steer_rmse=float(err[:,1].double().square().mean().sqrt()),executed_actions_exact=not bool(torch.count_nonzero(err))))
    assert all(r['executed_drive_max']==0. for r in rows) and not torch.count_nonzero(delta[:,0])
    return dict(patch_steer_rmse=patch_rmse,prefix248_executed_steer_max=float(delta[:248,1].max()),first20_executed_steer_max=float(delta[:20,1].max()),nonpatch_executed_steer_max=float(delta[~g['mask'],1].max()),factual_executed_steer_max=max(v['executed_steer_max'] for v in rows),factual_routes=rows,drive_on_identical_histories_exact=True,not_closed_loop_success=True)


def gate(p,metrics):
    t=p['diagnostic_gate_limits'];checks={name:metrics[name]<=limit for name,limit in t.items()}
    return dict(passed=all(checks.values()),checks=checks,limits=t,scope='Predeclared comparative screening only, no safe tolerance or success certificate; native centre decides outcome.')


def contract(p,obj,data):
    result=PREVIOUS.repair_contract(obj,data);m=locality(obj.actor,data)
    assert m['prefix248_executed_steer_max']==m['first20_executed_steer_max']==m['factual_executed_steer_max']==0.
    assert all(r['executed_actions_exact'] for r in m['factual_routes'])
    return dict(result,factual_parent_routes_exact=14,factual_decisions=sum(len(g['x']) for g in data[2]),current_parent_targets_not_cross_policy=True,losses='patch mean, route max, old mean, factual max; all weights1',native=0)
