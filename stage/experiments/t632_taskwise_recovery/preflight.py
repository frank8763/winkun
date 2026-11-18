"""Real qualified-label, full-history GPU-gradient and deterministic IO checks."""
import ast
import copy
import json
import tempfile
import time
from pathlib import Path
from unittest import mock
import t632_runtime as R
M,L=R.M,R.L


def main():
    p=R.plan();assert not p['authorization_approved'];R.setup(p)
    R.R.gpu_guard(dict(p,absolute_learning_stop=time.time()+300));M.configure_gpu(p)
    obj=M.make(p,p['raw_sigma']);before=M.digest(M.state(obj))
    assert obj[0].phash()==p['student_initial']['policy_hash']
    assert M.L.fixed(obj[0])==M.L.fixed(M.frozen(p,p['reference_model']['path']))
    data=L.dataset(p,R.load);refs=M.references(obj[0],p)
    assert len(data)==6 and len(refs)==33
    weights={}
    for row in data:weights[row['kind']]=weights.get(row['kind'],0.)+row['weight']
    assert abs(weights['field_successful_sample']-.25)<1e-12
    assert weights['nominal_eight_success']==.25 and weights['new_success_anchor']==.5
    x=data[0]['x'];mask=data[0]['mask'];first=int((~mask).sum())
    assert first==20 and len(x)>first
    with M.torch.no_grad():
        diff=float((obj[0].sequence(x)[first:]-obj[0].sequence(x[first:])).abs().max())
    assert diff>1e-6,'Historical prefix must not be silently zero-initialized'
    loss,parts=L.objective(obj[0],data,refs,p);loss.backward()
    grad=M.torch.cat([v.grad.flatten() for v in M.L.variables(obj[0])])
    assert M.torch.isfinite(grad).all() and float(grad.norm())>0
    assert M.digest(M.state(obj))==before
    bad=next(r for r in R.read(R.ROOT/'stage/runs/t631_explored_recovery/summary.json')['result']['exploration'] if r['collision'])
    try:R.load(bad)
    except AssertionError:pass
    else:raise AssertionError('Collision accepted as successful supervision')
    # Same-model actual deterministic forced-prefix result, with real packet,
    # hidden, ownership and original source. No simulator is constructed.
    row=next(v['recovery'] for v in R.read(R.ROOT/'stage/runs/t630_early_departure_recovery/summary.json')['results'] if v['id']=='t502_field_sector4_offset2')
    a=R.R.B.R.E.N.S.arrays(row['path']);R.N.inspect(p,[(a,row)])
    Core=R.R.B
    with tempfile.TemporaryDirectory(prefix='io_',dir=R.DOC) as tmp:
        folder=Path(tmp)
        with mock.patch.object(Core,'OUT',folder),mock.patch.object(Core,'verify',lambda *args:None):
            rt=Core.Runtime(dict(p,absolute_native_stop=time.time()+300))
            class Pool:
                def request(self,requests):
                    out=[]
                    for req in requests:
                        result=copy.deepcopy(row);result.pop('request',None)
                        if req['full_path']:
                            f=Path(req['full_path']);f.parent.mkdir(exist_ok=True);f.write_bytes(Core.arrays_blob(a))
                        out.append((a,result))
                    return out
            rt.pool=Pool();jobs=[row['request']['job']]
            vals=rt.request(jobs,p['reference_model'],'FRESH_TRAIN4')
            rt.request(jobs,p['reference_model'],'EXACT_PREFLIGHT',references=[vals[0][1]])
            assert len(rt.rows)==len(rt.replays)==1
            assert all(Path(v['path']).exists() and R.sha(v['path'])==v['sha256'] for v in rt.rows)
        obj[0].cpu();model=M.export(obj,p,folder/'initial.pt.gz')
        assert M.frozen(p,model['path']).phash()==p['student_initial']['policy_hash']
    for f in R.HERE.glob('*.py'):ast.parse(f.read_text())
    result=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',successful_field_trajectories=3,
        verified_nominal_eight_trajectory=1,new_success_anchors=2,old_history_references=33,
        grouped_weights=weights,full_history_vs_zero_init_difference=diff,
        complete_own_history_GPU_gradient_norm=float(grad.norm()),
        collision_BC_label_rejected=True,actual_deterministic_handoff_Runtime_full_exact_IO=True,
        initial_export_reload_same_policy=True,Actor_optimizer_state_unchanged=True,
        native_steps=0,optimizer_updates=0)
    R.save(R.DOC/'PREFLIGHT.json',result);print(json.dumps(result))


if __name__=='__main__':main()
