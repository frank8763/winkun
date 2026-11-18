"""Conditional density/score mask contract on real audited historical suffixes."""
from pathlib import Path
import ast
import copy
import json
import tempfile
import time
from unittest import mock
import t633_runtime as R
M,L=R.M,R.L


def rejected(fn):
    try:fn()
    except AssertionError:return
    raise AssertionError('Invalid batch accepted')


def main():
    p=R.plan();assert not p['authorization_approved'];R.setup(p)
    rows=[r for r in R.read(R.ROOT/'stage/runs/t631_explored_recovery/summary.json')['result']['exploration']
          if r['job_id']=='t502_field_sector4_offset2']
    assert len(rows)==4
    arrays=[R.load(r,p) for r in rows]
    R.R.gpu_guard(dict(p,absolute_learning_stop=time.time()+300));M.configure_gpu(p)
    # Actual historical behavior data are deliberately paired with their
    # original behavior model. They are never accepted as fresh T632 data.
    fixture=dict(p,student_initial=p['reference_model'])
    obj=M.make(fixture,p['raw_sigma']);before=M.digest(M.state(obj))
    parts,diag=L.batch(obj[0],arrays,fixture)
    assert len(parts)==4 and all(int(v['mask'].sum())==len(a['states'])-int(a['student_takeover_first']) for v,a in zip(parts,arrays))
    assert len(set(round(x,6) for x in diag['suffix_returns']))>1
    changed=[{k:v.copy() for k,v in a.items()} for a in arrays]
    for a in changed:
        first=int(a['student_takeover_first']);a['raw_actions'][:first]+=10.;a['logp'][:first]+=99.
    pp,dd=L.batch(obj[0],changed,fixture);assert dd==diag
    for a,b in zip(parts,pp):
        assert M.torch.equal(a['raw'],b['raw']) and M.torch.equal(a['old'],b['old']) and M.torch.equal(a['adv'],b['adv'])
    bad=[{k:v.copy() for k,v in a.items()} for a in arrays]
    bad[0]['logp'][int(bad[0]['student_takeover_first'])]+=.01
    rejected(lambda:L.batch(obj[0],bad,fixture))
    current=M.make(p,p['raw_sigma']);current_before=M.digest(M.state(current))
    rejected(lambda:L.batch(current[0],arrays,p))
    nominal=R.load(p['nominal_eight_row'],p,True)
    refs=L.references(obj[0],fixture,nominal,nominal)
    loss,detail=L.objective(obj[0],parts,refs,fixture);loss.backward()
    grad=M.torch.cat([v.grad.flatten() for v in M.L.variables(obj[0])])
    assert M.torch.isfinite(grad).all() and float(grad.norm())>0
    assert M.digest(M.state(obj))==before and M.digest(M.state(current))==current_before
    assert not (set(p['before_success_ids'])<=set(p['student_success_ids']))
    assert len(set(p['before_success_ids'])|set(p['student_success_ids']))==35
    Core=R.R.B
    with tempfile.TemporaryDirectory(prefix='io_',dir=R.DOC) as tmp:
        folder=Path(tmp)
        with mock.patch.object(Core,'OUT',folder),mock.patch.object(Core,'verify',lambda *args:None):
            rt=Core.Runtime(dict(p,absolute_native_stop=time.time()+300))
            class Pool:
                def request(self,requests):
                    output=[]
                    for req in requests:
                        i=next(i for i,r in enumerate(rows) if r['seed']==req['seed'])
                        a=arrays[i];row=copy.deepcopy(rows[i]);row.pop('request',None)
                        if req['full_path']:
                            f=Path(req['full_path']);f.parent.mkdir(exist_ok=True);f.write_bytes(Core.arrays_blob(a))
                        output.append((a,row))
                    return output
            rt.pool=Pool();jobs=[r['request']['job'] for r in rows];seeds=[r['seed'] for r in rows]
            vv=rt.request(jobs,p['reference_model'],'FRESH_TRAIN4',seeds=seeds)
            rt.request(jobs,p['reference_model'],'EXACT_FIXTURE',seeds=seeds,references=[r for a,r in vv])
            assert len(rt.rows)==len(rt.replays)==4
            assert all(Path(r['path']).exists() and R.sha(r['path'])==r['sha256'] for r in rt.rows)
        current[0].cpu();model=M.export(current,p,folder/'initial.pt.gz')
        assert M.frozen(p,model['path']).phash()==p['student_initial']['policy_hash']
    for f in R.HERE.glob('*.py'):ast.parse(f.read_text())
    report=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',historical_behavior_fixture_only=True,
        forced_prefix_raw_and_logp_mutation_has_no_loss_score_effect=True,
        modified_suffix_density_rejected=True,old_policy_data_rejected_for_current_T632=True,
        suffix_returns=diag['suffix_returns'],conditional_GPU_gradient_norm=float(grad.norm()),
        complete_history_reconstructed=True,actual_seeded_Runtime_full_exact_IO=True,
        initial_current_policy_export_reload_exact=True,original_union_student_gate_size=35,
        Actor_optimizer_state_unchanged=True,native_steps=0,optimizer_updates=0)
    R.save(R.DOC/'PREFLIGHT.json',report);print(json.dumps(report))


if __name__=='__main__':main()
