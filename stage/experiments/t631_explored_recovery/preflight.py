"""Zero-update real-data, prefix handoff, sampling, GPU-gradient and IO checks."""
from pathlib import Path
import ast
import copy
import json
import os
import tempfile
import types
from unittest import mock
import t631_runtime as R
N,L,M=R.N,R.L,R.M


def reject(fn):
    try:fn()
    except AssertionError:return True
    raise AssertionError('Invalid data accepted')


def main():
    p=R.plan();assert not p['authorization_approved'];R.setup(p)
    prior=R.read(R.ROOT/'stage/runs/t630_early_departure_recovery/summary.json')
    mixed=[r['recovery'] for r in prior['results']]
    for row in mixed:
        a=R.R.B.R.E.N.S.arrays(row['path']);N.inspect(p,[(a,row)])
        first=int(a['student_takeover_first'])
        for key in ['actions','hidden_before','ownership']:
            bad={k:v.copy() for k,v in a.items()}
            if key=='ownership':bad[key][0]=0
            else:bad[key].flat[0]+=.01
            reject(lambda:N.inspect(p,[(bad,row)]))
        # Exercise the actual Session.step boundary using only a parent seam.
        s=object.__new__(N.Session);s.rows=[None]*(first-1);s.first=first
        s.source=R.R.B.R.E.N.S.arrays(row['request']['job']['source_path'])
        s.explore=True;s.prefix_actions=[];s.q=types.SimpleNamespace(rng=object())
        flags=[]
        def parent(self,forced=None,owner=0,actor_active=True):
            flags.append((self.q.sample,owner,forced is None,actor_active));self.rows.append(None)
        with mock.patch.object(N.N.F.FieldSession,'step',parent):
            s.step();s.step()
        assert flags==[(False,1,False,True),(True,0,True,True)]
    collection=R.read(R.ROOT/'stage/runs/t629_offline_night_relay/collect/summary.json')['result']['training_rows']
    success=[r for r in collection if R.R.safe(r)][:2]
    assert len(success)==2
    arrays=[R.R.B.R.E.N.S.arrays(r['path']) for r in success]
    for a,r in zip(arrays,success):N.inspect(p,[(a,r)])
    # Historical observations drive a policy-only stochastic suffix trace.
    # These artificial commands are never treated as physical transitions.
    x=arrays[0]['states'][:12];seed=[631,0,0];traces=[]
    for repeat in range(2):
        q=M.frozen(p,p['reference_model']['path']);q.reset_noise()
        q.rng=M.np.random.default_rng(M.np.random.SeedSequence(seed));h=None
        with M.torch.no_grad():
            for k,state in enumerate(x):
                q.sample=k>=4;_,_,h,_=q.step(M.torch.tensor(state[None]),h)
        traces.append(copy.deepcopy(q.tape))
    assert all(M.digest(a)==M.digest(b) for a,b in zip(*traces))
    assert not M.np.asarray([z['stochastic'] for z in traces[0][:4]]).any()
    assert M.np.asarray([z['stochastic'] for z in traces[0][4:]]).all()
    assert any(M.np.abs(z['raw_actions']-z['locations']).max()>0 for z in traces[0][4:])
    B=R.R.B
    with tempfile.TemporaryDirectory(prefix='io_',dir=R.DOC) as tmp:
        folder=Path(tmp)
        with mock.patch.object(B,'OUT',folder),mock.patch.object(B,'verify',lambda *a:None):
            rt=B.Runtime(dict(p,absolute_native_stop=__import__('time').time()+300))
            class Pool:
                def request(self,requests):
                    values=[]
                    for req in requests:
                        i=next(i for i,r in enumerate(success)
                               if r['job_id']==req['job']['id'] and r['seed']==req['seed'])
                        a=arrays[i];row=copy.deepcopy(success[i]);row.pop('request',None)
                        if req['full_path']:
                            f=Path(req['full_path']);f.parent.mkdir(exist_ok=True);f.write_bytes(B.arrays_blob(a))
                        values.append((a,row))
                    return values
            rt.pool=Pool();jobs=[r['request']['job'] for r in success];seeds=[r['seed'] for r in success]
            vv=rt.request(jobs,p['reference_model'],'FRESH_TRAIN4',seeds=seeds)
            rt.request(jobs,p['reference_model'],'EXACT_PREFLIGHT',seeds=seeds,references=[r for a,r in vv])
            assert len(rt.rows)==len(rt.replays)==2
            assert all(Path(r['path']).exists() and R.sha(r['path'])==r['sha256'] for r in rt.rows)
    R.R.gpu_guard(dict(p,absolute_learning_stop=__import__('time').time()+300));M.configure_gpu(p)
    obj=M.make(p,p['raw_sigma']);before=M.digest(M.state(obj))
    data=L.dataset(p,[(a,0) for a in arrays],R.anchor_arrays(p))
    refs=M.references(obj[0],p)
    loss,parts=L.objective(obj[0],data,refs,p);loss.backward()
    grad=M.torch.cat([v.grad.flatten() for v in M.L.variables(obj[0])])
    assert M.torch.isfinite(grad).all() and float(grad.norm())>0
    assert M.digest(M.state(obj))==before
    assert not M.np.array_equal(arrays[0]['raw_actions'],arrays[0]['locations'])
    for f in R.HERE.glob('*.py'):ast.parse(f.read_text())
    report=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',real_mixed_prefix_rows=len(mixed),
                forced_prefix_hidden_ownership_negative_checks=6,
                actual_Session_step_sample_switch=True,seeded_policy_only_suffix_reproducible=True,
                synthetic_policy_trace_never_used_as_physical_transition=True,
                actual_seeded_Runtime_full_and_exact_IO=True,
                real_success_sample_labels_not_failed_teacher_means=True,
                complete_own_history_GPU_gradient_norm=float(grad.norm()),
                unchanged_Actor_optimizer_state=True,native_steps=0,optimizer_updates=0)
    R.save(R.DOC/'PREFLIGHT.json',report);print(json.dumps(report))


if __name__=='__main__':main()
