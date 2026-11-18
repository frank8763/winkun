"""Actual request/payload/exact-replay path with existing historical arrays."""
from pathlib import Path
import ast
import copy
import gzip
import json
import os
import tempfile
from unittest import mock
import t630_runtime as R
from early_departure import select


def main():
    p=R.plan();assert not p['authorization_approved']
    R.setup(p)
    assert not R.M.torch.cuda.is_initialized()
    source=p['candidate_sources']
    arrays=[R.R.B.R.E.N.S.arrays(r['path']) for r in source]
    assert all(select(a,a) is None for a in arrays)
    mixed=R.read(R.ROOT/'stage/runs/t629_offline_night_relay/recovery/summary.json')['result']['qualification_rows']
    for row in mixed:
        assert R.sha(row['path'])==row['sha256']
        actual=R.R.B.R.E.N.S.arrays(row['path'])
        R.Native.Prior.inspect(p,[(actual,row)])
    # Same clock/initial state, a true 3cm departure at one aligned decision.
    a=arrays[0];b={k:v.copy() for k,v in a.items()};b['poses'][1,0]+=.03
    assert select(a,b)['first']==1
    wrong={k:v.copy() for k,v in a.items()};wrong['initial_native_integration'][0]+=.01
    try:select(a,wrong)
    except AssertionError:pass
    else:raise AssertionError('Mismatched initial state was accepted')
    B=R.R.B
    with tempfile.TemporaryDirectory(prefix='io_',dir=R.DOC) as tmp:
        folder=Path(tmp)
        with mock.patch.object(B,'OUT',folder),mock.patch.object(B,'verify',lambda *a:None):
            rt=B.Runtime(dict(p,absolute_native_stop=__import__('time').time()+300))
            class VirtualPool:
                def request(self,requests):
                    out=[]
                    for req in requests:
                        i=next(i for i,r in enumerate(source) if r['job_id']==req['job']['id'])
                        a=arrays[i];row=copy.deepcopy(source[i]);row.pop('request',None)
                        if req['full_path']:
                            path=Path(req['full_path']);path.parent.mkdir(exist_ok=True)
                            path.write_bytes(B.arrays_blob(a))
                        out.append((a,row))
                    return out
            rt.pool=VirtualPool()
            jobs=[r['request']['job'] for r in source]
            model=p['recovery_fallback_model']
            # Use actual candidate-model metadata for the historical IO fixture.
            vv=rt.request(jobs,model,'FRESH_TRAIN4')
            rt.request(jobs,model,'EXACT_FIXTURE',references=[r for a,r in vv])
            assert len(rt.rows)==len(rt.replays)==2
            assert all(Path(r['path']).exists() and R.sha(r['path'])==r['sha256'] for r in rt.rows)
    for f in R.HERE.glob('*.py'):ast.parse(f.read_text())
    report=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',historical_real_full_data=True,
                actual_Runtime_full_exact_IO=True,early_departure_rule_checked=True,
                actual_forced_prefix_own_hidden_history_audited=len(mixed),
                initial_state_mismatch_rejected=True,all_source_syntax_valid=True,
                GPU_initialized=False,native_steps=0,new_optimizer=0)
    R.save(R.DOC/'PREFLIGHT.json',report)
    print(json.dumps(report))


if __name__=='__main__':main()
