"""Independent complete four frozen diagnostics and exact-replay review."""
from pathlib import Path
import sys,json,hashlib,time
ROOT=Path(__file__).resolve().parents[3];DOC=ROOT/'stage/docs/review_20261111'
sys.path.insert(0,str(ROOT/'stage/experiments/t488_left_failure_focus'));import t488_runtime as T

def main():
    root=ROOT/'stage/runs/t488_left_failure_focus';p=T.read(root/'protocol.json');s=T.read(root/'capture/summary.json')
    assert s['state']=='FAILED' and 'AssertionError' in s['error']
    for f in ['launch.json','driver_started.json']:assert not Path('/proc',str(T.read(root/f)['pid'])).exists()
    T.stopped(root/'capture');before={str(f):T.sha(f) for f in root.rglob('*') if f.is_file()}
    rr=T.rows(root/'capture');re=T.rows(root/'capture','replays.jsonl')
    assert len(rr)==len(re)==4 and all(r['role']=='FROZEN_DIAGNOSTIC' for r in rr)
    assert not (root/'capture/optimizer_calls.jsonl').exists()
    for row in rr:
        assert T.sha(row['path'])==row['sha256'];a=T.E.N.S.arrays(row['path'])
        assert all(T.digest(v)==row['array_hashes'][k] for k,v in a.items())
        T.M.audit_full_history(p,row,a)
        prev=p['diagnostic_original_rows'][row['job_id']]
        assert row['array_digest']==prev['array_digest'] and row['integration_sha256']==prev['integration_sha256']
    by={r['path']:r for r in rr}
    for row in re:
        prev=by[row['reference_path']];assert row['exact'] and row['array_hashes']==prev['array_hashes'] and row['integration_sha256']==prev['integration_sha256']
    assert s['physics']==sum(x['physics'] for x in rr+re) and s['commands']==len(rr)+len(re)+4
    assert {str(f):T.sha(f) for f in root.rglob('*') if f.is_file()}==before
    T.save(DOC/'T489_DIAGNOSTIC_REVIEW.json',dict(state='PASS_COMPLETE_DIAGNOSTICS_NOT_COMPLETE_T488_TRAINING',epoch=time.time(),native=0,optimizer=0,physics=s['physics'],commands=s['commands'],original_raw_unchanged=True,
        diagnostic_rows=rr,metrics=[T.diagnose(T.E.N.S.arrays(r['path']),r) for r in rr],training_collected=0,original_error_preserved=True))
    print('T489_DIAGNOSTIC_REVIEW_PASS',flush=True)
if __name__=='__main__':main()
