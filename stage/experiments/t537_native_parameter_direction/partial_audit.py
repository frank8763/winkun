"""Audit retained metadata only after the first-window NPZ-size stop."""
from pathlib import Path
import os,sys,time
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t537_runtime as R
def main():
    p=R.read(R.OUT/'protocol.json');R.verify(p,True);s=R.read(R.OUT/'summary.json');launch=R.read(R.OUT/'launch.json')
    assert s['state']=='FAILED' and '137994, 131072' in s['error'] and s['physics']==5000 and s['commands']==12
    assert not Path('/proc',str(s['pid'])).exists() and not Path('/proc',str(launch['pid'])).exists()
    assert not (R.OUT/'audit.json').exists()
    rows=[R.read_json if False else __import__('json').loads(x) for x in (R.OUT/'cases.jsonl').read_text().splitlines()]
    assert len(rows)==10 and all(r['id']==p['windows'][0]['id'] for r in rows)
    for r in rows[:5]:
        twin=next(z for z in rows if z['repeat']==1 and z['alpha']==r['alpha'])
        assert r['array_hashes']==twin['array_hashes'] and r['metrics']==twin['metrics'] and r['policy_hash']==twin['policy_hash']
    assert sum(r['metrics']['physics'] for r in rows)==s['physics'] and not (R.OUT/'window_0.npz').exists()
    R.save(R.OUT/'audit.json',dict(state='PASS',scope='METADATA_ONLY_NOT_FULL_RAW_ARRAY_VERIFICATION',pid=os.getpid(),new_native=0,new_optimizer=0,physics=5000,commands=12,reported_hashes_and_metrics_exact_twins=True,full_arrays_unsaved=True,reported_native_metrics_NOT_independently_validated=True,three_other_windows_unexecuted=True,all_direction_gate=None))
    R.save(R.OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=[s['pid'],launch['pid']],finalizer_writes_only_docs_after_this_point=True,metadata_only_failure_closeout=True,epoch=time.time()))
    R.save(R.DOC/'T537_PARTIAL_AUDIT_REGISTRATION.json',dict(new_native=0,new_optimizer=0,script_sha256=R.sha(__file__),failed_summary_sha256=R.sha(R.OUT/'summary.json'),protocol_unchanged=True,no_native_restart=True,audit_scope='metadata only; full arrays lost before persist'))
    print('T537_RETAINED_METADATA_AUDIT_PASS_FULL_ARRAYS_MISSING',flush=True)
if __name__=='__main__':main()
