"""Close metadata-only early result-adapter failure without rerunning native."""
from pathlib import Path
import os,sys,time
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t540_runtime as R
def main():
    p=R.read(R.OUT/'protocol.json');R.verify(p,True);s=R.read(R.OUT/'summary.json');launch=R.read(R.OUT/'launch.json')
    assert s['state']=='FAILED' and "KeyError: 'job_id'" in s['error'] and s['physics']==63476 and s['commands']==6
    gens=R.read(R.OUT/'worker_generations.json');ids=[s['pid'],launch['pid']]+[i for g in gens for i in [g['parent'],*g['workers']]]
    assert all(not Path('/proc',str(i)).exists() for i in ids) and not (R.OUT/'audit.json').exists()
    assert s['completed_new_cases']==0 and not (R.OUT/'cases.jsonl').exists() and not list(R.OUT.glob('case_*.npz'))
    model=p['candidate'];assert R.sha(model['path'])==model['sha256']
    audit=dict(state='PASS',scope='FAILED_RUN_METADATA_AND_IMMUTABLE_MODEL_ONLY',pid=os.getpid(),new_native=0,new_optimizer=0,physics=s['physics'],commands=s['commands'],fully_recorded_new_cases=0,two_worker_results_dispatched_but_arrays_NOT_saved=True,native_outcomes_unknown_NOT_navigation_failures=True,firstpass_field24=None,original_T539_candidate_unchanged=True,combined_T539_T540_physics=235017+s['physics'],candidate_not_promoted=True)
    R.save(R.OUT/'audit.json',audit);R.save(R.OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=sorted(set(ids)),finalizer_writes_only_docs_after_this_point=True,metadata_only_closeout=True,epoch=time.time()))
    R.save(R.DOC/'T540_PARTIAL_AUDIT_REGISTRATION.json',dict(new_native=0,new_optimizer=0,scope=audit['scope'],script_sha256=R.sha(__file__),original_summary_sha256=R.sha(R.OUT/'summary.json'),original_protocol_unchanged=True,no_native_restart=True));print('T540_METADATA_ONLY_CLOSEOUT_PASS_OUTCOMES_UNKNOWN',flush=True)
if __name__=='__main__':main()
