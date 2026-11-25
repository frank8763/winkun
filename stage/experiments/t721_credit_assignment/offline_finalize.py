"""Local-only postflight; no network client or source transport is called."""
from pathlib import Path
import importlib.util,os,signal,subprocess,time,traceback,json
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('T721_local_postflight',HERE/'finalize_transport_recovery.py');F=importlib.util.module_from_spec(s);s.loader.exec_module(F)
C,D,O=F.C,F.D,F.O

def main():
    p=C.root_plan();C.save(D/'OFFLINE_FINALIZER_STARTED.json',dict(pid=os.getpid(),epoch=time.time(),network_required=False,unique_local_postflight=True))
    while True:
        props=subprocess.check_output(['systemctl','--user','show',C.UNIT,'-p','MainPID','-p','ActiveState'],text=True)
        if 'MainPID=0' in props and ('ActiveState=inactive' in props or 'ActiveState=failed' in props):break
        if time.time()>=p['absolute_all_stop']-5:
            # Only our registered scientific cgroup, preserving the existing hard stop.
            subprocess.run(['systemctl','--user','stop',C.UNIT],check=True,timeout=10);break
        time.sleep(3)
    C.stopped();pending=C.read(D/'TRANSPORT_PENDING.json');F.charge(pending['remote_commit'])
    outcome='COMPLETE';error=None
    try:F.analysis()
    except BaseException:
        outcome='LOCAL_ANALYSIS_FAILED';error=traceback.format_exc();C.save(D/'OFFLINE_ANALYSIS_FAILURE.json',dict(error=error,raw_preserved=True,new_science=0))
    # The online finalizer was stopped while waiting, before any analysis/archive/source.
    # All unattempted remote work remains pending, never counted as verified or charged.
    C.save(D/'DEFERRED_TRANSPORT.json',dict(state='PENDING_NETWORK_AFTER_ALL_SCIENCE',new_remote_requests=0,remote_archive_attempted=False,final_source_attempted=False,all_full_raw_and_models_preserved=True,primary_readback_required_before_verified=True,source_exact_tree_required=True,reason='User leaves with network unavailable; all training and analysis run locally.'))
    C.save(D/'FINAL_SOURCE.json',dict(state='DEFERRED_NETWORK_UNAVAILABLE_NOT_ATTEMPTED',science_independent_of_network=True,already_verified_pre_source_commit=pending['remote_commit']))
    C.save(D/'FINALIZER_DONE.json',dict(state='STOPPED',local_outcome=outcome,error=error,network_required=False,no_next_science=True,remote_work_pending=True,epoch=time.time()))
    # A local commit makes tomorrow's report reviewable without a network.
    paths=['stage/docs/overnight_20261124_25/t721','stage/experiments/t721_credit_assignment','stage/docs/RESEARCH_STATUS.md','stage/docs/weekend_20261120_23/LEDGER.json']
    try:
        subprocess.run(['git','add','--',*paths],check=True,cwd=C.ROOT)
        subprocess.run(['git','commit','-m','Save offline T721 complete paired results and deferred remote receipts'],check=True,cwd=C.ROOT)
    except BaseException:C.save(D/'LOCAL_COMMIT_FAILURE.json',dict(error=traceback.format_exc(),all_originals_preserved=True))
if __name__=='__main__':
    try:main()
    except BaseException:C.save(D/'OFFLINE_FINALIZER_FAILURE.json',dict(error=traceback.format_exc(),all_local_preserved=True,epoch=time.time()));raise
