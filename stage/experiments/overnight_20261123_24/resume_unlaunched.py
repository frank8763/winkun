"""One source-boundary recovery; never repeat preparation, fits or physics."""
from pathlib import Path
import ast,hashlib,importlib.util,json,os,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('night_recovery_relay',HERE/'relay.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
DOC=R.DOC;D=DOC/'t710';H=ROOT/'stage/experiments/t710_whole_head_replay';O=ROOT/'stage/runs/t710_whole_head_replay'


def main():
    assert not (DOC/'recovery/STARTED.json').exists()
    R.save(DOC/'recovery/STARTED.json',dict(pid=os.getpid(),epoch=time.time(),single_entry=True))
    for unit in ['stage-night-20261123.service','stage-t709.service','stage-t709-finalizer.service','stage-t710.service','stage-t710-finalizer.service']:assert R.inactive(unit),unit
    assert not (O/'LAUNCH_STARTED.json').exists() and R.read(D/'REGISTERED.json')['state']=='REGISTERED_NOT_LAUNCHED'
    cs=importlib.util.spec_from_file_location('t710_recovery_controller',H/'run.py');C=importlib.util.module_from_spec(cs);cs.loader.exec_module(C)
    C.stopped();p=C.read(O/'protocol.json');C.verify(p,force=True)
    assert p['absolute_all_stop']==R.read(DOC/'recovery/PLAN.json')['old_T710_absolute_all_stop']
    for folder in [H,ROOT/'stage/experiments/t711_fresh_starts',HERE]:
        for file in folder.glob('*.py'):ast.parse(file.read_text())
    paths=['stage/experiments/t710_whole_head_replay','stage/experiments/t711_fresh_starts','stage/experiments/overnight_20261123_24','stage/docs/overnight_20261123_24','stage/docs/day_20261123/t709/TRANSPORT_LEDGER.json','stage/docs/day_20261123/t709/FINAL_SOURCE.json','stage/docs/day_20261123/t709/FINALIZER_DONE.json']
    subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True);subprocess.run(['git','commit','-m','Fix completed-stage receipt commit at overnight source handoff'],cwd=ROOT,check=True)
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True);assert not dirty.strip(),dirty
    R.save(DOC/'recovery/GREEN.json',dict(state='GREEN',real_source_clean_tree_guard_pass=True,all_current_scientific_inputs_SHA_PASS=True,no_preparation_register_fit_or_native_repeated=True,epoch=time.time()))
    target=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();start=time.time()
    code=R.command([str(R.PY),str(H/'run.py'),'sync'],D/'prelaunch_source_recovery.log',300);assert code==0,code
    receipt=R.read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==target
    R.save(D/'TRANSPORT_PENDING.json',dict(state='ACTUAL_EXIT0_EXACT_TREE_PENDING_FINAL_EXIT_CHARGE',remote_commit=receipt['remote_commit'],root_tree=target,charge_seconds=300,accounted=False,actual_shell_seconds=time.time()-start,original_failed_source300_still_charged=True))
    assert R.command([str(R.PY),str(H/'run.py'),'launch'],D/'launch.log',120)==0
    R.finalizer('t710',H,D,p['absolute_all_stop'])
    R.save(DOC/'STATE.json',dict(state='ACTIVE',stage='T710',epoch=time.time(),recovery='unexecuted_source_boundary_only',original_deadlines_unchanged=True))
    R.wait_finished('t710',D,p['absolute_all_stop'])
    prior=R.read(O/'FINAL_REPORT.json');next_result=R.run_stage('t711','t711_fresh_starts')
    R.save(DOC/'FINAL_REPORT.json',dict(state='COMPLETE',stages=dict(t710=prior['state'],t711=next_result['state']),epoch=time.time(),original_relay_failure_preserved=True,retained='T698',no_recurring_automation=True,no_more_science=True))
    R.save(DOC/'STATE.json',dict(state='STOPPED',epoch=time.time(),reason='Finite useful stages finished'))


if __name__=='__main__':
    try:main()
    except BaseException:R.save(DOC/'recovery/FAILURE.json',dict(state='FAILED',error=traceback.format_exc(),epoch=time.time(),no_duplicate_or_consumed_science_retry=True));raise
