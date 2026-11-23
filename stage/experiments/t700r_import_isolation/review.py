"""Only the never-started external audit; original T700 failure immutable."""
from pathlib import Path
import importlib.util,inspect,json,sys,types,time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t699_matched_value'))
# Establish the historical native dependency's generic `run` owner before
# loading process-only continuation helpers. No model forward occurs here.
import t699_native
spec=importlib.util.spec_from_file_location('t700_original_review',ROOT/'stage/experiments/t700_plain_external_review/review.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
M.HERE=Path(__file__).resolve().parent;M.DOC=ROOT/'stage/docs/day_20261123/t700r';M.OUT=ROOT/'stage/runs/t700r_import_isolation';M.UNIT='stage-t700r-review.service'
def old_stopped():
    s=importlib.util.spec_from_file_location('t700r_previous_guard',ROOT/'stage/experiments/t699_unexecuted_continuation/source_sync.py')
    guard=importlib.util.module_from_spec(s);s.loader.exec_module(guard)
    guard.R=types.SimpleNamespace(OUT=ROOT/'stage/runs/t699_unexecuted_continuation',DOC=ROOT/'stage/docs/overnight_20261122_23/continuation')
    guard.stopped()
    original=M.read(ROOT/'stage/runs/t700_plain_external_review/SEALED.json')
    assert original['rows_checked']==0 and original['state']=='FAILED'
    pid=M.read(ROOT/'stage/runs/t700_plain_external_review/DRIVER_STARTED.json')['pid']
    assert not Path('/proc',str(pid)).exists()
    state=M.subprocess.check_output(['systemctl','--user','show','stage-t700-review.service','-p','MainPID','-p','ActiveState'],text=True)
    assert 'MainPID=0' in state and ('ActiveState=inactive' in state or 'ActiveState=failed' in state)
M.old_stopped=old_stopped
exec(inspect.getsource(M.review).replace("key='T700_EXTERNAL_REVIEW'","key='T700R_EXTERNAL_REVIEW'"),M.__dict__)
def launch():
    M.stopped();p=M.read(M.OUT/'protocol.json');M.immutable(p,True);assert not (M.OUT/'launch.json').exists()
    receipt=M.read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==M.subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip()
    assert time.time()+920<p['absolute_all_stop']
    cmd=['systemd-run','--user','--unit=stage-t700r-review','--collect','--property=Type=exec','--property=CPUQuota=100%',
         '--property=CPUWeight=10','--property=AllowedCPUs=15','--property=Nice=19','--property=MemoryMax=2G','--property=MemorySwapMax=0',
         '--property=KillMode=control-group','--property=TimeoutStopSec=5','--property=RuntimeMaxSec=920',
         f'--property=WorkingDirectory={ROOT}',f'--property=StandardOutput=append:{M.OUT}/review.log','--property=StandardError=inherit',
         '--setenv=CUDA_VISIBLE_DEVICES=','--setenv=OMP_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1',str(ROOT/'run_stage.sh'),str(M.HERE/'review.py'),'review']
    c=M.subprocess.run(cmd,text=True,capture_output=True);assert c.returncode==0,c.stderr
    M.save(M.OUT/'launch.json',dict(epoch=time.time(),unit=M.UNIT,source_remote=receipt['remote_commit']));print(c.stderr)
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='preflight':
        old_stopped();M.C.resources();assert hasattr(sys.modules['run'],'P')
        print(json.dumps(dict(state='IMPORT_AND_ALL_OLD_STOPPED_PASS',generic_run_file=sys.modules['run'].__file__,forward=0,native=0,optimizer=0)))
    else:{'review':M.review,'sync':M.sync,'launch':launch,'stopped':M.stopped}[mode]()
