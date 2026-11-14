"""Launch only after live-window admission and exact source backup."""
from pathlib import Path
import os,subprocess,sys,time
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import runtime as R
p=R.read(R.OUT/'protocol.json');R.verify(p,True)
pre=R.read(R.OUT/'preflight.json');assert pre['state']=='PASS_ZERO_NATIVE' and not Path('/proc',str(pre['pid'])).exists()
assert not (R.OUT/'launch.json').exists()
src=R.read(R.ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=R.ROOT,text=True).strip()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=R.ROOT,text=True).strip()
seconds=int(p['absolute_all_stop']-time.time());assert seconds>p['main_seconds']+p['audit_seconds']+p['transport_seconds']+p['source_seconds']+90 and time.time()+p['main_seconds']<p['absolute_native_stop']
subprocess.run(['systemd-run','--user','--unit=stage-t551-retention-recovery-probe','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(R.ROOT),'-p','StandardOutput=append:'+str(R.OUT/'driver.log'),'-p','StandardError=append:'+str(R.OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T551','--why=Verified feedback recovery and MIR joint repair','taskset','-c','14',str(R.ROOT/'run_stage.sh'),str(HERE/'driver.py')],check=True)
print('T551_SERVICE_SUBMITTED',flush=True)
