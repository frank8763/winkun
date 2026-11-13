"""Single systemd cgroup, aggregate one CPU, hard next-morning deadline."""
from pathlib import Path
import os,subprocess,sys,time
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t542_runtime as R
p=R.read(R.OUT/'protocol.json');R.verify(p,True);assert not (R.OUT/'launch.json').exists()
assert R.read(R.OUT/'native_preflight/audit.json')['state']=='PASS'
src=R.read(R.ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=R.ROOT,text=True).strip();assert not subprocess.check_output(['git','status','--porcelain'],cwd=R.ROOT,text=True).strip()
seconds=int(p['absolute_all_stop']-time.time())-5;assert seconds>7200
subprocess.run(['systemd-run','--user','--unit=stage-t542-offline-field-relay','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(R.ROOT),'-p','StandardOutput=append:'+str(R.OUT/'driver.log'),'-p','StandardError=append:'+str(R.OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T542','--why=Registered offline fresh policy field relay until0830','taskset','-c','14',str(R.ROOT/'run_stage.sh'),str(HERE/'driver.py')],check=True);print('T542_SERVICE_SUBMITTED',flush=True)
