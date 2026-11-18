"""Single launch with external absolute cgroup deadline, after source readback."""
from pathlib import Path
import json,os,subprocess,sys,time
from register import ROOT,HERE,OUT,DOC,read,save,Resource

def main():
    p=read(OUT/'protocol.json');assert p['authorization_approved']
    assert not (OUT/'launch.json').exists() and not (OUT/'DRIVER_STARTED.json').exists()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    Resource.resources();seconds=int(p['absolute_all_stop']-time.time()-2);assert seconds>1800
    cmd=['systemd-run','--user','--unit=stage-t646','--collect','--property=Type=exec',
         '--property=CPUQuota=400%','--property=CPUWeight=10','--property=Nice=19',
         '--property=AllowedCPUs=8 10 12 14','--property=MemoryMax=4G','--property=MemorySwapMax=0',
         '--property=KillMode=control-group','--property=TimeoutStopSec=5',f'--property=RuntimeMaxSec={seconds}',
         f'--property=WorkingDirectory={ROOT}',f'--property=StandardOutput=append:{OUT}/driver.log',
         '--property=StandardError=inherit','--setenv=CUDA_VISIBLE_DEVICES=',
         'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage T646','--why=Bounded offline experiments until 09:00',
         str(ROOT/'run_stage.sh'),str(HERE/'driver.py')]
    c=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True);assert c.returncode==0,c.stderr
    path=OUT/'launch.json'
    with path.open('x') as f:json.dump(dict(state='SYSTEMD_LAUNCHED',epoch=time.time(),unit='stage-t646',source_remote=receipt['remote_commit'],absolute_all_stop=p['absolute_all_stop'],cgroup_runtime_seconds=seconds),f,indent=2)
    print(c.stderr.strip(),flush=True)

if __name__=='__main__':main()
