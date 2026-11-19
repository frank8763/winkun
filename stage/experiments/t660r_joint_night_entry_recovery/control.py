"""Authorization, bounded registration, resource admission and one launch."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,hashlib,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t660r_joint_night_entry_recovery';DOC=ROOT/'stage/docs/overnight_20261119/recovery'
sys.path.insert(0,str(ROOT/'stage/experiments/t645_iterative_recovery_relay'));import control as Resource
def read(f):return json.loads(Path(f).read_text())
def save(f,v):Path(f).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()

def register():
    assert not OUT.exists();p=read(HERE/'protocol.json');a=read(DOC/'AUTHORIZATION.json')
    assert a['approved'] and not p['authorization_approved']
    for key in ['night_max_physics','night_max_commands','night_main_seconds','Actor_each_cap','Value_each_cap','max_bytes','global_bytes','absolute_all_stop']:
        assert a[key]==p[key],key
    assert read(DOC/'PREFLIGHT.json')['state']=='PASS_ZERO_NATIVE_ZERO_OPTIMIZER'
    assert read(DOC/'ENTRY_PREFLIGHT.json')['state']=='PASS_ZERO_NATIVE_ZERO_OPTIMIZER'
    for f,h in p['inputs'].items():assert sha(f)==h,f
    save(DOC/'RESOURCE_PREFLIGHT.json',Resource.resources())
    assert time.time()+p['finalization_seconds']+3600<p['absolute_learning_stop']
    p.update(authorization_approved=True,registration_epoch=time.time())
    p['inputs'].update({str(DOC/'AUTHORIZATION.json'):sha(DOC/'AUTHORIZATION.json'),str(DOC/'PREFLIGHT.json'):sha(DOC/'PREFLIGHT.json'),str(DOC/'ENTRY_PREFLIGHT.json'):sha(DOC/'ENTRY_PREFLIGHT.json')})
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'REGISTERED.json',dict(state='REGISTERED_NOT_LAUNCHED',epoch=p['registration_epoch'],all_stop=p['absolute_all_stop']))
    print('T660_REGISTERED_NOT_LAUNCHED',p['absolute_all_stop'])

def launch():
    p=read(OUT/'protocol.json');assert p['authorization_approved'] and not (OUT/'launch.json').exists()
    Resource.resources();receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    left=int(p['absolute_all_stop']-time.time()-2);assert left>3600
    cmd=['systemd-run','--user','--unit=stage-t660r','--collect','--property=Type=exec',
        '--property=CPUQuota=400%','--property=CPUWeight=10','--property=Nice=19','--property=AllowedCPUs=8 10 12 14',
        '--property=MemoryMax=4G','--property=MemorySwapMax=0','--property=KillMode=control-group','--property=TimeoutStopSec=5',
        f'--property=RuntimeMaxSec={left}',f'--property=WorkingDirectory={ROOT}',f'--property=StandardOutput=append:{OUT}/driver.log',
        '--property=StandardError=inherit','--setenv=CUDA_VISIBLE_DEVICES=','systemd-inhibit','--what=sleep:idle','--mode=block',
        '--who=Stage T660','--why=Bounded offline joint learning multibranch relay',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')]
    c=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True);assert c.returncode==0,c.stderr
    with (OUT/'launch.json').open('x') as f:json.dump(dict(epoch=time.time(),unit='stage-t660r',source_remote=receipt['remote_commit'],absolute_all_stop=p['absolute_all_stop']),f,indent=2)
    print(c.stderr.strip())

if __name__=='__main__':{'register':register,'launch':launch}[sys.argv[1]]()
