"""Register only after explicit independent computation allowance approval."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t646_offline_night';DOC=ROOT/'stage/docs/overnight_20261118'
sys.path.insert(0,str(ROOT/'stage/experiments/t645_iterative_recovery_relay'))
import control as Resource

def read(f):return json.loads(Path(f).read_text())
def save(f,v):Path(f).write_text(json.dumps(v,indent=2)+'\n')
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()

def main():
    assert not (OUT/'protocol.json').exists() and not (OUT/'launch.json').exists()
    p=read(HERE/'protocol.json');a=read(DOC/'AUTHORIZATION.json')
    assert not p['authorization_approved'] and a['approved']
    for k in ['night_max_physics','night_max_commands','night_Actor_each_cap','night_main_seconds','max_bytes','global_bytes']:assert a[k]==p[k],k
    assert read(DOC/'PREFLIGHT.json')['state']=='PASS_ZERO_NATIVE_ZERO_OPTIMIZER'
    for f,h in p['inputs'].items():assert sha(f)==h,f
    Resource.resources()
    def epoch(s):return datetime.fromisoformat(s).replace(tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
    p.update(authorization_approved=True,registration_epoch=time.time(),absolute_learning_stop=epoch('2026-11-19T07:30:00'),
             absolute_native_stop=epoch('2026-11-19T08:15:00'),absolute_all_stop=epoch('2026-11-19T09:00:00'))
    assert time.time()+3*(p['main_seconds']+p['audit_seconds']+60)+p['profile_seconds']+p['profile_audit_seconds']+p['finalization_reserve_seconds']<p['absolute_all_stop']
    p['inputs'].update({str(DOC/'AUTHORIZATION.json'):sha(DOC/'AUTHORIZATION.json'),str(DOC/'PREFLIGHT.json'):sha(DOC/'PREFLIGHT.json')})
    save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    for name in p['arms']+['profile','coverage']:
        folder=OUT/name;folder.mkdir();(DOC/name).mkdir()
        child=dict(arm=name,_root_sha256=sha(OUT/'protocol.json'),max_bytes=p['child_bytes'])
        if name=='profile':child.update(max_bytes=96*2**20,max_physics=5200000,max_commands=102,main_seconds=p['profile_seconds'])
        if name=='coverage':child.update(max_bytes=2*2**30,max_physics=p['coverage_max_physics'],max_commands=1222,main_seconds=p['coverage_seconds'],fresh_bytes=3*2**20,full_bytes=4*2**20,compact_bytes=32768,absolute_native_stop=epoch('2026-11-19T07:45:00'))
        save(folder/'protocol.json',child)
    save(DOC/'REGISTERED.json',dict(state='REGISTERED_NOT_LAUNCHED',all_stop=p['absolute_all_stop'],registration_epoch=p['registration_epoch']))
    print('REGISTERED_NOT_LAUNCHED',p['absolute_all_stop'])

if __name__=='__main__':main()
