"""Zero-native launch/admission bounds and immutable preflight reuse."""
from pathlib import Path
import ast,json,sys,time,shutil
from datetime import datetime
from zoneinfo import ZoneInfo
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import control as C
import driver as D
R=C.R;ROOT,DOC=C.ROOT,C.DOC

def main():
    for f in HERE.glob('*.py'):compile(f.read_text(),str(f),'exec')
    pref=C.read(DOC/'PREFLIGHT.json')
    for f,h in pref['source'].items():
        if Path(f).name!='control.py':assert C.sha(f)==h
    old=ast.parse((DOC/'PREFLIGHT_CONTROL_SOURCE.py').read_text());new=ast.parse((HERE/'control.py').read_text())
    fn=lambda tree:ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='preflight'))
    assert fn(old)==fn(new)
    p=C.read(C.PRIOR/'protocol.json')
    epoch=lambda h,m:datetime(2026,11,15,h,m,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
    p.update(main_seconds=2400,learning_seconds=1200,audit_seconds=600,final_reserve_seconds=7200,transport_seconds=600,source_seconds=180,absolute_learning_stop=epoch(13,0),absolute_native_stop=epoch(14,30),absolute_all_stop=epoch(15,0),max_physics=19_000_000,max_commands=600,max_main_seconds=12300,max_bytes=int(22.5*2**20),child_reserve_bytes=7*2**20,final_bytes=8*2**20)
    for key in ['restore_rows','keep_rows','neighbor_rows']:p[key]=[{k:r[k] for k in ['job_id','path','sha256','success','collision']} for r in p[key]]
    p['pair_rows']=[{k:{a:v[a] for a in ['path','sha256']} for k,v in pair.items()} for pair in p['pair_rows']]
    p.pop('old8_jobs');p.pop('field_jobs')
    roundphys=2*sum(j['case']['step_cap'] for j in p['training_jobs']+p['evaluation_jobs']);finalphys=sum(j['case']['step_cap'] for j in p['final_jobs'])
    assert 2*roundphys+finalphys+2000==18_971_363
    ledger=dict(physics=2000,commands=8,main_seconds=300.)
    used=R.used(ROOT/'stage/runs');reserve=p['max_bytes']+131072
    assert used+reserve<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    protocol_estimate=len(json.dumps(p,indent=2).encode())+8192
    native_estimate=128*1024;root_other=64*1024
    assert protocol_estimate+native_estimate+root_other<=512*1024
    now=time.time();assert D.admission(p,ledger,2,now,512*1024)
    assert not D.admission(p,dict(ledger,physics=p['max_physics']),2,now,0)
    assert not D.admission(p,dict(ledger,commands=p['max_commands']),2,now,0)
    assert not D.admission(p,ledger,2,p['absolute_native_stop'],0)
    assert not D.admission(p,ledger,2,now,p['max_bytes'])
    assert not D.admission(p,dict(ledger,main_seconds=p['max_main_seconds']),2,now,0)
    assert not R.OUT.exists() and not (HERE/'protocol.json').exists()
    C.save(DOC/'READINESS.json',dict(state='PREPARED_ZERO_NATIVE_NOT_REGISTERED_NOT_LAUNCHED',epoch=now,source={str(f):C.sha(f) for f in HERE.glob('*.py')},preflight_scientific_code_unchanged=True,preflight_discarded_Adam_reused=18,new_native=0,new_optimizer=0,admission_budget_deadline_storage_negative_cases_pass=True,not_full_native_driver_dry_run=True,actual_global_bytes=used,own_bytes=p['max_bytes'],global_worst_bytes=used+reserve,free_bytes=shutil.disk_usage(ROOT).free,root_protocol_upper_bytes=protocol_estimate,engineering_root_reserve_bytes=native_estimate+root_other,round_physics=roundphys,final_physics=finalphys,max_physics=18_971_363,max_commands=595,max_main_seconds=12300,proposed_all_stop='2026-11-15 15:00 Asia/Shanghai',requires_current_window_authorization=True))
    print('T555_READY_NO_NATIVE',used+reserve,protocol_estimate,flush=True)
if __name__=='__main__':main()
