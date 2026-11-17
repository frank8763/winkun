"""Prepare a finite new daytime protocol. Registration requires fresh authorization."""
from pathlib import Path
import sys,json,datetime,math,ast,hashlib,shutil,time,copy
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;DOC=ROOT/'stage/docs/day_20261117/t608';OUT=ROOT/'stage/runs/t608_current_policy_memory'
read=lambda f:json.loads(Path(f).read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
def save(f,j):Path(f).write_text(json.dumps(j,indent=2,allow_nan=False)+'\n')
def used(d):return sum(f.stat().st_size for f in Path(d).rglob('*') if f.is_file())
def prepare():
 assert not OUT.exists() and not (DOC/'PROPOSAL.json').exists()
 root=read(ROOT/'stage/runs/t606r_offline_retention_recovery/protocol.json');cfg=read(root['actor_configuration']);r=read(ROOT/'stage/runs/t606r_offline_retention_recovery/rl_00/summary.json')['result'];a=read(ROOT/'stage/runs/t606r_offline_retention_recovery/rl_00/audit.json');assert a['state']=='PASS' and r['accepted'] and len(r['check']['success_ids'])==33 and not r['lost_prior'] and len(r['check']['lost_initial'])==6
 assert sha(r['model']['path'])==r['model']['sha256'];pref=read(DOC/'PREFLIGHT.json');assert pref['state']=='PASS_ZERO_NATIVE_ZERO_OPTIMIZER'
 epoch=lambda s:datetime.datetime.fromisoformat(s).timestamp();jobs={j['id']:j for j in cfg['evaluation_jobs']};ids=r['check']['success_ids'];memory=[jobs[i] for i in ids]+[x['request']['job'] for x in cfg['recovery_rows']]
 memory_upper=[]
 for j in memory:
  decisions=math.ceil(j['case']['step_cap']/100)+1;raw=decisions*993+4096;upper=raw+raw//1000+32768;assert upper<2**20;memory_upper.append(upper)
 collect_storage=sum(memory_upper)+36*12*1024+2**20;assert collect_storage<16*2**20
 probe=copy.deepcopy(cfg['evaluation_jobs'][0]);probe['case']['step_cap']=1000
 root.update(id='T608_CURRENT_POLICY_MEMORY_RL',out=str(OUT),student_initial=r['model'],memory_source_model=r['model'],teacher_initial_success_ids=ids,remaining_T5506_ids=r['check']['lost_initial'],arms=['rl'],rounds_per_arm=2,fit_steps=8,rl_steps=8,max_physics=22000000,max_commands=660,max_main_seconds=9000,max_bytes=50*2**20,live_Adam_cap=16,shadow_Adam_cap=16,audit_historical_Adam_cap=16,preflight_discarded_Adam_cap=4,absolute_learning_stop=epoch('2026-11-17T11:30:00+08:00'),absolute_native_stop=epoch('2026-11-17T12:30:00+08:00'),absolute_all_stop=epoch('2026-11-17T13:00:00+08:00'),authorization_approved=False,stress_chunks=0,final_storage_reserve=6*2**20,root_tail_bytes=2*2**20,finalization_seconds=900,transport_seconds=600,source_seconds=120,memory_file_bytes=2**20,memory_manifest=str(OUT/'collect/MEMORY_MANIFEST.json'),target_jobs=[jobs['t502_field_sector0_offset2'],jobs['t502_field_sector1_offset2']])
 root['engineering_template']=dict(main_seconds=300,learning_seconds=0,audit_seconds=120,storage_reserve=2**20,max_physics=4000,max_commands=8,full_bytes=256*1024,probe_job=probe)
 root['optimizer_preflight_template']=dict(main_seconds=300,learning_seconds=150,audit_seconds=120,storage_reserve=2**20,max_physics=0,max_commands=4)
 root['collect_template']=dict(main_seconds=1800,learning_seconds=0,audit_seconds=300,storage_reserve=16*2**20,max_physics=2*sum(j['case']['step_cap'] for j in memory),max_commands=76,memory_jobs=memory)
 root['round_template']=dict(main_seconds=1800,learning_seconds=900,audit_seconds=300,storage_reserve=12*2**20,max_physics=root['rl_template']['max_physics'],max_commands=71)
 root['rl_template'].update(main_seconds=1800,learning_seconds=900,audit_seconds=300,storage_reserve=12*2**20)
 root['final_template'].update(main_seconds=2400,learning_seconds=0,audit_seconds=300,storage_reserve=6*2**20)
 bounds=dict(physics=4000+root['collect_template']['max_physics']+2*root['rl_template']['max_physics']+root['final_template']['max_physics'],commands=8+76+4+2*71+411,main_seconds=300+1800+300+2*1800+2400,storage_bytes=2*2**20+16*2**20+2*12*2**20+6*2**20+2*2**20)
 assert bounds['physics']<=root['max_physics'] and bounds['commands']<=root['max_commands'] and bounds['main_seconds']<=root['max_main_seconds'] and bounds['storage_bytes']<=root['max_bytes']
 assert used(ROOT/'stage/runs')+root['max_bytes']+131072<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
 for f in HERE.glob('*.py'):ast.parse(f.read_text());root['inputs'][str(f)]=sha(f)
 for f in [DOC/'PLAN.md',DOC/'PREFLIGHT.json',ROOT/'stage/runs/t606r_offline_retention_recovery/rl_00/summary.json',ROOT/'stage/runs/t606r_offline_retention_recovery/rl_00/audit.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T606R.json']:
  root['inputs'][str(f)]=sha(f)
 root['inputs'][r['model']['path']]=r['model']['sha256'];deps=read(ROOT/'stage/docs/overnight_20261116/recovery_001/DEPENDENCY_RESTORE.json');root['inputs'].update(deps['source_input_hashes']);root['inputs'][deps['source_protocol']]=deps['source_sha256']
 for f,h in root['inputs'].items():assert sha(f)==h,f
 save(DOC/'PROPOSAL.json',root);save(DOC/'BUDGET.json',dict(state='PREPARED_NOT_AUTHORIZED_NOT_REGISTERED_NOT_LAUNCHED',proposed_caps={k:root[k] for k in ['max_physics','max_commands','max_main_seconds','max_bytes','live_Adam_cap','shadow_Adam_cap','audit_historical_Adam_cap','preflight_discarded_Adam_cap','absolute_learning_stop','absolute_native_stop','absolute_all_stop']},full_physics_commands_main_storage_worst=bounds,memory_each_file_bounds=memory_upper,initial49_success=33,initial_field24_success=13,global_full_reserve=used(ROOT/'stage/runs')+root['max_bytes']+131072,free_GiB=shutil.disk_usage(ROOT).free/2**30,all_phases_remaining_time_admitted_separately=True,full407_conditional=True))
 print('T608_PREPARED_NOT_LAUNCHED',bounds,flush=True)

def authorize():
 assert not OUT.exists();p=read(DOC/'PROPOSAL.json');f=DOC/'AUTHORIZATION.json';assert f.exists();a=read(f);assert a['approved'] and a['all_stop_epoch']==p['absolute_all_stop']
 for k in ['max_physics','max_commands','max_main_seconds','max_bytes']:assert a[k]==p[k]
 assert time.time()+2400<p['absolute_learning_stop'];assert used(ROOT/'stage/runs')+p['max_bytes']+131072<2**30
 for f,h in p['inputs'].items():assert sha(f)==h,f
 p['authorization_approved']=True;p['inputs'][str(DOC/'AUTHORIZATION.json')]=sha(DOC/'AUTHORIZATION.json');OUT.mkdir();blob=json.dumps(p,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);print('T608_REGISTERED_NOT_LAUNCHED')
if __name__=='__main__':globals()[sys.argv[1]]()
