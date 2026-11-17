"""Prepare only. No run registration, simulation, optimizer or cold eviction."""
from pathlib import Path
import json,gzip,datetime,hashlib,time,copy
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;DOC=ROOT/'stage/docs/overnight_20261117'
def main():
 root=ROOT;here=HERE;doc=DOC;base=json.loads((root/'stage/experiments/t628s_actual_memory_recovery/protocol.json').read_text());old=[j for j in base['evaluation_jobs'] if j['id'] in base['before_success_ids']];new=[j for j in base['evaluation_jobs'] if j['id'] in base['new_sentinel_ids']];fallback=json.loads((root/'stage/runs/t626_multitask_closed_loop/summary.json').read_text())['result']
 with gzip.open(root/'stage/runs/t626_multitask_closed_loop/cases.jsonl.gz','rt') as f:rows=[json.loads(line) for line in f if line.strip()]
 rows=[r for r in rows if r['role']=='FINAL49'];assert len(rows)==49;p=copy.deepcopy(base)
 p.update(id='T629_OFFLINE_NIGHT_RELAY_PREPARED',out=str(root/'stage/runs/t629_offline_night_relay'),authorization_approved=False,authorization_file=str(doc/'AUTHORIZATION.json'),student_initial=base['student_initial'],reference_model=base['student_initial'],training_jobs=old+new,new_ids=[j['id'] for j in new],core_ids=base['training_job_ids'],core_jobs=base['training_jobs'],actor_steps=8,recovery_steps=128,Actor_each_cap=146,actor_lr=.001,seed=62920261117,raw_sigma=[.2,.2],retention_sigma=[.2,.2],decision_stride_native=100,recovery_fallback_model=fallback['candidate'],recovery_fallback_rows=rows,full_bytes=2**21,fresh_bytes=2**21,max_bytes=144*2**20,max_physics=80000000,max_commands=2500,max_main_seconds=30000,final_storage_reserve=8*2**20,finalization_seconds=600,working_RAM_cap_bytes=4*2**30,offline_local_science=True,network_required=False,new_Critic=0,new_optimizer_preflight=0,learning_admission_seconds=0)
 for key,stamp in [('absolute_learning_stop','2026-11-18T07:30:00+08:00'),('absolute_native_stop','2026-11-18T08:15:00+08:00'),('absolute_all_stop','2026-11-18T09:00:00+08:00')]:p[key]=datetime.datetime.fromisoformat(stamp).timestamp()
 profiles=json.loads((root/'stage/experiments/t606r_offline_retention_recovery/protocol.json').read_text())['final_template']['final_jobs'];assert len(profiles)==407;p['profile_jobs']=profiles
 p['templates']={
  'collect':dict(max_physics=4*sum(j['case']['step_cap'] for j in old+new),max_commands=150,main_seconds=3000,learning_seconds=0,audit_seconds=600,storage_reserve=62*2**20),
  'core':dict(max_physics=2434054,max_commands=59,main_seconds=1800,learning_seconds=900,audit_seconds=600,storage_reserve=3*2**20),
  'broad':dict(max_physics=2434054,max_commands=59,main_seconds=2400,learning_seconds=1200,audit_seconds=900,storage_reserve=3*2**20),
  'recovery':dict(max_physics=3600000,max_commands=75,main_seconds=2400,learning_seconds=1500,audit_seconds=900,storage_reserve=24*2**20),
  'recovery_rl':dict(max_physics=3373622,max_commands=75,main_seconds=1800,learning_seconds=900,audit_seconds=600,storage_reserve=14*2**20),
  'profile':dict(max_physics=sum(j['case']['step_cap'] for j in profiles)+sum(j['case']['step_cap'] for j in profiles if j['id'] in p['key_replay_ids']),max_commands=417,main_seconds=3600,learning_seconds=0,audit_seconds=600,storage_reserve=7*2**20)}
 chunks=[];jobs=[]
 for job in profiles:
  if not job['id'].startswith('t502_field_'):continue
  for dx,dyaw in [(-.03,-2.),(0.,2.),(.03,0.)]:
   j=copy.deepcopy(job);j['id']=job['id']+f'_T629_dx{dx}_yaw{dyaw}';j['fullfield_start_pose'][0]+=dx;j['fullfield_start_pose'][2]+=dyaw*3.141592653589793/180.;j['neighborhood_parent_id']=job['id'];jobs.append(j)
 for off in range(0,len(jobs),8):chunks.append(jobs[off:off+8])
 p['neighborhood_chunks']=chunks
 for f in here.glob('*.py'):p['inputs'][str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
 for f in [root/'stage/runs/t626_multitask_closed_loop/proposal.pt.gz',root/'stage/runs/t626_multitask_closed_loop/summary.json',root/'stage/runs/t628s_actual_memory_recovery/audit.json',root/'stage/runs/t628s_actual_memory_recovery/summary.json']:
  assert f.is_file(),f;p['inputs'][str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
 for f,h in p['inputs'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
 (here/'protocol.json').write_text(json.dumps(p,indent=2)+'\n');print('T629_PREPARED_NO_REGISTRATION',len(old+new),'tasks; neighborhood',len(jobs))
if __name__=='__main__':main()
