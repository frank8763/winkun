"""One feedback-state repair of a frozen12/24 research candidate.

Reuse the audited20 full histories from T543. Validate one new terminal
feedback teacher continuation in native MuJoCo before action imitation.
Final evaluation has no teacher at runtime.
"""
from pathlib import Path
import copy,fcntl,json,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t544_terminal_feedback_repair';DOC=ROOT/'stage/docs/day_20261114'
sys.path.insert(0,str(HERE));import t544_learning as L
sys.path.insert(0,str(ROOT/'stage/experiments/t542_offline_field_relay'));import t542_runtime as Base
F,G,E,P,np,torch=Base.F,L.G,Base.E,L.P,L.np,L.torch
read,save,sha,digest=L.read,L.save,L.sha,L.digest
BASE_PAYLOAD=E.payload

def used(folder):return sum(f.stat().st_size for f in Path(folder).rglob('*') if f.is_file())
def verify(p,full=False):
 root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
 assert time.time()<root['absolute_all_stop'] and used(OUT)<=root['max_bytes'] and used(ROOT/'stage/runs')-used(OUT)+root['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized() and set(os.sched_getaffinity(0))<={14,15}
 for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f

def configuration(root):
 p=read(root['actor_configuration']);p.update(root);p['out']=str(OUT);p['evaluation_jobs']=p['old8_jobs']+p['field_jobs'];assert p['out']==str(OUT)
 return p
Base.OUT=OUT;Base.HERE=HERE;Base.DOC=DOC;Base.verify=verify

def setup():
 Base.setup(OUT);F.OUT=OUT;F.HERE=HERE;F.verify=verify;F.R.OUT=OUT;F.R.HERE=HERE;F.R.verify=verify;P.frozen=L.A.frozen

def feedback(p,pose,goal,parameters):
 import math
 speed_gain,max_wheel,wheelbase,yaw_gain,max_steer=map(float,parameters)
 assert speed_gain>0 and max_wheel>0 and wheelbase>0 and yaw_gain>0 and 0<max_steer<math.pi/2
 d=float(np.linalg.norm(np.asarray(goal)-np.asarray(pose)[:2]));angle=math.atan2(goal[1]-pose[1],goal[0]-pose[0]);e=math.atan2(math.sin(angle-pose[2]),math.cos(angle-pose[2]))
 steer=float(np.clip(math.atan(p['physical_wheelbase_m']*2*math.sin(e)/max(d,p['curvature_distance_floor_m'])),-max_steer,max_steer));vmax=speed_gain*max_wheel
 desired=min(p['nominal_speed_m_s'],vmax)*min(1.,d/p['slow_distance_m'])*max(p['minimum_heading_speed_fraction'],math.cos(e))
 u=np.asarray([np.clip(2*math.sqrt(desired/vmax)-1,-1,1),np.clip(steer/max_steer,-1,1)],np.float32)
 return u,steer,desired

class RecoverySession(F.FieldSession):
 def __init__(self,job,*args,**kwargs):
  super().__init__(job,*args,**kwargs);self.teacher_actions=[]
  from training.e2e_phase10_lidar_v7_curriculum.phase10_strict_mujoco_env import Phase10StrictMuJoCoEnv
  from training.e2e_phase10_lidar_v7_curriculum.stage9_runtime_fullplant_fix_v5_2 import WHEELBASE_M
  assert type(self.env)._policy_to_commands is Phase10StrictMuJoCoEnv._policy_to_commands and float(WHEELBASE_M)==self.q.p['physical_wheelbase_m']
  self.teacher_parameters=np.asarray([self.env.speed_gain,self.env.max_wheel,self.env.wheelbase,self.env.yaw_gain,self.env.max_steer],np.float64)
 def step(self,forced=None,owner=0,actor_active=True):
  assert forced is None and owner==0;before=E.N.K.R.integration(self.env).copy();u,steer,speed=feedback(self.q.p,self.env._pose()[:3],self.goal,self.teacher_parameters);mapped=self.env._policy_to_commands(u)
  assert abs(mapped[0]-steer)<1e-7 and abs(mapped[2]-speed)<1e-7 and np.array_equal(before,E.N.K.R.integration(self.env));self.teacher_actions.append(u.copy());use=len(self.rows)>=self.q.p['teacher_first']
  return super().step(u if use else None,owner=int(use),actor_active=True)

def payload(s,tape=None):
 a=BASE_PAYLOAD(s,tape);n=len(a['states']);a['teacher_actions']=np.asarray(s.teacher_actions[:n],np.float32);a['teacher_vehicle_parameters']=s.teacher_parameters.copy();return a

def worker(pipe,index,p,counts):
 G.T.Session=RecoverySession;G.B.native_setup=F.field_setup;P.frozen=L.A.frozen;E.payload=payload;G.cpu_worker(pipe,index,p,counts)

class RecoveryPool(F.Pool):
 def __init__(self,p,counts,guard):
  self.p=p;self.guard,self.children,self.pipes=guard,[],[];ctx=G.R.M.mp.get_context('spawn')
  try:
   for i in range(2):
    a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
   self.ready=[self.receive(i) for i in range(2)];assert all(v['state']=='READY' for v in self.ready)
   f=OUT/'worker_generations.json';old=read(f) if f.exists() else [];old.append(dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time(),CPU=[14,15],GPU=False,teacher_training_only=True));save(f,old)
  except BaseException:self.close();raise
 def receive(self,i):
  value=super().receive(i)
  if isinstance(value,tuple):value[1].update(teacher=True,teacher_first=self.p['teacher_first'],teacher_contract=self.p['controller_contract'],training_mixture_NOT_autonomous=True)
  return value

def prefix_rule(p,a):
 d=np.linalg.norm(a['poses'][:,:2]-a['task_goal_xy'],axis=1);hit=np.flatnonzero(d<=p['terminal_trigger_m']);assert len(hit)>0;first=int(hit[0]);assert first>0
 return first,dict(teacher_first=first,prefix_seconds=float(a['native_counts'][:first].sum()*.002),distance_at_intervention_m=float(d[first]),rule='FIRST_ACTUAL_STUDENT_DISTANCE_AT_MOST_1M_NO_WINDOW_GRID')

def recovery_audit(p,row,a,baseline):
 k=p['teacher_first']
 for name in ['states','next_states','actions','native_counts','poses','next_poses','hidden_before','history_poses','history_goals','history_times','locations']:assert np.array_equal(a[name][:k],baseline[name][:k]),name
 for name in ['states','poses','hidden_before','history_poses','history_goals','history_times']:assert np.array_equal(a[name][k],baseline[name][k]),name
 owner=np.zeros(len(a['states']),np.uint8);owner[k:]=1;assert np.array_equal(owner,a['ownership']) and not a['stochastic'].any();assert np.array_equal(a['initial_native_integration'],baseline['initial_native_integration'])
 for i,x in enumerate(a['poses']):
  u,_,_=feedback(p,x,a['task_goal_xy'],a['teacher_vehicle_parameters']);assert np.array_equal(u,a['teacher_actions'][i])
 F.M.audit_full_history(p,row,a)

def assess(p,rows):
 c=Base.assess(p,rows);ids=set(c['field_success_ids']);before=set(p['initial_check']['field_success_ids']);c.update(lost_prior_field_ids=sorted(before-ids),gained_vs_T507=sorted(ids-set(p['accepted_baseline_field_ids'])),panel_repaired=c['old8']['success']==8 and before<=ids)
 c['accepted']=bool(c['panel_repaired'] and c['field24']['contact']<=p['max_accepted_field_contacts'] and c['field24']['invalid']==c['field24']['censored']==c['old8']['invalid']==c['old8']['censored']==0)
 return c

def main():
 root=read(OUT/'protocol.json');p=configuration(root);setup();verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=Base.Runtime(p);state='FAILED';error=None;result=None;obj=shadow=before=None
 try:
  source=G.rows(Path(p['audited_source_folder']));assert len(source)==21
  sr=source[:20];successful=[r for r in sr if r['success'] and not r['collision']];assert len(successful)==19
  for r in sr:assert sha(r['path'])==r['sha256']
  oldrow=next(r for r in sr if r['job_id']==p['probe_job']['id']);baseline=E.N.S.arrays(oldrow['path']);first,derived=prefix_rule(p,baseline);p['teacher_first']=first;save(OUT/'derived_prefix.json',derived)
  rt.pool=RecoveryPool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
  vr=rt.request([p['probe_job']],p['initial'],'ACTUAL_TERMINAL_FEEDBACK_TEACHER',full=True);a,row=vr[0];recovery_audit(p,row,a,baseline);rt.request([p['probe_job']],p['initial'],'EXACT_FEEDBACK_TEACHER',full=True,reference=[row]);eligible=bool(row['success'] and not row['collision'])
  result=dict(source_model=p['initial'],reused_audited_success_histories=19,derived_prefix=derived,teacher_qualified=eligible,teacher_result=dict(success=row['success'],contact=row['collision'],timeout=row['timeout'],remaining_m=row['remaining_m']),Actor_Adam=0,Value_updates=0,new_policy_gradient_updates=0,update=None,model=None,check=None,accepted=False,old275=None,six54=None,eight54=None,fresh16=None,reload78=None)
  rt.pool.close();rt.pool=None;rt.commands+=2
  if eligible:
   obj=L.make(p);shadow=L.make(p);before=L.state(obj);refs=L.references(successful);rt.role='FEEDBACK_MIR_IMITATION'
   aa=L.fit(obj,a,p,refs,lambda:rt.guard(True),Base.event('live'));bb=L.fit(shadow,a,p,refs,lambda:rt.guard(True),Base.event('shadow'));assert aa==bb and digest(L.state(obj))==digest(L.state(shadow))
   model=L.export(obj,p,OUT/'proposal.pt.gz');assert L.A.frozen(p,model['path']).phash()==model['policy_hash']
   rt.pool=F.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard);rt.commands+=2
   vals=rt.request(p['evaluation_jobs'],model,'PURE_STUDENT_ALL32');er=[r for a,r in vals];rt.request(p['evaluation_jobs'],model,'EXACT_ALL32',reference=er);check=assess(p,er)
   result.update(update=aa,Actor_Adam=aa['Actor_Adam'],virtual_discarded_Adam=aa['virtual_discarded_Adam'],model=model,check=check,accepted=check['accepted'])
   if not result['accepted']:L.load(obj,before);L.load(shadow,before)
   result['retained_joint_hash']=digest(L.state(obj))
  result['reason']='JOINT_REPAIR_RESEARCH_GATE_PASSED' if result['accepted'] else 'REPAIR_REJECTED_ROLLBACK' if eligible else 'TERMINAL_FEEDBACK_NOT_QUALIFIED_ZERO_LEARNING';state='COMPLETE'
 except BaseException:
  error=traceback.format_exc()
  if before is not None:
   for o in [obj,shadow]:L.load(o,before)
  save(OUT/'failure.json',dict(error=error,all_raw_preserved=True,no_restart=True))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=2
  save(OUT/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
 assert state=='COMPLETE',error

def audit():
 p=configuration(read(OUT/'protocol.json'));setup();verify(p,True);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists();G.stopped(OUT)
 rows=G.rows(OUT);rep=G.rows(OUT,'replays.jsonl');source=G.rows(Path(p['audited_source_folder']))[:20];baseline=E.N.S.arrays(next(r['path'] for r in source if r['job_id']==p['probe_job']['id']));first,derived=prefix_rule(p,baseline);assert derived==read(OUT/'derived_prefix.json');p['teacher_first']=first
 for r in source:
  assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert all(digest(v)==r['array_hashes'][k] for k,v in a.items());F.M.audit_full_history(p,r,a)
 for r in rows:
  assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert all(digest(v)==r['array_hashes'][k] for k,v in a.items());assert int(a['native_counts'].sum())==r['physics']
  if r['full']:
   if r['teacher']:recovery_audit(p,r,a,baseline)
   else:F.M.audit_full_history(p,r,a)
 by={r['path']:r for r in rows}
 for r in rep:assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256']
 result=s['result'];teacher=rows[0];assert result['teacher_qualified']==bool(teacher['success'] and not teacher['collision']);steps=virtual=0
 if result['teacher_qualified']:
  obj=L.make(p);before=L.state(obj);u=L.fit(obj,E.N.S.arrays(teacher['path']),p,L.references([r for r in source if r['success']]),lambda:None,lambda *_:None);steps=u['Actor_Adam'];virtual=u['virtual_discarded_Adam'];assert u==result['update'] and digest(L.state(obj))==result['model']['joint_hash'] and sha(result['model']['path'])==result['model']['sha256'];check=assess(p,rows[1:]);assert check==result['check'] and len(rows)==33
  if not result['accepted']:L.load(obj,before)
  assert digest(L.state(obj))==result['retained_joint_hash']
 else:assert len(rows)==1 and not G.rows(OUT,'optimizer_calls.jsonl')
 expected=[[who,stage,k] for who in ['live','shadow'] for k in range(steps) for stage in ['VIRTUAL_BEFORE','VIRTUAL_ROLLED_BACK','BEFORE','AFTER']];assert G.rows(OUT,'optimizer_calls.jsonl')==expected
 assert len(rows)==len(rep);assert s['physics']==sum(r['physics'] for r in rows+rep)<=p['max_physics'];assert s['commands']==len(rows)+len(rep)+(8 if steps else 4)<=p['max_commands']
 save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,discarded_historical_Actor_Adam=steps,discarded_virtual_Adam=virtual,teacher_actual_feedback_commands_and_intervention_rebuilt=True,all_saved_full_training_density_rewards_hidden_rebuilt=True,Actor_Adam_and_virtual_MIR_ranking_rollback_exact=True,compact_evaluation_full120_NOT_reconstructed_offline=True,all_native_replay_full_hashes_and_integration_exact=True,physics=s['physics'],commands=s['commands']));print('T544_FULL_REPAIR_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
