"""Two fixed current-student recovery qualifications. No optimizer or policy change."""
from pathlib import Path
import sys,os,time,fcntl,traceback,copy,ast,json
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t594_current_goal_recovery';DOC=ROOT/'stage/docs/day_20261116/t594'
sys.path.insert(0,str(ROOT/'stage/experiments/t591_joint_representation_pair'));import t591_runtime as R
L,Base,F,G,E,P,torch,np=R.L,R.Base,R.F,R.G,R.E,R.P,R.torch,R.np
read,save,sha,digest=R.read,R.save,R.sha,R.digest
sys.path.insert(0,str(ROOT/'stage/experiments/t550_offline_obstacle_relay'));import t550_teacher as Teacher

def setup(folder):R.OUT=OUT;R.HERE=HERE;R.DOC=DOC;R.setup(folder)
def verify(p,full=False):return R.verify(p,full)

class Session(F.FieldSession):
 def __init__(self,job,*a,**kw):
  super().__init__(job,*a,**kw);self.first=int(job.get('gap_recovery_first',-1));self.teacher_actions=[];self.teacher_scans=[]
  self.params=np.asarray([self.env.speed_gain,self.env.max_wheel,self.env.wheelbase,self.env.yaw_gain,self.env.max_steer],np.float64)
 def step(self,forced=None,owner=0,actor_active=True):
  assert forced is None and owner==0
  if self.first<0:return super().step()
  before=E.N.K.R.integration(self.env).copy();scan=self.env._lidar48().copy();u=Teacher.control('GAP',self.env._pose()[:3],self.goal,scan,self.params)
  assert np.array_equal(before,E.N.K.R.integration(self.env));self.teacher_actions.append(u.copy());self.teacher_scans.append(scan)
  use=len(self.rows)>=self.first;return super().step(u if use else None,owner=int(use),actor_active=True)

def payload(s,tape=None):
 a=R.NORMAL(s,tape)
 if s.first>=0:
  n=len(a['states']);a.update(teacher_actions=np.asarray(s.teacher_actions[:n],np.float32),teacher_scans=np.asarray(s.teacher_scans[:n],np.float32),teacher_vehicle_parameters=s.params.copy(),teacher_first_declared=np.asarray(s.first,np.int64))
 return a

def worker(pipe,index,p,counts):
 G.T.Session=Session;G.B.native_setup=F.field_setup;P.frozen=L.frozen;E.payload=payload;G.cpu_worker(pipe,index,p,counts)

class Pool(R.Pool):
 def __init__(self,p,counts,guard):
  self.p=p;self.guard,self.children,self.pipes=guard,[],[];ctx=G.R.M.mp.get_context('spawn')
  try:
   for i in range(2):
    a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
   self.ready=[self.receive(i) for i in range(2)];assert all(x['state']=='READY' for x in self.ready);save(Path(p['out'])/'worker_generations.json',[dict(parent=os.getpid(),workers=[x.pid for x in self.children],epoch=time.time(),CPU=[14,15],GPU=False)])
  except BaseException:self.close();raise

def mixed_audit(p,row,a,source=None):
 k=int(a['teacher_first_declared']);assert len(a['states'])>k>0
 if source is not None:
  for name in ['states','next_states','actions','native_counts','poses','next_poses','hidden_before','history_poses','history_goals','history_times','locations','raw_actions','logp','stochastic']:
   assert np.array_equal(a[name][:k],source[name][:k]),name
  for name in ['states','poses','hidden_before','history_poses','history_goals','history_times']:assert np.array_equal(a[name][k],source[name][k]),name
  assert np.array_equal(a['initial_native_integration'],source['initial_native_integration'])
 assert np.array_equal(a['ownership'],np.arange(len(a['states']))>=k) and not a['stochastic'].any()
 assert np.array_equal(a['states'][1:,117:119],a['actions'][:-1])
 for i,(pose,scan) in enumerate(zip(a['poses'],a['teacher_scans'])):
  u=Teacher.control('GAP',pose,a['task_goal_xy'],scan,a['teacher_vehicle_parameters']);assert np.array_equal(u,a['teacher_actions'][i])
  if i>=k:assert np.array_equal(u,a['actions'][i])
 proposals=np.clip(a['raw_actions'],-1,1).astype(np.float32);assert np.array_equal(proposals[:k],a['actions'][:k])
 F.M.audit_full_history(p,dict(row,teacher=False),dict(a,actions=proposals,ownership=np.zeros(len(a['states']),np.uint8)))

def prepare():
 assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True)
 prev=ROOT/'stage/runs/t591_joint_representation_pair';p=copy.deepcopy(read(prev/'protocol.json'));s=read(prev/'summary.json');launch=read(prev/'launch.json');diag=ROOT/'stage/runs/t592_five_loss_attribution'
 assert read(diag/'audit.json')['state']=='PASS' and read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T592.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
 assert all(not Path('/proc',str(i)).exists() for i in read(diag/'SEALED.json')['scientific_pids'])
 ids=['t502_field_sector1_offset1','t502_field_sector0_offset3'];jobs=[next(j for j in p['evaluation_jobs'] if j['id']==i) for i in ids]
 rows=G.rows(prev/'trainable_representation');source=[next(x for x in rows if x['job_id']==i and x['role']=='ORIGINAL_START49') for i in ids]
 near=next(x for x in G.rows(diag) if x['job_id']==ids[0] and x['role']=='FROZEN_after600');a=E.N.S.arrays(near['path']);dist=np.linalg.norm(a['poses'][:,:2]-a['task_goal_xy'],axis=1);k=max(1,int(dist.argmin())-1)
 assert not source[0]['success'] and not source[1]['success'] and ids[1] not in p['initial_success_ids']
 firsts=[k,1];cap=4*sum(j['case']['step_cap'] for j in jobs);prior=read(diag/'summary.json');used_phys=s['physics']+prior['physics'];used_cmd=s['commands']+prior['commands']
 assert used_phys+cap<=6025900 and used_cmd+12<=136
 end=launch['all_stop']-4
 p.update(id='T594_CURRENT_STUDENT_NEAR_GOAL_AND_ORIGINAL_FAILED_FIELD_RECOVERY',out=str(OUT),jobs=jobs,source_rows=source,teacher_firsts=firsts,initial=s['result'][1]['proposal'],
  main_seconds=300,learning_seconds=0,audit_seconds=120,transport_seconds=120,source_seconds=90,absolute_learning_stop=time.time(),absolute_native_stop=end-360,absolute_all_stop=end,
  max_physics=cap,max_commands=12,max_bytes=8*2**20,child_bytes=8*2**20,inputs={},protected_files={},new_Actor_Adam_cap=0,shadow_Adam_cap=0,audit_discarded_Adam_cap=0,preflight_discarded_Adam_cap=0,new_RL=0,new_Critic=0,authorization_approved=True,
  user_authorization='2026-11-16 user 好，下一步开始 after approved training-only recovery research. Two fixed teacher qualifications0learning within remaining original90minute physical/command budget, no deadline extension.',
  scope='Fixed GAP training teacher at candidate closest-approach minus1decision for field1off1; field0off3 original failure at first completed decision.2compact baselines+2exact,2full mixed+2exact. No noise/teacher/time grid, Actor update or automatic next training.')
 upper=4*p['full_bytes']+2*p['compact_bytes']+2*2**20;glob=R.used(ROOT/'stage/runs')+p['max_bytes']+131072
 assert upper<=p['max_bytes'] and glob<2**30 and time.time()+660<end and time.time()+300<p['absolute_native_stop']
 setup(DOC);q=L.frozen(p,p['initial']['path']);assert q.phash()==p['initial']['policy_hash'];before=digest(q.state_dict())
 for row in [near]+p['recovery_rows']:
  z=E.N.S.arrays(row['path']);x=torch.tensor(z['states'][:16]);h=None;uu=[];q.sample=False;q.reset_noise()
  with torch.no_grad():
   expected=q.sequence(x).clamp(-1,1).float()
   for xx in x:u,_,h,_=q.step(xx[None],h);uu.append(u[0])
  assert float((torch.stack(uu)-expected).abs().max())<=2e-6
 assert digest(q.state_dict())==before and not torch.cuda.is_initialized()
 for f in HERE.glob('*.py'):ast.parse(f.read_text())
 save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',teacher_firsts=firsts,near_source_sha256=near['sha256'],near_closest_pose_decision=int(dist.argmin()),unchanged_policy_hash=q.phash(),own_upper=upper,global_worst=glob))
 for mod in list(sys.modules.values()):
  f=getattr(mod,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
 for f in [*HERE.glob('*.py'),DOC/'PLAN.md',DOC/'PREFLIGHT.json',Path(near['path']),prev/'audit.json',diag/'audit.json',Path(p['actor_configuration'])]+[Path(j['exact_snapshot']) for j in jobs]+[Path(r['path']) for r in p['recovery_rows']]:p['inputs'][str(f)]=sha(f)
 p['inputs'][p['initial']['path']]=p['initial']['sha256'];p['protected_files'][p['initial']['path']]=p['initial']['sha256']
 save(DOC/'AUTHORIZATION.json',dict(user_reply='好，下一步开始；然后再回答我用ppo与sac有什么区别，之前好像用过sac吧？',prior_physics=used_phys,new_physics=cap,combined_max_physics=used_phys+cap,prior_commands=used_cmd,new_commands=12,combined_max_commands=used_cmd+12,old_physical_limit=6025900,old_command_limit=136,all_stop=end,new_optimizer=0))
 p['inputs'][str(DOC/'AUTHORIZATION.json')]=sha(DOC/'AUTHORIZATION.json');save(DOC/'BUDGET.json',dict(max_physics=cap,max_commands=12,own_bytes=p['max_bytes'],own_upper=upper,global_worst=glob,main_seconds=300,audit_seconds=120,backup_seconds=120,source_seconds=90,total_admission_seconds=660,all_stop=end))
 OUT.mkdir();save(OUT/'protocol.json',p);save(HERE/'protocol.json',p);verify(dict(p,_root_sha256=sha(OUT/'protocol.json')),True);print('T594_REGISTERED_NO_LAUNCH',cap,firsts,glob,flush=True)

def main():
 p=dict(read(OUT/'protocol.json'),_root_sha256=sha(OUT/'protocol.json'));setup(OUT);verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=Base.Runtime(p);state='FAILED';error=None;result=None
 try:
  rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
  vv=rt.request(p['jobs'],p['initial'],'CURRENT_STUDENT_SOURCE_REBUILD');rr=[r for a,r in vv]
  for (a,r),old in zip(vv,p['source_rows']):F.M.audit_full_history(p,r,a);assert r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
  rt.request(p['jobs'],p['initial'],'EXACT_CURRENT_STUDENT_SOURCE',reference=rr)
  jobs=[dict(j,gap_recovery_first=k) for j,k in zip(p['jobs'],p['teacher_firsts'])];mixed=rt.request(jobs,p['initial'],'TRAINING_ONLY_FIXED_GAP_RECOVERY',full=True);out=[]
  for (a,r),(b,_) in zip(mixed,vv):
   mixed_audit(p,r,a,b);safe=R.safe(r);out.append(dict(job_id=r['job_id'],teacher_first=int(a['teacher_first_declared']),qualified_safe_goal=safe,collision=r['collision'],timeout=r['timeout'],remaining_m=r['remaining_m'],row=r,not_autonomous_student_success=True))
  rt.request(jobs,p['initial'],'EXACT_GAP_RECOVERY',full=True,reference=[r for a,r in mixed]);result=dict(outcomes=out,qualified_rows=[x['row'] for x in out if x['qualified_safe_goal']],new_Actor=0,new_Adam=0,new_RL=0,full407=None,globally_promoted=False);state='COMPLETE'
 except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,no_restart=True,raw_preserved=True))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=2
  save(OUT/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
 assert state=='COMPLETE',error

def audit():
 p=dict(read(OUT/'protocol.json'),_root_sha256=sha(OUT/'protocol.json'));setup(OUT);verify(p,True);G.stopped(OUT);assert not (OUT/'audit.json').exists();s=read(OUT/'summary.json');assert s['state']=='COMPLETE';rows=G.rows(OUT);reps=G.rows(OUT,'replays.jsonl');assert len(rows)==len(reps)==4
 for r in rows:
  a=E.N.S.arrays(r['path']);assert sha(r['path'])==r['sha256'] and int(a['native_counts'].sum())==r['physics'] and all(digest(v)==r['array_hashes'][k] for k,v in a.items())
  if r['full']:mixed_audit(p,r,a)
 for i in range(2):
  assert rows[i]['array_hashes']==p['source_rows'][i]['array_hashes'] and rows[i]['integration_sha256']==p['source_rows'][i]['integration_sha256']
  mixed=E.N.S.arrays(rows[i+2]['path']);prefix=np.clip(mixed['raw_actions'][:p['teacher_firsts'][i]],-1,1).astype(np.float32);assert np.array_equal(prefix,mixed['actions'][:p['teacher_firsts'][i]])
  assert R.safe(rows[i+2])==s['result']['outcomes'][i]['qualified_safe_goal']
 by={r['path']:r for r in rows}
 for r in reps:assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256']
 assert s['commands']==12 and s['physics']==sum(r['physics'] for r in rows+reps)<=p['max_physics']
 save(OUT/'audit.json',dict(state='PASS',new_native=0,new_optimizer=0,all4_exact_native_replays=True,full_mixed_teacher_actions_history_density_verified=True,prefix_physical_match_live_verified=True,compact_baseline_no_independent_full_pose_reconstruction=True,all_models_unchanged=True,physics=s['physics'],commands=s['commands']));print('T594_AUDIT_PASS',flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
