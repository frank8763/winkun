"""One bounded offline recovery/learning child, with exact evidence and independent audit."""
from pathlib import Path
import sys,os,time,shutil,fcntl,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t577_offline_recovery_night';DOC=ROOT/'stage/docs/overnight_20261115'
sys.path.insert(0,str(HERE));import t577_learning as L
sys.path.insert(0,str(ROOT/'stage/experiments/t575_complete_neural_recovery'));import t575_runtime as Old
sys.path.insert(0,str(ROOT/'stage/experiments/t574_neural_recovery_handoff'));import t574_runtime as Neural
Base,F,G,E,P,torch,np=Old.Base,Old.F,Old.G,Old.E,Old.P,L.torch,L.np
read,save,sha,digest=L.read,L.save,L.sha,L.digest
NORMAL=Old.Old.Old.Prior.NORMAL

def used(d):return sum(f.stat().st_size for f in Path(d).rglob('*') if f.is_file() and not f.is_symlink())
def verify(p,full=False):
 r=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json');assert time.time()<r['absolute_all_stop'] and used(OUT)<=r['max_bytes'] and used(ROOT/'stage/runs')-used(OUT)+r['max_bytes']+131072<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized() and set(os.sched_getaffinity(0))<={14,15}
 for f,h in (r['inputs'] if full else r['protected_files']).items():assert sha(f)==h,f

def plan(child):
 p=read(OUT/'protocol.json');cfg=read(p['actor_configuration']);cfg.update(p);c=read(OUT/child/'protocol.json');assert c['_root_sha256']==sha(OUT/'protocol.json');cfg.update(c);cfg['out']=str(OUT/child);return cfg

def setup(folder):
 Old.OUT=folder;Old.HERE=HERE;Old.verify=verify;Old.setup();Base.ACTIVE=folder;P.frozen=L.frozen;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

class Session(F.FieldSession):
 def __init__(self,job,*args,**kwargs):
  super().__init__(job,*args,**kwargs);self.first=int(job.get('recovery_first',-1));self.teacher_labels=[];self.th=None;self.boundary=np.zeros((1,1,64),np.float64)
  if self.first>=0:self.teacher=L.frozen(self.q.p,self.q.p['neural_teacher']['path']);self.teacher.sample=False;self.teacher.reset_noise()
 def step(self,forced=None,owner=0,actor_active=True):
  assert forced is None and owner==0
  if self.first<0:return super().step()
  k=len(self.rows);before=E.N.K.R.integration(self.env).copy();x=self.packet().copy()
  if k==self.first:self.boundary=np.zeros((1,1,64)) if self.th is None else self.th.numpy().copy()
  with torch.no_grad():u,_,self.th,_=self.teacher.step(torch.tensor(x[None]),self.th)
  assert np.array_equal(before,E.N.K.R.integration(self.env));self.teacher_labels.append(u[0].numpy().copy());use=k>=self.first;return super().step(u[0].numpy().copy() if use else None,owner=int(use),actor_active=True)

def payload(s,tape=None):
 a=NORMAL(s,tape)
 if s.first>=0:a.update(teacher_actions=np.asarray(s.teacher_labels[:len(a['states'])],np.float32),teacher_takeover_hidden_before=s.boundary.copy(),teacher_first_declared=np.asarray(s.first,np.int64))
 return a

def worker(pipe,index,p,counts):
 G.T.Session=Session;G.B.native_setup=F.field_setup;P.frozen=L.frozen;E.payload=payload;G.cpu_worker(pipe,index,p,counts)
class Pool(F.Pool):
 def __init__(self,p,counts,guard):
  self.p=p;self.guard,self.children,self.pipes=guard,[],[];ctx=G.R.M.mp.get_context('spawn')
  try:
   for i in range(2):
    a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
   self.ready=[self.receive(i) for i in range(2)];assert all(v['state']=='READY' for v in self.ready);f=Path(p['out'])/'worker_generations.json';save(f,(read(f) if f.exists() else [])+[dict(parent=os.getpid(),workers=[x.pid for x in self.children],epoch=time.time(),CPU=[14,15],GPU=False)])
  except BaseException:self.close();raise
 def receive(self,i):
  v=super().receive(i)
  if isinstance(v,tuple):
   a,r=v;teacher='teacher_first_declared' in a;r.update(teacher=teacher)
   if teacher:r.update(teacher_first=int(a['teacher_first_declared']),teacher_model=self.p['neural_teacher'],training_only=True)
  return v

def recovery_audit(p,r,a,baseline=None):
 k=int(a['teacher_first_declared']);pp=dict(p,teacher_first=k);assert len(a['states'])>k and not a['ownership'][:k].any() and a['ownership'][k:].all() and not a['stochastic'].any()
 if baseline is not None:
  for key in ['states','next_states','actions','native_counts','poses','next_poses','hidden_before','history_poses','history_goals','history_times','locations']:assert np.array_equal(a[key][:k],baseline[key][:k]),key
  for key in ['states','poses','hidden_before','history_poses','history_goals','history_times']:assert np.array_equal(a[key][k],baseline[key][k]),key
  assert np.array_equal(a['initial_native_integration'],baseline['initial_native_integration'])
 teacher=L.frozen(p,p['neural_teacher']['path']);teacher.sample=False;teacher.reset_noise();th=None
 with torch.no_grad():
  for i,x in enumerate(a['states']):
   if i==k:assert np.array_equal(a['teacher_takeover_hidden_before'],np.zeros((1,1,64)) if th is None else th.numpy())
   u,_,th,_=teacher.step(torch.tensor(x[None]),th);assert np.array_equal(u[0].numpy(),a['teacher_actions'][i])
 F.M.audit_full_history(pp,r,a)

def assess(p,rows):
 assert len(rows)==32 and [r['job_id'] for r in rows]==[j['id'] for j in p['evaluation_jobs']];ids={r['job_id'] for r in rows if r['success'] and not r['collision'] and r['field_detail']['field_qualified_success']};keep=set(p['before_success_ids']);return dict(old8=G.R.M.tally(rows[:8]),field24=G.R.M.tally(rows[8:]),success_ids=sorted(ids),lost_prior=sorted(keep-ids),gained=sorted(ids-keep),reference23_missing=sorted(set(p['reference23_success_ids'])-ids),accepted=keep<=ids and len(ids)>len(keep),complete32_safe=len(ids)==32)
def event(w):return lambda s,k:Base.append('optimizer_calls.jsonl',[w,s,k])

def main(child):
 p=plan(child);folder=Path(p['out']);setup(folder);verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=Base.Runtime(p);state='FAILED';error=None;result=None;obj=shadow=before=None
 try:
  rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
  if p['kind']=='engineering':
   jobs=[dict(p['probe_job']),dict(p['probe_job'],recovery_first=2)];vv=rt.request(jobs,p['initial'],'ENGINEERING_SHORT_NORMAL_AND_TEACHER',full=True);rr=[r for a,r in vv]
   for a,r in vv:
    if r['teacher']:recovery_audit(p,r,a)
    else:F.M.audit_full_history(p,r,a)
   rt.request(jobs,p['initial'],'EXACT_ENGINEERING_SHORT',full=True,reference=rr)
   fresh=rt.request([p['probe_job']]*4,p['initial'],'ENGINEERING_REAL_STOCHASTIC_DENSITY',seeds=[[577,0,k] for k in range(4)],full=True);rt.request([p['probe_job']]*4,p['initial'],'EXACT_ENGINEERING_STOCHASTIC',seeds=[[577,0,k] for k in range(4)],full=True,reference=[r for a,r in fresh])
   sys.path.insert(0,str(ROOT/'stage/experiments/t519_fresh_full_actor_rl'));import t519_learning as Credit
   _,density=Credit.peer_batch((L.frozen(p,p['initial']['path']),None),[a for a,r in fresh],p);result=dict(kind='engineering',new_Adam=0,new_RL=0,actual_native_stochastic_density=density,short_cap_not_navigation_success=True)
  elif p['kind']=='recovery':
   vv=rt.request([p['target_job']],p['query_model'],'ACTUAL_QUERY_STUDENT_FAILURE',full=False);a,r=vv[0];F.M.audit_full_history(p,r,a);rt.request([p['target_job']],p['query_model'],'EXACT_QUERY_FAILURE',reference=[r]);k=0;ref=p.get('reference_row');rule='HALFWAY_BEFORE_ACTUAL_TERMINATION'
   if ref:
    old=E.N.S.arrays(ref['path']);n=min(len(a['poses']),len(old['poses']));delta=a['poses'][:n]-old['poses'][:n];pos=np.linalg.norm(delta[:,:2],axis=1);ang=np.abs(np.arctan2(np.sin(delta[:,2]),np.cos(delta[:,2])));hits=np.flatnonzero((pos>=.03)|(ang>=np.deg2rad(3.)));k=max(0,int(hits[0])-1) if len(hits) else max(0,len(a['states'])//2);rule='BEFORE_FIRST_3CM_OR_3DEG_OR_HALFWAY'
   else:k=max(0,len(a['states'])//2)
   assert k<len(a['states']);witness={key:digest(a[key][:k]) for key in ['states','next_states','actions','native_counts','poses','next_poses','hidden_before','history_poses','history_goals','history_times','locations']};save(folder/'prefix_witness.json',dict(teacher_first=k,prefix_hashes=witness,initial_integration=digest(a['initial_native_integration']),source_failure_array_digest=r['array_digest']));job=dict(p['target_job'],recovery_first=k);vv=rt.request([job],p['query_model'],'VERIFIED_FULL_NEURAL_RECOVERY',full=True);z,row=vv[0];recovery_audit(p,row,z,a);rt.request([job],p['query_model'],'EXACT_FULL_NEURAL_RECOVERY',full=True,reference=[row]);qualified=bool(row['success'] and not row['collision'] and row['field_detail']['field_qualified_success']);result=dict(kind='recovery',teacher_qualified=qualified,correction=row if qualified else None,teacher_first=k,prefix_rule=rule,query_model=p['query_model'],new_Adam=0,new_RL=0)
  elif p['kind']=='round':
   rt.pool.close();rt.pool=None;rt.commands+=2;data=L.dataset(p);obj=L.make(p);shadow=L.make(p);before=L.state(obj);rt.role='JOINT_RECOVERY_'+p['arm'];b=L.bc(obj,data,p,lambda:rt.guard(True),event('live'));b2=L.bc(shadow,data,p,lambda:rt.guard(True),event('shadow'));assert b==b2 and digest(L.state(obj))==digest(L.state(shadow));behavior=L.export(obj,p,folder/'behavior.pt.gz');u=None
   rt.commands+=2;rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
   if p['arm']=='rl':
    vv=rt.request([p['target_job']]*4,behavior,'FRESH_POST_BC_CURRENT_POLICY',seeds=p['training_seeds'],full=True);sr=[r for a,r in vv];rt.request([p['target_job']]*4,behavior,'EXACT_CURRENT_POLICY_TRAINING',seeds=p['training_seeds'],full=True,reference=sr);arrays=[a for a,r in vv]
    for a,r in vv:F.M.audit_full_history(p,r,a)
    u=L.rl(obj,arrays,data,p,lambda:rt.guard(True),event('live'));v=L.rl(shadow,arrays,data,p,lambda:rt.guard(True),event('shadow'));assert u==v and digest(L.state(obj))==digest(L.state(shadow))
   proposal=L.export(obj,p,folder/'proposal.pt.gz');assert L.frozen(p,proposal['path']).phash()==obj[0].phash();save(folder/'LEARNING_COMPLETE.json',dict(epoch=time.time(),learning_stop=p['absolute_learning_stop']));vv=rt.request(p['evaluation_jobs'],proposal,'PURE_STUDENT_ALL32');rows=[r for a,r in vv];keys=[i for i,j in enumerate(p['evaluation_jobs']) if j['id'] in p['key_replay_ids']];rt.request([p['evaluation_jobs'][i] for i in keys],proposal,'EXACT_KEY2',reference=[rows[i] for i in keys]);check=assess(p,rows)
   if not check['accepted']:
    for o in [obj,shadow]:L.load(o,before)
   result=dict(kind='round',BC=b,update=u,behavior=behavior,proposal=proposal,check=check,accepted=check['accepted'],model=proposal if check['accepted'] else p['initial'],retained_joint_hash=digest(L.state(obj)),new_RL=0 if u is None else u['actual_onpolicy_Adam'],globally_promoted=False)
  elif p['kind']=='final':
   vv=rt.request(p['final_jobs'],p['initial'],'FINAL_SAME_POLICY_407');rows=[r for a,r in vv];result=dict(kind='final',model=p['initial'],old275=G.R.M.tally(rows[:275]),six54=G.R.M.tally(rows[275:329]),eight54=G.R.M.tally(rows[329:383]),field24=G.R.M.tally(rows[383:]),full407_safe=all(r['success'] and not r['collision'] for r in rows),fresh16=None,reload78=None)
  elif p['kind']=='stress':
   # Pre-registered paired diagnostic, frozen models, no learning/model choice.
   rt.role='PAIRED_PERTURBATION_DIAGNOSTIC';compact=[]
   for idx in range(len(p['stress_jobs'])):
    rt.guard();job=p['stress_jobs'][idx];assert sum(rt.counts)+2*job['case']['step_cap']<=p['max_physics'] and rt.commands+4<=p['max_commands'];rt.commands+=2;vals=rt.pool.request([dict(model=m['path'],job=job,seed=None,query=False,full_path=None) for m in p['stress_models']])
    for mi,(a,r) in enumerate(vals):
     assert r['policy_hash']==p['stress_models'][mi]['policy_hash'];assert int(a['native_counts'].sum())==r['physics'];row=dict(pair=idx,model=mi,job_id=job['id'],policy_hash=r['policy_hash'],success=bool(r['success'] and not r['collision'] and not (np.abs(np.r_[a['poses'][:,:2],a['next_poses'][:,:2]])>p['field_limit_m']).any()),native_success=bool(r['success']),boundary_crossed=bool((np.abs(np.r_[a['poses'][:,:2],a['next_poses'][:,:2]])>p['field_limit_m']).any()),collision=bool(r['collision']),timeout=bool(r['timeout']),censored=bool(r.get('administrative_censor',False)),remaining_m=r['remaining_m'],physics=r['physics'],array_digest=r['array_digest'],integration_sha256=r['integration_sha256']);assert len(__import__('json').dumps(row))<1024;Base.append('stress_results.jsonl',row);compact.append(row)
   result=dict(kind='stress',pairs=len(p['stress_jobs']),models=p['stress_models'],new_Adam=0,new_RL=0,summary_only_raw_array_digests_no_saved_full_arrays=True)
  else:raise AssertionError(p['kind'])
  state='COMPLETE'
 except BaseException:
  error=traceback.format_exc()
  if before is not None:
   for o in [obj,shadow]:L.load(o,before)
  save(folder/'failure.json',dict(error=error,no_restart=True,raw_preserved=True))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=2
  save(folder/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
 assert state=='COMPLETE',error

def audit(child):
 p=plan(child);folder=Path(p['out']);setup(folder);verify(p,True);G.stopped(folder);assert not (folder/'audit.json').exists();s=read(folder/'summary.json');assert s['state']=='COMPLETE';rows=G.rows(folder);reps=G.rows(folder,'replays.jsonl');by={r['path']:r for r in rows};r=s['result'];hist=0
 for x in rows:
  a=E.N.S.arrays(x['path']);assert sha(x['path'])==x['sha256'] and int(a['native_counts'].sum())==x['physics'];assert all(digest(v)==x['array_hashes'][k] for k,v in a.items())
  if x['full']:
   if x['teacher']:recovery_audit(p,x,a)
   else:F.M.audit_full_history(p,x,a)
 for x in reps:assert x['exact'] and x['array_hashes']==by[x['reference_path']]['array_hashes'] and x['integration_sha256']==by[x['reference_path']]['integration_sha256']
 if p['kind']=='round':
  obj=L.make(p);before=L.state(obj);data=L.dataset(p);b=L.bc(obj,data,p,lambda:None,lambda *_:None);assert b==r['BC'] and digest(L.state(obj))==r['behavior']['joint_hash'];hist+=len(b)
  if p['arm']=='rl':
   u=L.rl(obj,[E.N.S.arrays(x['path']) for x in rows[:4]],data,p,lambda:None,lambda *_:None);assert u==r['update'];hist+=u['actual_onpolicy_Adam']
  assert digest(L.state(obj))==r['proposal']['joint_hash'] and sha(r['proposal']['path'])==r['proposal']['sha256'];assert assess(p,rows[4:] if p['arm']=='rl' else rows)==r['check']
  if not r['accepted']:L.load(obj,before)
  assert digest(L.state(obj))==r['retained_joint_hash'];assert read(folder/'LEARNING_COMPLETE.json')['epoch']<p['absolute_learning_stop']
 if p['kind']=='recovery':
  assert len(rows)==len(reps)==2;row=rows[1];z=E.N.S.arrays(row['path']);w=read(folder/'prefix_witness.json');k=w['teacher_first'];assert all(digest(z[key][:k])==h for key,h in w['prefix_hashes'].items()) and digest(z['initial_native_integration'])==w['initial_integration'] and w['source_failure_array_digest']==rows[0]['array_digest'];assert r['teacher_qualified']==bool(row['success'] and not row['collision'] and row['field_detail']['field_qualified_success'])
 if p['kind']=='engineering':
  assert len(rows)==len(reps)==6;sys.path.insert(0,str(ROOT/'stage/experiments/t519_fresh_full_actor_rl'));import t519_learning as Credit
  _,density=Credit.peer_batch((L.frozen(p,p['initial']['path']),None),[E.N.S.arrays(x['path']) for x in rows[2:]],p);assert density==r['actual_native_stochastic_density']
 if p['kind']=='stress':
  sr=G.rows(folder,'stress_results.jsonl');assert len(sr)==2*len(p['stress_jobs']);physics=sum(x['physics'] for x in sr)
 else:physics=sum(x['physics'] for x in rows+reps)
 assert physics==s['physics']<=p['max_physics'];assert s['commands']<=p['max_commands'];save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,new_RL=0,discarded_historical_Adam=hist,all_saved_array_hashes_and_native_integrations_rechecked=True,full_saved_training_history_rebuilt=True,not_global_stability_certificate=True,physics=s['physics'],commands=s['commands']));print('T577_AUDIT_PASS',child,flush=True)
if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])
