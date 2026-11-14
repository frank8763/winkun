"""One feedback-state repair of a frozen12/24 research candidate.

Rebuild its actual19 successful trajectories and failed oldleft in native
MuJoCo. Only a genuinely successful neural teacher continuation supplies
conditional-mean labels; final evaluation has no teacher at runtime.
"""
from pathlib import Path
import copy,fcntl,json,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t543_feedback_joint_repair';DOC=ROOT/'stage/docs/day_20261114'
sys.path.insert(0,str(HERE));import t543_learning as L
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

class RecoverySession(F.FieldSession):
 def __init__(self,job,*args,**kwargs):
  super().__init__(job,*args,**kwargs);self.expert=L.A.frozen(self.q.p,self.q.p['neural_teacher']['path']);self.expert.sample=False;self.expert_hidden=None;self.teacher_actions=[];self.teacher_locations=[]
 def step(self,forced=None,owner=0,actor_active=True):
  assert forced is None and owner==0;state=self.packet().copy();native_before=E.N.K.R.integration(self.env).copy()
  with torch.no_grad():command,loc,self.expert_hidden,_=self.expert.step(torch.tensor(state[None]),self.expert_hidden)
  assert np.array_equal(native_before,E.N.K.R.integration(self.env));self.teacher_actions.append(command[0].numpy().copy());self.teacher_locations.append(loc[0].numpy().copy());use=len(self.rows)>=self.q.p['teacher_first']
  return super().step(command[0].numpy() if use else None,owner=int(use),actor_active=True)

def payload(s,tape=None):
 a=BASE_PAYLOAD(s,tape);n=len(a['states']);a['teacher_actions']=np.asarray(s.teacher_actions[:n],np.float32);a['teacher_locations']=np.asarray(s.teacher_locations[:n],np.float64);return a

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
  if isinstance(value,tuple):value[1].update(teacher=True,teacher_first=self.p['teacher_first'],teacher_model=self.p['neural_teacher'],training_mixture_NOT_autonomous=True)
  return value

def prefix_rule(p,a):
 b=E.N.S.arrays(p['old_success_row']['path']);n=min(len(a['poses']),len(b['poses']));xy=np.linalg.norm(a['poses'][:n,:2]-b['poses'][:n,:2],axis=1);dy=a['poses'][:n,2]-b['poses'][:n,2];yaw=np.abs(np.arctan2(np.sin(dy),np.cos(dy)))*180/np.pi
 crossing=np.flatnonzero((xy>=p['divergence_m'])|(yaw>=p['divergence_deg']));assert len(crossing)>0
 first=max(1,int(crossing[0])-1);assert first<len(a['states']);return first,dict(first_crossing_hold=int(crossing[0]),teacher_first=first,prefix_seconds=float(a['native_counts'][:first].sum()*.002),crossing_XY_m=float(xy[crossing[0]]),crossing_yaw_deg=float(yaw[crossing[0]]),rule='FIRST_3CM_OR3DEG_PREVIOUS_HOLD_NO_WINDOW_GRID')

def recovery_audit(p,row,a,baseline):
 k=p['teacher_first']
 for name in ['states','next_states','actions','native_counts','poses','next_poses','hidden_before','history_poses','history_goals','history_times','locations']:assert np.array_equal(a[name][:k],baseline[name][:k]),name
 for name in ['states','poses','hidden_before','history_poses','history_goals','history_times']:assert np.array_equal(a[name][k],baseline[name][k]),name
 owner=np.zeros(len(a['states']),np.uint8);owner[k:]=1;assert np.array_equal(owner,a['ownership']) and not a['stochastic'].any();assert np.array_equal(a['initial_native_integration'],baseline['initial_native_integration'])
 teacher=L.A.frozen(p,p['neural_teacher']['path']);teacher.sample=False;h=None
 with torch.no_grad():
  for i,x in enumerate(a['states']):
   u,loc,h,_=teacher.step(torch.tensor(x[None]),h);assert np.array_equal(u[0].numpy(),a['teacher_actions'][i]) and np.array_equal(loc[0].numpy(),a['teacher_locations'][i])
 F.M.audit_full_history(p,row,a)

def assess(p,rows):
 c=Base.assess(p,rows);ids=set(c['field_success_ids']);before=set(p['initial_check']['field_success_ids']);c.update(lost_prior_field_ids=sorted(before-ids),gained_vs_T507=sorted(ids-set(p['accepted_baseline_field_ids'])),panel_repaired=c['old8']['success']==8 and before<=ids)
 c['accepted']=bool(c['panel_repaired'] and c['field24']['contact']<=p['max_accepted_field_contacts'] and c['field24']['invalid']==c['field24']['censored']==c['old8']['invalid']==c['old8']['censored']==0)
 return c

def main():
 root=read(OUT/'protocol.json');p=configuration(root);setup();verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=Base.Runtime(p);state='FAILED';error=None;result=None;obj=shadow=before=None
 try:
  rt.pool=F.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
  jobs=p['source_jobs'];vv=rt.request(jobs,p['initial'],'REBUILD_CANDIDATE19_SUCCESSES_AND_OLD_FAILURE',full=True);sr=[r for a,r in vv]
  source={r['job_id']:r for r in G.rows(Path(p['source_cases']).parent) if r['role']=='PURE_STUDENT_FULL32'}
  for a,r in vv:assert r['array_hashes']==source[r['job_id']]['array_hashes'] and r['integration_sha256']==source[r['job_id']]['integration_sha256']
  rt.request(jobs,p['initial'],'EXACT_REBUILD19_AND_FAILURE',full=True,reference=sr);successful=[r for r in sr if r['success'] and not r['collision']];assert len(successful)==19
  oldrow=next(r for r in sr if r['job_id']==p['probe_job']['id']);baseline=E.N.S.arrays(oldrow['path']);first,derived=prefix_rule(p,baseline);p['teacher_first']=first;save(OUT/'derived_prefix.json',derived)
  rt.pool.close();rt.pool=None;rt.commands+=2;rt.pool=RecoveryPool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard);rt.commands+=2
  vr=rt.request([p['probe_job']],p['initial'],'ACTUAL_OLDLEFT_FEEDBACK_TEACHER',full=True);a,row=vr[0];recovery_audit(p,row,a,baseline);rt.request([p['probe_job']],p['initial'],'EXACT_FEEDBACK_TEACHER',full=True,reference=[row]);eligible=bool(row['success'] and not row['collision'])
  result=dict(source_model=p['initial'],rebuilt_successes=19,derived_prefix=derived,teacher_qualified=eligible,teacher_result=dict(success=row['success'],contact=row['collision'],timeout=row['timeout'],remaining_m=row['remaining_m']),Actor_Adam=0,Value_updates=0,new_policy_gradient_updates=0,update=None,model=None,check=None,accepted=False,old275=None,six54=None,eight54=None,fresh16=None,reload78=None)
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
  result['reason']='JOINT_REPAIR_RESEARCH_GATE_PASSED' if result['accepted'] else 'REPAIR_REJECTED_ROLLBACK' if eligible else 'NEURAL_FEEDBACK_NOT_QUALIFIED_ZERO_LEARNING';state='COMPLETE'
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
 rows=G.rows(OUT);rep=G.rows(OUT,'replays.jsonl');source=rows[:20];baseline=E.N.S.arrays(next(r['path'] for r in source if r['job_id']==p['probe_job']['id']));first,derived=prefix_rule(p,baseline);assert derived==read(OUT/'derived_prefix.json');p['teacher_first']=first
 for r in rows:
  assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert all(digest(v)==r['array_hashes'][k] for k,v in a.items());assert int(a['native_counts'].sum())==r['physics']
  if r['full']:
   if r['teacher']:recovery_audit(p,r,a,baseline)
   else:F.M.audit_full_history(p,r,a)
 by={r['path']:r for r in rows}
 for r in rep:assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256']
 result=s['result'];teacher=rows[20];assert result['teacher_qualified']==bool(teacher['success'] and not teacher['collision']);steps=virtual=0
 if result['teacher_qualified']:
  obj=L.make(p);before=L.state(obj);u=L.fit(obj,E.N.S.arrays(teacher['path']),p,L.references([r for r in source if r['success']]),lambda:None,lambda *_:None);steps=u['Actor_Adam'];virtual=u['virtual_discarded_Adam'];assert u==result['update'] and digest(L.state(obj))==result['model']['joint_hash'] and sha(result['model']['path'])==result['model']['sha256'];check=assess(p,rows[21:]);assert check==result['check'] and len(rows)==53
  if not result['accepted']:L.load(obj,before)
  assert digest(L.state(obj))==result['retained_joint_hash']
 else:assert len(rows)==21 and not G.rows(OUT,'optimizer_calls.jsonl')
 expected=[[who,stage,k] for who in ['live','shadow'] for k in range(steps) for stage in ['VIRTUAL_BEFORE','VIRTUAL_ROLLED_BACK','BEFORE','AFTER']];assert G.rows(OUT,'optimizer_calls.jsonl')==expected
 assert len(rows)==len(rep);assert s['physics']==sum(r['physics'] for r in rows+rep)<=p['max_physics'];assert s['commands']==len(rows)+len(rep)+(12 if steps else 8)<=p['max_commands']
 save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,discarded_historical_Actor_Adam=steps,discarded_virtual_Adam=virtual,teacher_actual_history_locations_and_intervention_rebuilt=True,all_saved_full_training_density_rewards_hidden_rebuilt=True,Actor_Adam_and_virtual_MIR_ranking_rollback_exact=True,compact_evaluation_full120_NOT_reconstructed_offline=True,all_native_replay_full_hashes_and_integration_exact=True,physics=s['physics'],commands=s['commands']));print('T543_FULL_REPAIR_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
