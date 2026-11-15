"""Training-only neural teacher, independent true carry on actual A prefix."""
from pathlib import Path
import sys,os,time,shutil,traceback,fcntl
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t574_neural_recovery_handoff';DOC=ROOT/'stage/docs/day_20261115/t574'
sys.path.insert(0,str(ROOT/'stage/experiments/t573_actual_failure_recovery'));import t573_runtime as Old
L,Base,F,G,E,P,np,torch=Old.L,Old.Base,Old.F,Old.G,Old.E,Old.P,Old.np,Old.torch
read,save,sha,digest=L.read,L.save,L.sha,L.digest
BASE_PAYLOAD=Old.Teacher.BASE_PAYLOAD

def used(p):return sum(f.stat().st_size for f in Path(p).rglob('*') if f.is_file() and not f.is_symlink())
def verify(p,full=False):
 r=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json');assert time.time()<r['absolute_all_stop'] and used(OUT)<=r['max_bytes'] and used(ROOT/'stage/runs')-used(OUT)+r['max_bytes']+131072<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized() and set(os.sched_getaffinity(0))<={14,15}
 for f,h in (r['inputs'] if full else r['protected_files']).items():assert sha(f)==h,f

def configuration():
 p=read(OUT/'protocol.json');cfg=read(p['actor_configuration']);cfg.update(p);cfg['out']=str(OUT);return cfg

def setup():
 Old.OUT=OUT;Old.HERE=HERE;Old.verify=verify;Old.setup();Base.ACTIVE=OUT;P.frozen=L.frozen;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

class RecoverySession(F.FieldSession):
 def __init__(self,job,*args,**kwargs):
  super().__init__(job,*args,**kwargs);p=self.q.p;assert sha(p['neural_teacher']['path'])==p['neural_teacher']['sha256'];self.teacher=L.frozen(p,p['neural_teacher']['path']);assert self.teacher.phash()==p['neural_teacher']['policy_hash'];self.teacher.sample=False;self.teacher.reset_noise();self.teacher_hidden=None;self.teacher_labels=[];self.teacher_boundary=np.zeros((1,1,64),np.float64);self.holds=int(job['teacher_holds'])
 def step(self,forced=None,owner=0,actor_active=True):
  assert forced is None and owner==0;k=len(self.rows);before=E.N.K.R.integration(self.env).copy();x=self.packet().copy()
  if k==self.q.p['teacher_first']:self.teacher_boundary=np.zeros((1,1,64),np.float64) if self.teacher_hidden is None else self.teacher_hidden.detach().numpy().copy()
  with torch.no_grad():u,_,self.teacher_hidden,_=self.teacher.step(torch.tensor(x[None]),self.teacher_hidden)
  assert np.array_equal(before,E.N.K.R.integration(self.env));label=u[0].numpy().copy();self.teacher_labels.append(label);first=self.q.p['teacher_first'];use=k>=first and (self.holds<0 or k<first+self.holds)
  return super().step(label if use else None,owner=int(use),actor_active=True)

def payload(s,tape=None):
 a=BASE_PAYLOAD(s,tape);a.update(teacher_actions=np.asarray(s.teacher_labels[:len(a['states'])],np.float32),teacher_holds=np.asarray(s.holds,np.int64),teacher_takeover_hidden_before=s.teacher_boundary.copy());return a

def worker(pipe,index,p,counts):
 G.T.Session=RecoverySession;G.B.native_setup=F.field_setup;P.frozen=L.frozen;E.payload=payload;G.cpu_worker(pipe,index,p,counts)
class Pool(F.Pool):
 def __init__(self,p,counts,guard):
  self.p=p;self.guard,self.children,self.pipes=guard,[],[];ctx=G.R.M.mp.get_context('spawn')
  try:
   a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,0,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a);self.ready=[self.receive(0)];assert self.ready[0]['state']=='READY';save(OUT/'worker_generations.json',[dict(parent=os.getpid(),workers=[c.pid],epoch=time.time(),CPU=[14],GPU=False,training_only_neural_teacher=True)])
  except BaseException:self.close();raise
 def receive(self,i):
  v=super().receive(i)
  if isinstance(v,tuple):v[1].update(teacher=True,teacher_first=self.p['teacher_first'],teacher_model=self.p['neural_teacher'],training_mixture_NOT_autonomous=True,teacher_contract='T550_INDEPENDENT_CAUSAL_CARRY_ON_REAL_STUDENT_PREFIX')
  return v

def recovery_audit(p,row,a,baseline):
 k=p['teacher_first'];assert len(a['states'])>k
 for key in ['states','next_states','actions','native_counts','poses','next_poses','hidden_before','history_poses','history_goals','history_times','locations']:assert np.array_equal(a[key][:k],baseline[key][:k]),key
 for key in ['states','poses','hidden_before','history_poses','history_goals','history_times']:assert np.array_equal(a[key][k],baseline[key][k]),key
 assert np.array_equal(a['initial_native_integration'],baseline['initial_native_integration']);owner=np.zeros(len(a['states']),np.uint8);holds=int(a['teacher_holds']);owner[k:None if holds<0 else k+holds]=1;assert np.array_equal(owner,a['ownership']) and not a['stochastic'].any()
 teacher=L.frozen(p,p['neural_teacher']['path']);teacher.sample=False;teacher.reset_noise();th=None;q=L.frozen(p,p['model']['path']);q.sample=False;q.reset_noise();h=None
 with torch.no_grad():
  for i,x in enumerate(a['states']):
   if i==k:assert np.array_equal(a['teacher_takeover_hidden_before'],np.zeros((1,1,64)) if th is None else th.numpy())
   own,loc,h,_=q.step(torch.tensor(x[None]),h);command,_,th,_=teacher.step(torch.tensor(x[None]),th);assert np.array_equal(a['teacher_actions'][i],command[0].numpy());assert np.array_equal(a['actions'][i],command[0].numpy() if owner[i] else own[0].numpy())
 F.M.audit_full_history(p,row,a)
 return F.detail(p,row,a)

def main():
 p=configuration();setup();verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=Base.Runtime(p);state='FAILED';error=None;result=None
 try:
  baseline=E.N.S.arrays(p['baseline_row']['path']);rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard);job=dict(p['job'],teacher_holds=-1);vv=rt.request([job],p['model'],'FULL_NEURAL_TEACHER_RECOVERY_FROM_REAL_A_PREFIX',full=True);a,r=vv[0];detail=recovery_audit(p,r,a,baseline);rt.request([job],p['model'],'EXACT_FULL_NEURAL_TEACHER_RECOVERY',full=True,reference=[r]);qualified=detail['field_qualified_success'];handoff=None
  if qualified:
   short=dict(p['job'],teacher_holds=20);ss=rt.request([short],p['model'],'FIXED_FOUR_SECONDS_CORRECTION_THEN_STUDENT',full=True);b,sr=ss[0];hd=recovery_audit(p,sr,b,baseline);rt.request([short],p['model'],'EXACT_FOUR_SECONDS_THEN_STUDENT',full=True,reference=[sr]);autonomous=max(0,len(b['states'])-p['teacher_first']-20);handoff=dict(detail=hd,autonomous_decisions_after_return=autonomous,safe_goal_after_handoff=bool(hd['field_qualified_success'] and autonomous>0))
  result=dict(model=p['model'],neural_teacher=p['neural_teacher'],teacher_first=p['teacher_first'],prefix_seconds=p['prefix_seconds'],teacher_qualified=qualified,teacher_detail=detail,handoff=handoff,new_optimizer=0,new_RL=0,new_student_model=None,new_student_autonomous_full_task=None,learning_eligible=bool(qualified and handoff and handoff['safe_goal_after_handoff']),globally_promoted=False,scope='Training-only neural recovery/handoff qualification;notstudentlearning/whole-taskautonomy');state='COMPLETE'
 except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,no_restart=True,raw_preserved=True))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=2
  save(OUT/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
 assert state=='COMPLETE',error

def audit():
 p=configuration();setup();verify(p,True);G.stopped(OUT);assert not (OUT/'audit.json').exists();s=read(OUT/'summary.json');assert s['state']=='COMPLETE';rows=G.rows(OUT);reps=G.rows(OUT,'replays.jsonl');baseline=E.N.S.arrays(p['baseline_row']['path']);details=[]
 for r in rows:
  a=E.N.S.arrays(r['path']);assert sha(r['path'])==r['sha256'] and all(digest(v)==r['array_hashes'][k] for k,v in a.items());assert int(a['native_counts'].sum())==r['physics'];details.append(recovery_audit(p,r,a,baseline))
 result=s['result'];assert details[0]==result['teacher_detail'] and result['teacher_qualified']==details[0]['field_qualified_success'];expected=2 if result['teacher_qualified'] else 1;assert len(rows)==len(reps)==expected
 if expected==2:
  a=E.N.S.arrays(rows[1]['path']);n=max(0,len(a['states'])-p['teacher_first']-20);assert result['handoff']==dict(detail=details[1],autonomous_decisions_after_return=n,safe_goal_after_handoff=bool(details[1]['field_qualified_success'] and n>0))
 by={r['path']:r for r in rows}
 for r in reps:assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256']
 assert s['physics']==sum(r['physics'] for r in rows+reps)<=p['max_physics'];assert s['commands']==len(rows)+len(reps)+4<=p['max_commands'];save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,new_optimizer=0,new_RL=0,student_and_neural_teacher_causal_hidden_independently_rebuilt=True,prefix_and_teacher_takeover_hidden_exact=True,actual_executed_owned_actions_verified=True,all_full_history_density_rewards_and_exact_native_replays=True,physics=s['physics'],commands=s['commands'],teacher_qualified=result['teacher_qualified'],learning_eligible=result['learning_eligible'],not_student_improvement=True));print('T574_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
