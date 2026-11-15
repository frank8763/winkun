"""Freeze A candidate; record its sole lost success and qualify one recovery."""
from pathlib import Path
import sys,os,time,traceback,fcntl,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t573_actual_failure_recovery';DOC=ROOT/'stage/docs/day_20261115/t573'
sys.path.insert(0,str(ROOT/'stage/experiments/t571_self_replay_continual'));import t571_learning as L
sys.path.insert(0,str(ROOT/'stage/experiments/t568_student_recovery_qualification'));import t568_runtime as Old
Base,F,G,E,P,np,torch=Old.Base,Old.F,Old.G,Old.E,Old.P,L.np,L.torch
Teacher=Old.Teacher
read,save,sha,digest=L.read,L.save,L.sha,L.digest

def used(p):return sum(f.stat().st_size for f in Path(p).rglob('*') if f.is_file() and not f.is_symlink())
def verify(p,full=False):
 r=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json');assert time.time()<r['absolute_all_stop'] and used(OUT)<=r['max_bytes'] and used(ROOT/'stage/runs')-used(OUT)+r['max_bytes']+131072<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized() and set(os.sched_getaffinity(0))<={14,15}
 for f,h in (r['inputs'] if full else r['protected_files']).items():assert sha(f)==h,f

def configuration():
 r=read(OUT/'protocol.json');p=read(r['actor_configuration']);p.update(r);p['out']=str(OUT);return p

def setup():
 Old.OUT=OUT;Old.HERE=HERE;Old.verify=verify;Old.setup();Base.ACTIVE=OUT;P.frozen=L.frozen;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

def worker(pipe,index,p,counts):
 G.T.Session=Teacher.RecoverySession if p['teacher_enabled'] else F.FieldSession;G.B.native_setup=F.field_setup;P.frozen=L.frozen;E.payload=Teacher.payload if p['teacher_enabled'] else Old.Old.Prior.NORMAL;G.cpu_worker(pipe,index,p,counts)
class Pool(F.Pool):
 def __init__(self,p,counts,guard):
  self.p=p;self.guard,self.children,self.pipes=guard,[],[];ctx=G.R.M.mp.get_context('spawn')
  try:
   a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,0,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a);self.ready=[self.receive(0)];assert self.ready[0]['state']=='READY';f=OUT/'worker_generations.json';save(f,(read(f) if f.exists() else [])+[dict(parent=os.getpid(),workers=[c.pid],epoch=time.time(),CPU=[14],GPU=False,training_only_teacher=p['teacher_enabled'])])
  except BaseException:self.close();raise
 def receive(self,i):
  v=super().receive(i)
  if isinstance(v,tuple) and self.p['teacher_enabled']:v[1].update(teacher=True,teacher_first=self.p['teacher_first'],teacher_contract='TRAINING_ONLY_VERIFIED_REACTIVE_RECOVERY',training_mixture_NOT_autonomous=True)
  return v

def prefix(p,a,reference):
 assert np.array_equal(a['initial_native_integration'],reference['initial_native_integration']) and np.array_equal(a['task_goal_xy'],reference['task_goal_xy']);n=min(len(a['poses']),len(reference['poses']));assert np.array_equal(a['native_counts'][:n-1],reference['native_counts'][:n-1]);delta=a['poses'][:n]-reference['poses'][:n];pos=np.linalg.norm(delta[:,:2],axis=1);ang=np.abs(np.arctan2(np.sin(delta[:,2]),np.cos(delta[:,2])));hits=np.flatnonzero((pos>=.03)|(ang>=np.deg2rad(3.)))
 if not len(hits):return None,dict(rule='ONE_STEP_BEFORE_FIRST_3CM_OR_3DEG_FROM_OLD_SUCCESS',available=False,no_grid=True)
 k=max(0,int(hits[0])-1);return k,dict(rule='ONE_STEP_BEFORE_FIRST_3CM_OR_3DEG_FROM_OLD_SUCCESS',available=True,teacher_first=k,prefix_seconds=float(a['native_counts'][:k].sum()*.002),first_divergence_decision=int(hits[0]),position_difference_at_intervention_m=float(pos[k]),heading_difference_at_intervention_deg=float(np.rad2deg(ang[k])),no_grid=True)

def main():
 p=configuration();setup();verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=Base.Runtime(p);state='FAILED';error=None;result=None
 try:
  rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline,teacher_enabled=False),rt.counts,rt.guard);v=rt.request([p['job']],p['model'],'FROZEN_A_SOLE_LOST_SUCCESS_FULL',full=True);a,row=v[0];assert row['array_hashes']==p['baseline_compact_row']['array_hashes'] and row['integration_sha256']==p['baseline_compact_row']['integration_sha256'];F.M.audit_full_history(p,row,a);rt.request([p['job']],p['model'],'EXACT_A_FAILURE_FULL',full=True,reference=[row]);reference=E.N.S.arrays(p['old_success_row']['path']);k,derived=prefix(p,a,reference);save(OUT/'derived_prefix.json',derived);diag=F.detail(p,row,a);save(OUT/'diagnosis.json',diag);rt.pool.close();rt.pool=None;rt.commands+=2
  result=dict(model=p['model'],pure_student_failure=diag,derived_prefix=derived,teacher_qualified=False,teacher_result=None,new_optimizer=0,new_RL=0,new_student_model=None,new_student_test=None,globally_promoted=False,scope='Frozen failure reconstruction and one teacher recovery qualification, not learning or autonomous improvement')
  if k is not None:
   p['teacher_first']=k;rt.commands+=2;rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline,teacher_enabled=True),rt.counts,rt.guard);vv=rt.request([p['job']],p['model'],'ACTUAL_A_PREFIX_THEN_TRAINING_ONLY_RECOVERY',full=True);z,r=vv[0];Teacher.recovery_audit(p,r,z,a);rt.request([p['job']],p['model'],'EXACT_ACTUAL_A_RECOVERY',full=True,reference=[r]);qualified=bool(r['success'] and not r['collision']);result.update(teacher_qualified=qualified,teacher_result=dict(success=r['success'],contact=r['collision'],timeout=r['timeout'],remaining_m=r['remaining_m'],seconds=r['physics']*.002))
  state='COMPLETE'
 except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,no_restart=True,raw_preserved=True))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=2
  save(OUT/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
 assert state=='COMPLETE',error

def audit():
 p=configuration();setup();verify(p,True);G.stopped(OUT);assert not (OUT/'audit.json').exists();s=read(OUT/'summary.json');assert s['state']=='COMPLETE';rows=G.rows(OUT);reps=G.rows(OUT,'replays.jsonl');assert len(rows)==len(reps);a=E.N.S.arrays(rows[0]['path']);reference=E.N.S.arrays(p['old_success_row']['path']);k,derived=prefix(p,a,reference);assert derived==read(OUT/'derived_prefix.json')==s['result']['derived_prefix'];assert rows[0]['array_hashes']==p['baseline_compact_row']['array_hashes'];F.M.audit_full_history(p,rows[0],a);assert F.detail(p,rows[0],a)==read(OUT/'diagnosis.json')
 for r in rows:
  z=E.N.S.arrays(r['path']);assert sha(r['path'])==r['sha256'] and all(digest(v)==r['array_hashes'][key] for key,v in z.items());assert int(z['native_counts'].sum())==r['physics']
 if k is not None:
  assert len(rows)==2;p['teacher_first']=k;z=E.N.S.arrays(rows[1]['path']);Teacher.recovery_audit(p,rows[1],z,a);assert s['result']['teacher_qualified']==bool(rows[1]['success'] and not rows[1]['collision'])
 else:assert len(rows)==1
 by={r['path']:r for r in rows}
 for r in reps:assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256']
 assert s['physics']==sum(r['physics'] for r in rows+reps)<=p['max_physics'];assert s['commands']==len(rows)+len(reps)+(8 if k is not None else 4)<=p['max_commands'];save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,new_optimizer=0,new_RL=0,actual_full_history_density_reward_hidden_rebuilt=True,prefix_and_boundary_exact=True,teacher_native_commands_and_recovery_exact=True,physics=s['physics'],commands=s['commands'],teacher_qualified=s['result']['teacher_qualified'],not_student_improvement=True));print('T573_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
