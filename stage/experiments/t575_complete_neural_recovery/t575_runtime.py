"""Own-state replay extraction and bounded whole-set native policy acceptance."""
from pathlib import Path
import sys,os,time,fcntl,traceback,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t575_complete_neural_recovery';DOC=ROOT/'stage/docs/day_20261115/t575'
sys.path.insert(0,str(HERE));import t575_learning as L
sys.path.insert(0,str(ROOT/'stage/experiments/t569_frozen_complete_panel'));import t569_runtime as Old
Base,F,G,E,P,np,torch=Old.Base,Old.F,Old.G,Old.E,Old.P,L.np,L.torch
read,save,sha,digest=L.read,L.save,L.sha,L.digest

def used(p):return sum(f.stat().st_size for f in Path(p).rglob('*') if f.is_file() and not f.is_symlink())
def verify(p,full=False):
 root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json');assert time.time()<root['absolute_all_stop'] and used(OUT)<=root['max_bytes'] and used(ROOT/'stage/runs')-used(OUT)+root['max_bytes']+131072<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized() and set(os.sched_getaffinity(0))<={14,15}
 for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f

def configuration():
 p=read(OUT/'protocol.json');cfg=read(p['actor_configuration']);cfg.update(p);cfg['out']=str(OUT);return cfg

def setup():
 Old.OUT=OUT;Old.HERE=HERE;Old.verify=verify;Old.setup();Base.ACTIVE=OUT;P.frozen=L.frozen;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

def worker(pipe,index,p,counts):
 G.T.Session=F.FieldSession;G.B.native_setup=F.field_setup;P.frozen=L.frozen;E.payload=Old.Old.Prior.NORMAL;G.cpu_worker(pipe,index,p,counts)
class Pool(F.Pool):
 def __init__(self,p,counts,guard):
  self.p=p;self.guard,self.children,self.pipes=guard,[],[];ctx=G.R.M.mp.get_context('spawn')
  try:
   for i in range(2):
    a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
   self.ready=[self.receive(i) for i in range(len(self.children))];assert all(v['state']=='READY' for v in self.ready);save(OUT/'worker_generations.json',(read(OUT/'worker_generations.json') if (OUT/'worker_generations.json').exists() else [])+[dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time(),CPU=[14,15],CPUQuota='200%',GPU=False)])
  except BaseException:self.close();raise

def assess(p,rows):
 assert len(rows)==32 and [r['job_id'] for r in rows]==[j['id'] for j in p['evaluation_jobs']];ids={r['job_id'] for r in rows if r['success'] and not r['collision'] and (not r['job_id'].startswith('t502_') or r['field_detail']['field_qualified_success'])};current=set(p['protected_success_ids']);reference=set(p['reference23_success_ids']);targets=set(p['restoration_target_ids']);lost=current-ids
 return dict(old8=G.R.M.tally(rows[:8]),field24=G.R.M.tally(rows[8:]),success_ids=sorted(ids),lost_current21=sorted(lost),restored_targets=sorted(targets&ids),missing_reference23=sorted(reference-ids),accepted=not lost and reference<=ids,complete32_safe=len(ids)==32,not_full407_or_unseen_qualification=True)
def event(w):return lambda s,k:Base.append('optimizer_calls.jsonl',[w,s,k])
def main():
 p=configuration();setup();verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=Base.Runtime(p);state='FAILED';error=None;result=None;obj=shadow=before=None;memory=[]
 try:
  rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard);old={r['job_id']:r for r in p['baseline_rows']}
  for off in range(0,len(p['retention_jobs']),2):
   jobs=p['retention_jobs'][off:off+2]
   for a,r in rt.request(jobs,p['initial'],'EXACT_REGEN_CURRENT21_OWN_STATES'):
    reference=old[r['job_id']];assert r['array_hashes']==reference['array_hashes'] and r['integration_sha256']==reference['integration_sha256'] and r['field_detail']==reference['field_detail'];assert r['success'] and not r['collision'];F.M.audit_full_history(p,r,a)
    subset={k:a[k] for k in ['states','locations','actions']};path=OUT/'memory'/('own_'+str(len(memory))+'.npz');path.parent.mkdir(exist_ok=True);E.N.write_npz(path,subset,p['memory_file_bytes']);entry=dict(job_id=r['job_id'],path=str(path),sha256=sha(path),array_hashes={k:r['array_hashes'][k] for k in subset},original_case_path=r['path'],original_case_sha256=r['sha256'],exact_prior_model_hashes=True);memory.append(entry);Base.append('memory_rows.jsonl',entry)
  assert len(memory)==22;rt.pool.close();rt.pool=None;rt.commands+=2
  rt.role='COMPLETE_NEURAL_RECOVERY_FIXED_RIDGE';rt.guard(True);obj=L.make(p);shadow=L.make(p);before=L.state(obj);data=L.dataset(p,memory);w,u=L.solve(data,p);w2,v=L.solve(data,p);assert u==v and torch.equal(w,w2);rt.guard(True)
  for tag,o,coef in [('live',obj,w),('shadow',shadow,w2)]:Base.append('parameter_applications.jsonl',[tag,'BEFORE',0]);L.apply(o,coef);Base.append('parameter_applications.jsonl',[tag,'AFTER',0])
  rt.guard(True);assert digest(L.state(obj))==digest(L.state(shadow));save(OUT/'solver.json',u);E.N.write_npz(OUT/'coefficients.npz',dict(coefficients=w.numpy()),65536);model=L.export(obj,p,OUT/'proposal.pt.gz');assert L.frozen(p,model['path']).phash()==obj[0].phash();save(OUT/'LEARNING_COMPLETE.json',dict(epoch=time.time(),learning_stop=p['absolute_learning_stop'],model=model))
  rt.commands+=2;rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard);vv=rt.request(p['evaluation_jobs'],model,'PURE_STUDENT_COMPLETE32');rows=[r for a,r in vv];keys=[i for i,j in enumerate(p['evaluation_jobs']) if j['id'] in p['key_replay_ids']];assert len(keys)==2;rt.request([p['evaluation_jobs'][i] for i in keys],model,'EXACT_KEY2',reference=[rows[i] for i in keys]);check=assess(p,rows)
  if not check['accepted']:
   for z in [obj,shadow]:L.load(z,before)
  result=dict(fit=u,proposal=model,check=check,accepted=check['accepted'],retained_research_model=model if check['accepted'] else p['initial'],retained_joint_hash=digest(L.state(obj)),Actor_BC_Adam=0,shadow_BC_Adam=0,linear_solves_live=1,linear_solves_shadow=1,new_RL=0,new_Critic=0,own22_physical_history_repeats_not_new_independent_samples=True,teacher_runtime_at_evaluation=False,old275=None,six54=None,eight54=None,fresh16=None,reload78=None,globally_promoted=False);state='COMPLETE'
 except BaseException:
  error=traceback.format_exc()
  if before is not None:
   for z in [obj,shadow]:L.load(z,before)
  save(OUT/'failure.json',dict(error=error,no_restart=True,all_raw_preserved=True,whole_ActorAdam_rollback=before is not None))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=2
  save(OUT/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
 assert state=='COMPLETE',error

def audit():
 p=configuration();setup();verify(p,True);G.stopped(OUT);assert not (OUT/'audit.json').exists();s=read(OUT/'summary.json');assert s['state']=='COMPLETE';rows=G.rows(OUT);reps=G.rows(OUT,'replays.jsonl');memory=G.rows(OUT,'memory_rows.jsonl');assert len(memory)==22 and len(rows)==54 and len(reps)==2;by={r['path']:r for r in rows}
 for r in rows:
  a=E.N.S.arrays(r['path']);assert sha(r['path'])==r['sha256'] and int(a['native_counts'].sum())==r['physics'];assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
 for r in memory:
  a=E.N.S.arrays(r['path']);assert sha(r['path'])==r['sha256'] and all(digest(v)==r['array_hashes'][k] for k,v in a.items());raw=by[r['original_case_path']];assert all(h==raw['array_hashes'][k] for k,h in r['array_hashes'].items());assert r['job_id'] in p['protected_success_ids']
 for r in reps:
  x=by[r['reference_path']];assert r['exact'] and r['array_hashes']==x['array_hashes'] and r['integration_sha256']==x['integration_sha256']
 obj=L.make(p);before=L.state(obj);w,u=L.solve(L.dataset(p,memory),p);assert u==read(OUT/'solver.json') and torch.equal(w,torch.tensor(E.N.S.arrays(OUT/'coefficients.npz')['coefficients']));L.apply(obj,w);r=s['result'];assert digest(L.state(obj))==r['proposal']['joint_hash'] and sha(r['proposal']['path'])==r['proposal']['sha256'];assert assess(p,rows[22:])==r['check']
 if not r['accepted']:L.load(obj,before)
 assert digest(L.state(obj))==r['retained_joint_hash'];assert s['physics']==sum(x['physics'] for x in rows+reps)<=p['max_physics'];assert s['commands']==64;assert read(OUT/'LEARNING_COMPLETE.json')['epoch']<p['absolute_learning_stop']
 save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,new_Adam=0,new_RL=0,independent_discarded_linear_solve=1,exact_key_native_replays=2,full32_independent_native_replay=False,all_memory_array_hashes_and_solver_model_rollback_exact=True,physics=s['physics'],commands=s['commands']));print('T575_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
