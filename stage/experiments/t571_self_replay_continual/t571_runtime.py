"""Own-state replay extraction and bounded whole-set native policy acceptance."""
from pathlib import Path
import sys,os,time,fcntl,traceback,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t571_self_replay_continual';DOC=ROOT/'stage/docs/day_20261115/t571'
sys.path.insert(0,str(HERE));import t571_learning as L
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
  assert len(memory)==21;rt.pool.close();rt.pool=None;rt.commands+=2
  rt.role='TASK_BALANCED_OWN_STATE_REHEARSAL_AND_RECOVERY_BC';obj=L.make(p);shadow=L.make(p);before=L.state(obj);data=L.dataset(p,memory);u=L.fit(obj,data,p,lambda:rt.guard(True),event('live'));v=L.fit(shadow,data,p,lambda:rt.guard(True),event('shadow'));assert u==v and digest(L.state(obj))==digest(L.state(shadow));model=L.export(obj,p,OUT/'proposal.pt.gz');assert L.frozen(p,model['path']).phash()==obj[0].phash()
  # A new generation after learning; explicit two additional startup commands.
  rt.commands+=2;rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard);vv=rt.request(p['evaluation_jobs'],model,'PURE_STUDENT_COMPLETE32');rows=[r for a,r in vv];rt.request(p['evaluation_jobs'],model,'EXACT_PURE_STUDENT_COMPLETE32',reference=rows);check=assess(p,rows)
  if not check['accepted']:
   for z in [obj,shadow]:L.load(z,before)
  result=dict(fit=u,proposal=model,check=check,accepted=check['accepted'],retained_research_model=model if check['accepted'] else p['initial'],retained_joint_hash=digest(L.state(obj)),Actor_BC_Adam=u['steps'],shadow_BC_Adam=v['steps'],new_RL=0,new_Critic=0,own21_physical_history_repeats_not_new_independent_samples=True,teacher_runtime_at_evaluation=False,old275=None,six54=None,eight54=None,fresh16=None,reload78=None,globally_promoted=False);state='COMPLETE'
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
 p=configuration();setup();verify(p,True);G.stopped(OUT);assert not (OUT/'audit.json').exists();s=read(OUT/'summary.json');assert s['state']=='COMPLETE';rows=G.rows(OUT);reps=G.rows(OUT,'replays.jsonl');memory=G.rows(OUT,'memory_rows.jsonl');assert len(memory)==21 and len(rows)==53 and len(reps)==32;by={r['path']:r for r in rows}
 for r in rows:
  a=E.N.S.arrays(r['path']);assert sha(r['path'])==r['sha256'] and int(a['native_counts'].sum())==r['physics'];assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
 for r in memory:
  a=E.N.S.arrays(r['path']);assert sha(r['path'])==r['sha256'] and all(digest(v)==r['array_hashes'][k] for k,v in a.items());raw=by[r['original_case_path']];assert all(h==raw['array_hashes'][k] for k,h in r['array_hashes'].items());assert r['job_id'] in p['protected_success_ids']
 for r in reps:
  a=by[r['reference_path']];assert r['exact'] and r['array_hashes']==a['array_hashes'] and r['integration_sha256']==a['integration_sha256']
 obj=L.make(p);before=L.state(obj);data=L.dataset(p,memory);u=L.fit(obj,data,p,lambda:None,lambda *_:None);result=s['result'];assert u==result['fit'] and digest(L.state(obj))==result['proposal']['joint_hash'] and sha(result['proposal']['path'])==result['proposal']['sha256'];assert assess(p,rows[21:])==result['check']
 if not result['accepted']:L.load(obj,before)
 assert digest(L.state(obj))==result['retained_joint_hash'];assert G.rows(OUT,'optimizer_calls.jsonl')==[[who,phase,k] for who in ['live','shadow'] for k in range(p['fit_steps']) for phase in ['BEFORE','AFTER']];assert s['physics']==sum(r['physics'] for r in rows+reps)<=p['max_physics'];assert s['commands']==len(rows)+len(reps)+8<=p['max_commands']
 save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),discarded_historical_Actor_Adam=u['steps'],new_native=0,new_RL=0,all_memory_states_targets_same_prior_native_array_hashes=True,full_student_hidden_recomputed_from_causal120_sequence=True,not_independent_offline_native_history_reconstruction=True,all32_new_policy_exact_native_hashes_and_integrations=True,ActorAdam_and_rollback_rebuilt=True,physics=s['physics'],commands=s['commands']));print('T571_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
