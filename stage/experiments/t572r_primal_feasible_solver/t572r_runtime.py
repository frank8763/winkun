"""Two independently evaluated analytical output-layer policy proposals."""
from pathlib import Path
import sys,os,time,fcntl,traceback,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t572r_primal_feasible_solver';DOC=ROOT/'stage/docs/day_20261115/t572r'
sys.path.insert(0,str(HERE));import t572r_learning as L
sys.path.insert(0,str(ROOT/'stage/experiments/t571_self_replay_continual'));import t571_runtime as Old
Base,F,G,E,P,np,torch=Old.Base,Old.F,Old.G,Old.E,Old.P,L.np,L.torch
read,save,sha,digest=L.read,L.save,L.sha,L.digest

def verify(p,full=False):
 root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json');assert time.time()<root['absolute_all_stop'] and Old.used(OUT)<=root['max_bytes'] and Old.used(ROOT/'stage/runs')-Old.used(OUT)+root['max_bytes']+131072<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized() and set(os.sched_getaffinity(0))<={14,15}
 for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f

def configuration(arm):
 assert arm=='bounded';p=read(OUT/'protocol.json');cfg=read(p['actor_configuration']);cfg.update(p);cfg.update(arm=arm,out=str(OUT/arm),max_physics=p['per_arm_max_physics'],max_commands=68,main_seconds=p['per_arm_main_seconds'],learning_seconds=p['per_arm_learning_seconds'],child_bytes=p['max_bytes']);return cfg

def setup(folder):
 Old.OUT=folder;Old.HERE=HERE;Old.verify=verify;Old.setup();Base.ACTIVE=folder;P.frozen=L.frozen;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

def main(arm):
 p=configuration(arm);folder=Path(p['out']);setup(folder);verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=Base.Runtime(p);rt.commands=0;state='FAILED';error=None;result=None;obj=shadow=before=None
 try:
  data=L.dataset(p);Q,c,A,norm=L.matrices(data,p);rt.role='EXACT_LINEAR_FUNCTIONAL_'+arm;obj=L.make(p);shadow=L.make(p);before=L.state(obj)
  w,lam,u=L.solve(Q,c,A,norm,arm,p,lambda:rt.guard(True));w2,lam2,v=L.solve(Q,c,A,norm,arm,p,lambda:rt.guard(True));assert u==v and torch.equal(w,w2) and torch.equal(lam,lam2);path=folder/'solve.npz';E.N.write_npz(path,dict(coefficients=w.numpy(),dual=lam.numpy()),p['solver_bytes']);save(folder/'solver.json',dict(result=u,sha256=sha(path),before=L.measure(torch.zeros_like(w),data),after=L.measure(w,data),matrix_hashes=dict(Q=digest(Q),c=digest(c),A=digest(A))))
  result=dict(solver=u,model=None,check=None,accepted=False,retained_research_model=p['initial'],new_Adam=0,new_RL=0,exact_linear_parameter_applications=0,old275=None,six54=None,eight54=None,fresh16=None,reload78=None,globally_promoted=False)
  if u['certificate_pass']:
   for tag,o,coef in [('live',obj,w),('shadow',shadow,w2)]:Base.append('parameter_applications.jsonl',[tag,'BEFORE',0]);L.apply(o,coef);Base.append('parameter_applications.jsonl',[tag,'AFTER',0])
   assert digest(L.state(obj))==digest(L.state(shadow));model=L.export(obj,p,folder/'proposal.pt.gz');assert L.frozen(p,model['path']).phash()==obj[0].phash();result.update(model=model,exact_linear_parameter_applications=1,proposal_joint_hash=digest(L.state(obj)))
   rt.commands+=2;rt.pool=Old.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard);vv=rt.request(p['evaluation_jobs'],model,'PURE_STUDENT_COMPLETE32_'+arm);rows=[r for a,r in vv];rt.request(p['evaluation_jobs'],model,'EXACT_PURE_STUDENT_COMPLETE32_'+arm,reference=rows);check=Old.assess(p,rows);result.update(check=check,accepted=check['accepted'],retained_research_model=model if check['accepted'] else p['initial'])
  if not result['accepted']:
   for o in [obj,shadow]:L.load(o,before)
  result['retained_joint_hash']=digest(L.state(obj));state='COMPLETE'
 except BaseException:
  error=traceback.format_exc()
  if before is not None:
   for o in [obj,shadow]:L.load(o,before)
  save(folder/'failure.json',dict(error=error,no_restart=True,all_raw_preserved=True))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=2
  save(folder/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
 assert state=='COMPLETE',error

def audit(arm):
 p=configuration(arm);folder=Path(p['out']);setup(folder);verify(p,True);G.stopped(folder);assert not (folder/'audit.json').exists();s=read(folder/'summary.json');assert s['state']=='COMPLETE';r=s['result'];data=L.dataset(p);Q,c,A,norm=L.matrices(data,p);w,lam,u=L.solve(Q,c,A,norm,arm,p,lambda:None);a=E.N.S.arrays(folder/'solve.npz');assert torch.equal(w,torch.tensor(a['coefficients'])) and torch.equal(lam,torch.tensor(a['dual'])) and u==r['solver'];assert read(folder/'solver.json')['matrix_hashes']==dict(Q=digest(Q),c=digest(c),A=digest(A));obj=L.make(p);before=L.state(obj);rows=G.rows(folder);reps=G.rows(folder,'replays.jsonl')
 if u['certificate_pass']:
  L.apply(obj,w);assert digest(L.state(obj))==r['proposal_joint_hash']==r['model']['joint_hash'];assert sha(r['model']['path'])==r['model']['sha256'];assert Old.assess(p,rows)==r['check'];assert len(rows)==len(reps)==32
  for x in rows:
   z=E.N.S.arrays(x['path']);assert sha(x['path'])==x['sha256'] and int(z['native_counts'].sum())==x['physics'];assert all(digest(v)==x['array_hashes'][k] for k,v in z.items())
  by={x['path']:x for x in rows}
  for x in reps:assert x['exact'] and x['array_hashes']==by[x['reference_path']]['array_hashes'] and x['integration_sha256']==by[x['reference_path']]['integration_sha256']
  assert G.rows(folder,'parameter_applications.jsonl')==[[tag,stage,0] for tag in ['live','shadow'] for stage in ['BEFORE','AFTER']]
 else:assert not rows and not reps
 if not r['accepted']:L.load(obj,before)
 assert digest(L.state(obj))==r['retained_joint_hash'];assert s['physics']==sum(x['physics'] for x in rows+reps)<=p['max_physics'];assert s['commands']==(68 if rows else 0)
 save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,new_Adam=0,new_RL=0,independent_historical_solver_reconstruction=1,coefficients_and_duals_exact=True,finite_memory_constraint_and_KKT_rechecked=True,not_closed_loop_certificate=True,all32_native_array_and_integrations_replay_exact=bool(rows),physics=s['physics'],commands=s['commands']));print('T572R_AUDIT_PASS',arm,flush=True)
if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])
