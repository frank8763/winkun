"""Finite offline child: qualify one teacher, transfer once, test all32/exact."""
from pathlib import Path
import fcntl,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t550_offline_obstacle_relay';DOC=ROOT/'stage/docs/day_20261114'
sys.path.insert(0,str(HERE));import t550_actor as L;import t550_teacher as T
sys.path.insert(0,str(ROOT/'stage/experiments/t549_curvature_gated_transfer'));import t549_runtime as Prior
Base,F,G,E,P,np,torch=Prior.Base,Prior.F,Prior.G,Prior.E,Prior.P,L.np,L.torch
read,save,sha,digest=L.read,L.save,L.sha,L.digest
NORMAL=Prior.NORMAL_PAYLOAD
ACTIVE=OUT

def used(path):return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file() and not f.is_symlink())
def verify(p,full=False):
 root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
 assert time.time()<root['absolute_all_stop'] and used(OUT)<=root['max_bytes'] and used(ROOT/'stage/runs')-used(OUT)+root['max_bytes']+131072<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized() and set(os.sched_getaffinity(0))<={14,15}
 for path,h in (root['inputs'] if full else root['protected_files']).items():assert sha(path)==h,path

def plan(folder):
 root=read(OUT/'protocol.json');c=read(folder/'protocol.json');assert c['_root_sha256']==sha(OUT/'protocol.json');p=read(root['actor_configuration']);p.update(root);p.update(c);p['out']=str(folder);p['evaluation_jobs']=p['old8_jobs']+p['field_jobs'];return p

def setup(folder):
 global ACTIVE
 ACTIVE=folder;Base.OUT=OUT;Base.HERE=HERE;Base.verify=verify;Base.setup(folder);Base.ACTIVE=folder;F.OUT=folder;F.HERE=HERE;F.verify=verify;F.R.OUT=folder;F.R.HERE=HERE;F.R.verify=verify;G.B.ACTIVE=folder;G.B.verify=verify;P.frozen=L.frozen;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
def worker(pipe,index,p,counts):
 G.T.Session=T.Session if p.get('teacher_training') else F.FieldSession;G.B.native_setup=F.field_setup;P.frozen=L.frozen;E.payload=T.payload if p.get('teacher_training') else NORMAL;G.cpu_worker(pipe,index,p,counts)
class Pool(F.Pool):
 def __init__(self,p,counts,guard):
  self.p=p;self.guard,self.children,self.pipes=guard,[],[];ctx=G.R.M.mp.get_context('spawn')
  try:
   for i in range(2):
    a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
   self.ready=[self.receive(i) for i in range(2)];assert all(x['state']=='READY' for x in self.ready);path=Path(p['out'])/'worker_generations.json';old=read(path) if path.exists() else [];old.append(dict(parent=os.getpid(),workers=[x.pid for x in self.children],epoch=time.time(),teacher=bool(p.get('teacher_training')),CPU=[14,15],GPU=False));save(path,old)
  except BaseException:self.close();raise
 def receive(self,i):
  v=super().receive(i)
  if self.p.get('teacher_training') and isinstance(v,tuple):v[1].update(teacher=True,teacher_kind=self.p['teacher_kind'],training_teacher_NOT_autonomous=True)
  return v

def assess(p,rows):
 by={r['job_id']:r for r in rows};assert len(by)==len(rows)==32;c=Base.assess(p,[by[j['id']] for j in p['evaluation_jobs']]);prior=set(p['before_check']['field_success_ids']);now=set(c['field_success_ids']);c.update(lost_prior_field_ids=sorted(prior-now),gained_field_ids=sorted(now-prior));identity=[]
 for old in p['before_rows']:
  if old['success'] and not old['collision']:
   r=by[old['job_id']];identity.append(dict(id=r['job_id'],all_array_hashes_equal=r['array_hashes']==old['array_hashes'],integration_equal=r['integration_sha256']==old['integration_sha256'],safe=bool(r['success'] and not r['collision'])))
 c['protected_native_identity']=identity;c['all_protected_native_exact']=all(x['all_array_hashes_equal'] and x['integration_equal'] and x['safe'] for x in identity);c['retain_local_progress']=bool(c['old8']['success']==8 and prior<=now and len(now)>len(prior) and c['field24']['contact']<=p['before_check']['field24']['contact'] and c['all_protected_native_exact']);c['strict_research_gate_pass']=c['retain_local_progress'] and c['field24']['contact']<=9;c['globally_promoted']=False;return c

def main(child):
 folder=OUT/child;p=plan(folder);setup(folder);verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=Base.Runtime(p);state='FAILED';error=None;result=dict(reason='NOT_EVALUATED',teacher_qualified=False,fit_attempted=False,retained=False,model=p['initial'],check=None,accepted=False,new_Adam=0,new_RL=0,old275=None,six54=None,eight54=None,fresh16=None,reload78=None)
 try:
  if p['kind']=='final':
   rt.pool=Pool(dict(p,phase='student',teacher_training=False,native_stop_epoch=rt.deadline),rt.counts,rt.guard);vals=rt.request(p['final_jobs'],p['initial'],'FINAL_SAME_MODEL_DEVELOPMENT407');rows=[r for a,r in vals];assert len(rows)==407
   result.update(reason='FINAL407_MEASURED_NOT_STABILITY_GATE',old275=G.R.M.tally(rows[:275]),six54=G.R.M.tally(rows[275:329]),eight54=G.R.M.tally(rows[329:383]),field24=G.R.M.tally(rows[383:]),full407_safe=all(r['success'] and not r['collision'] for r in rows),fresh16=None,reload78=None,not_independent_full_model_reload=True)
  elif p['kind']=='engineering':
   rt.pool=Pool(dict(p,phase='student',teacher_training=True,native_stop_epoch=rt.deadline),rt.counts,rt.guard);vv=rt.request([p['probe_job']],p['initial'],'SHORT_NATIVE_CONTROLLER_CONTRACT',full=True);T.audit(p,vv[0][1],vv[0][0]);rt.request([p['probe_job']],p['initial'],'SHORT_NATIVE_CONTROLLER_EXACT',full=True,reference=[vv[0][1]]);result.update(reason='ENGINEERING_ONLY_NOT_NAVIGATION_SUCCESS')
  else:
   rt.pool=Pool(dict(p,phase='student',teacher_training=True,native_stop_epoch=rt.deadline),rt.counts,rt.guard);vv=rt.request([p['probe_job']],p['initial'],'ACTUAL_REACTIVE_TRAINING_RECOVERY',full=True);a,row=vv[0];T.audit(p,row,a);rt.request([p['probe_job']],p['initial'],'EXACT_REACTIVE_RECOVERY',full=True,reference=[row]);eligible=bool(row['success'] and not row['collision'] and row['field_detail']['field_qualified_success']);result.update(teacher_qualified=eligible,teacher_outcome=dict(success=row['success'],contact=row['collision'],timeout=row['timeout'],remaining_m=row['remaining_m']));rt.pool.close();rt.pool=None;rt.commands+=2
   if eligible and p.get('allow_fit',True):
    result['fit_attempted']=True;rt.guard(True)
    try:
     q,m=L.fit(p,a);shadow,other=L.fit(p,a);assert m==other and q.phash()==shadow.phash();rt.guard(True);model=L.export(q,p,folder/'proposal.pt.gz');assert L.frozen(p,model['path']).phash()==q.phash();result.update(model=model,fit=m)
    except AssertionError as ex:
     result.update(reason='FIXED_FUNCTION_QUALIFICATION_REJECTED',fit_failure=str(ex));save(folder/'qualification_rejection.json',dict(error=traceback.format_exc(),no_parameter_retry=True))
    if result.get('fit'):
     rt.pool=Pool(dict(p,phase='student',teacher_training=False,native_stop_epoch=rt.deadline),rt.counts,rt.guard);rt.commands+=2
     target=rt.request([p['probe_job']],model,'PURE_ACTOR_TARGET_FULL',full=True);otherjobs=[j for j in p['evaluation_jobs'] if j['id']!=p['probe_job']['id']];rest=rt.request(otherjobs,model,'PURE_ACTOR_OTHER31');rr=[r for a,r in target+rest];by={r['job_id']:r for r in rr};rt.request([p['probe_job']],model,'EXACT_PURE_ACTOR_TARGET_FULL',full=True,reference=[by[p['probe_job']['id']]]);rt.request(otherjobs,model,'EXACT_PURE_ACTOR_OTHER31',reference=[by[j['id']] for j in otherjobs]);c=assess(p,rr);result.update(check=c,retained=c['retain_local_progress'],reason='RETAINED_FINITE_LOCAL_PROGRESS' if c['retain_local_progress'] else 'REJECTED_CANDIDATE_PARENT_RETAINED')
   else:result['reason']='QUALIFIED_DEMO_ONLY_FIXED_FIT_QUOTA' if eligible else 'TEACHER_NOT_QUALIFIED_ZERO_FIT'
  assert sha(p['initial']['path'])==p['initial']['sha256'];state='COMPLETE'
 except BaseException:error=traceback.format_exc();save(folder/'failure.json',dict(error=error,no_restart=True,raw_preserved=True))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=2
  save(folder/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
 assert state=='COMPLETE',error

def audit(child):
 folder=OUT/child;p=plan(folder);setup(folder);verify(p,True);G.stopped(folder);assert not (folder/'audit.json').exists();s=read(folder/'summary.json');assert s['state']=='COMPLETE';rows=G.rows(folder);reps=G.rows(folder,'replays.jsonl');by={r['path']:r for r in rows}
 for r in rows:
  a=E.N.S.arrays(r['path']);assert sha(r['path'])==r['sha256'] and int(a['native_counts'].sum())==r['physics'];assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
  if r['full']:
   if r.get('teacher'):T.audit(p,r,a)
   else:F.M.audit_full_history(p,r,a)
 for r in reps:assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256']
 result=s['result'];rebuild=0
 if p['kind']=='final':assert len(rows)==407 and not reps
 elif p['kind']=='engineering':assert len(rows)==len(reps)==1
 elif result['teacher_qualified'] and result['fit_attempted']:
  assert bool(rows[0]['success'] and not rows[0]['collision'])
  if result.get('fit'):
   q,m=L.fit(p,E.N.S.arrays(rows[0]['path']));assert m==result['fit'] and q.phash()==result['model']['policy_hash'] and sha(result['model']['path'])==result['model']['sha256'];assert L.frozen(p,result['model']['path']).phash()==q.phash();assert assess(p,rows[1:])==result['check'];assert len(rows)==len(reps)==33;rebuild=1
  else:
   try:L.fit(p,E.N.S.arrays(rows[0]['path']))
   except AssertionError as ex:assert str(ex)==result['fit_failure']
   else:raise AssertionError('Qualification changed')
 else:assert len(rows)==len(reps)==1 and not result['fit_attempted']
 overhead=4 if p['kind'] in ['final','engineering'] or not result.get('fit') else 8
 assert s['physics']==sum(r['physics'] for r in rows+reps)<=p['max_physics'];assert s['commands']==len(rows)+len(reps)+overhead<=p['max_commands'];assert sha(p['initial']['path'])==p['initial']['sha256']
 save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,new_Adam=0,independent_readout_rebuilds=rebuild,physics=s['physics'],commands=s['commands'],actual_teacher_control_verified=True,all_saved_full_history_density_hidden_reward=True,all_same_candidate_replays_exact=True,accepted=False));print('T550_CHILD_AUDIT_PASS',child,flush=True)
if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])
