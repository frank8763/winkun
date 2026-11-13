"""Offline-native relay children. Durable raw results precede validation.

Only this module owns the new output namespace; ended runtime files remain
immutable. Full training histories retained, compact evaluation plus all-array
hashes independently replayed. No network/LLM dependency in science.
"""
from pathlib import Path
import copy,fcntl,io,json,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t542_offline_field_relay';DOC=ROOT/'stage/docs/overnight_20261113'
sys.path.insert(0,str(HERE));import t542_learning as L
sys.path.insert(0,str(ROOT/'stage/experiments/t519_fresh_full_actor_rl'));import t519_runtime as Old
sys.path.insert(0,str(ROOT/'stage/experiments/common'));from native_case_record import envelope
F,G,E,P,np,torch=Old.F,L.G,Old.E,L.P,L.np,L.torch
read,save,sha,digest=L.read,L.save,L.sha,L.digest
ACTIVE=OUT

def used(f):return sum(x.stat().st_size for x in Path(f).rglob('*') if x.is_file())
def verify(p,full=False):
 root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
 assert time.time()<root['absolute_all_stop'] and used(OUT)<=root['max_bytes'] and used(ROOT/'stage/runs')-used(OUT)+root['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized() and set(os.sched_getaffinity(0))<={14,15}
 for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f

def plan(folder):
 root=read(OUT/'protocol.json');child=read(Path(folder)/'protocol.json');assert child.pop('_root_sha256')==sha(OUT/'protocol.json')
 p=read(root['actor_configuration']);p.update(root);p.update(child);p['out']=str(folder);p['evaluation_jobs']=p['old8_jobs']+p['field_jobs'];assert p['out']==str(folder)
 return p

def setup(folder):
 global ACTIVE
 ACTIVE=Path(folder);F.R.setup(ACTIVE);F.OUT=ACTIVE;F.HERE=HERE;F.verify=verify;F.R.OUT=ACTIVE;F.R.HERE=HERE;F.R.verify=verify;G.B.ACTIVE=ACTIVE;G.B.verify=verify;P.frozen=L.A.frozen

def append(name,row):
 with (ACTIVE/name).open('a') as h:h.write(json.dumps(row,allow_nan=False)+'\n');h.flush();os.fsync(h.fileno())

def pack(a):
 b=io.BytesIO();np.savez_compressed(b,**{k:a[k] for k in ['actions','native_counts','end_flags']});return b.getvalue()

def assess(p,rows):
 assert len(rows)==32 and [r['job_id'] for r in rows]==[j['id'] for j in p['evaluation_jobs']]
 old=G.R.M.tally(rows[:8]);field=G.R.M.tally(rows[8:]);ids=sorted(r['job_id'] for r in rows[8:] if r['field_detail']['field_qualified_success'])
 return dict(old8=old,field24=field,field_success_ids=ids,distance_sum=sum(min(20.,r['remaining_m']) for r in rows[8:]),all_field24_pass=len(ids)==24 and old['success']==8)

def accepts(before,after):
 no_loss=after['old8']['success']==8 and set(before['field_success_ids'])<=set(after['field_success_ids']) and after['field24']['contact']<=before['field24']['contact']
 gain=len(after['field_success_ids'])>len(before['field_success_ids']);flat=len(after['field_success_ids'])==len(before['field_success_ids']) and after['distance_sum']<before['distance_sum']-.05
 return bool(no_loss and (gain or flat))

class Runtime:
 def __init__(self,p):
  self.p=p;self.start=time.time();self.deadline=min(self.start+p['main_seconds'],p['absolute_native_stop']);self.learnstop=min(self.start+p['learning_seconds'],p['absolute_learning_stop']);self.commands=2;self.rows=[];self.replays=[];self.role='SETUP';self.last=0.;self.pool=None
  self.counts=G.R.M.mp.get_context('spawn').RawArray('q',3);self.lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def guard(self,learning=False):
  E.L.check_stop();assert time.time()<(self.learnstop if learning else self.deadline)-5 and sum(self.counts)<=self.p['max_physics'] and self.commands+2<=self.p['max_commands']
  if time.time()-self.last>10:
   self.last=time.time();verify(self.p);assert used(ACTIVE)<=self.p['child_bytes'];save(ACTIVE/'status.json',dict(state=self.role,epoch=self.last,pid=os.getpid(),physics=sum(self.counts),commands=self.commands))
 def request(self,jobs,model,role,seeds=None,full=False,reference=None):
  results=[];self.role=role
  for off in range(0,len(jobs),2):
   js=jobs[off:off+2];self.guard();assert sum(self.counts)+sum(j['case']['step_cap'] for j in js)<=self.p['max_physics'] and self.commands+len(js)+2<=self.p['max_commands'];self.commands+=len(js)
   req=[]
   for i,j in enumerate(js):
    file=(reference[off+i]['path'] if full else None) if reference else str(ACTIVE/'episodes'/('full_'+str(len(self.rows)+i)+'.npz')) if full else None
    req.append(dict(model=model['path'],job=j,seed=None if seeds is None else seeds[off+i],query=False,full_path=file))
   values=self.pool.request(req);pending=[]
   # Persist BOTH returned workers before any derived caller metric can throw.
   for i,((a,r),j,request) in enumerate(zip(values,js,req)):
    tag=('replay_' if reference else 'case_')+str((len(self.replays) if reference else len(self.rows))+i)
    save(ACTIVE/(tag+'_raw.json'),dict(request=request,native_result=r))
    if reference:path=Path(reference[off+i]['path'])
    elif full:path=Path(request['full_path']);assert path.exists()
    else:
     path=ACTIVE/'episodes'/(tag+'.npz');path.parent.mkdir(exist_ok=True);b=pack(a);assert len(b)<=self.p['compact_bytes']
     with path.open('xb') as h:h.write(b);h.flush();os.fsync(h.fileno())
    pending.append((a,r,j,request,path))
   for i,(a,r,j,request,path) in enumerate(pending):
    assert r['policy_hash']==model['policy_hash'];r=envelope(j,r,a,self.p['field_limit_m'])
    r.update(role=role,seed=request['seed'],request=request,loaded_model_sha256=model['sha256'],model_sha256=model['sha256'],full=full)
    if reference:
     old=reference[off+i];assert r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256'] and r['field_detail']==old['field_detail']
     r.update(reference_path=old['path'],exact=True,independent_new_sample=False);self.replays.append(r);name='replays.jsonl'
    else:
     r.update(path=str(path),sha256=sha(path),full_arrays_generated_but_not_saved=not full);self.rows.append(r);name='cases.jsonl'
    append(name,r);results.append((a,r))
  return results

def event(which):return lambda stage,k:append('optimizer_calls.jsonl',[which,stage,k])

def main(child):
 folder=OUT/child;p=plan(folder);setup(folder);verify(p,True);E.N.offline();E.L.install_stop_handlers();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 rt=Runtime(p);state='FAILED';error=None;result=None;obj=shadow=before=None
 try:
  rt.pool=F.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
  if p['kind']=='preflight_native':
   jobs=p['probe_jobs'];vv=rt.request(jobs,p['initial'],'ENGINEERING_TRUNCATED_CAP_FULL_RESULT',full=True);rt.request(jobs,p['initial'],'ENGINEERING_EXACT',full=True,reference=[r for a,r in vv]);result=dict(scope='TRUNCATED_CAP_ENGINEERING_ONLY_NOT_NAVIGATION_ACCEPTANCE',steps=0,complete=True)
  elif p['kind']=='qualify':
   vv=rt.request(p['evaluation_jobs'],p['initial'],'FROZEN_T539_FULL32');rr=[r for a,r in vv];rt.request(p['evaluation_jobs'],p['initial'],'FROZEN_T539_FULL32_EXACT',reference=rr);check=assess(p,rr)
   result=dict(check=check,accepted=accepts(p['before_check'],check),model=p['initial'],steps=0,not_new_training=True)
  elif p['kind']=='round':
   samples=rt.request(p['training_jobs'],p['initial'],'FRESH_CURRENT_POLICY_TRAIN',seeds=p['training_seeds'],full=True);sr=[r for a,r in samples]
   rt.request(p['training_jobs'],p['initial'],'EXACT_FRESH_CURRENT_POLICY',seeds=p['training_seeds'],full=True,reference=sr)
   # Learn only from arrays reloaded from the immutable physical evidence.
   arrays=[E.N.S.arrays(r['path']) for r in sr];obj=L.make(p);shadow=L.make(p);before=L.state(obj)
   u=L.fit(obj,arrays,p,lambda:rt.guard(True),event('live'));v=L.fit(shadow,arrays,p,lambda:rt.guard(True),event('shadow'));assert u==v and digest(L.state(obj))==digest(L.state(shadow))
   result=dict(update=u,steps=u['steps'],training=G.R.M.tally(sr),accepted=False,check=None,model=p['initial'])
   if u['steps']:
    model=L.export(obj,p,folder/'proposal.pt.gz');vv=rt.request(p['evaluation_jobs'],model,'PURE_STUDENT_FULL32');rr=[r for a,r in vv];rt.request(p['evaluation_jobs'],model,'PURE_STUDENT_FULL32_EXACT',reference=rr);check=assess(p,rr)
    result.update(model=model,check=check,accepted=accepts(p['before_check'],check))
   if not result['accepted']:L.load(obj,before);L.load(shadow,before)
   result['retained_joint_hash']=digest(L.state(obj))
  elif p['kind']=='final':
   vals=rt.request(p['final_jobs'],p['initial'],'FROZEN_SELECTED_COMPLETE_DEVELOPMENT');rs=[r for a,r in vals]
   result=dict(old275=G.R.M.tally(rs[:275]),wide6=G.R.M.tally(rs[275:329]),wide8=G.R.M.tally(rs[329:383]),field24=G.R.M.tally(rs[383:]),model=p['initial'],complete_development_pass=all(r['success'] and not r['collision'] for r in rs) and all(r['field_detail']['field_qualified_success'] for r in rs[383:]),fresh16=None,reload78=None,full_stability_gate=False)
   # One development pass is not a stability claim. Fresh/reload are not
   # preimplemented for these field tasks and cannot be invented by the driver.
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
 folder=OUT/child;p=plan(folder);setup(folder);verify(p,True);s=read(folder/'summary.json');assert s['state']=='COMPLETE' and not (folder/'audit.json').exists();G.stopped(folder)
 rows=G.rows(folder);reps=G.rows(folder,'replays.jsonl');by={r['path']:r for r in rows}
 for r in rows:
  assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert int(a['native_counts'].sum())==r['physics']
  assert a['end_flags'][-1].tolist()==[int(r['collision']),int(r['success']),int(r['timeout'])]
  assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
  if r['full']:F.M.audit_full_history(p,r,a)
 for r in reps:
  old=by[r['reference_path']];assert r['exact'] and r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
 result=s['result'];steps=0
 if p['kind']=='round':
  sr=rows[:4];obj=L.make(p);before=L.state(obj);u=L.fit(obj,[E.N.S.arrays(r['path']) for r in sr],p,lambda:None,lambda *_:None);assert u==result['update'];steps=u['steps']
  if steps:
   assert digest(L.state(obj))==result['model']['joint_hash'] and sha(result['model']['path'])==result['model']['sha256'];check=assess(p,rows[4:]);assert check==result['check'] and accepts(p['before_check'],check)==result['accepted'];assert len(rows)==len(reps)==36
  else:assert len(rows)==len(reps)==4
  if not result['accepted']:L.load(obj,before)
  assert digest(L.state(obj))==result['retained_joint_hash']
  assert G.rows(folder,'optimizer_calls.jsonl')==[[w,t,k] for w in ['live','shadow'] for k in range(steps) for t in ['BEFORE','AFTER']]
 elif p['kind']=='qualify':assert assess(p,rows)==result['check'] and len(rows)==len(reps)==32
 elif p['kind']=='preflight_native':assert len(rows)==len(reps)==2
 elif p['kind']=='final':assert len(rows)==407 and not reps
 assert s['physics']==sum(r['physics'] for r in rows+reps)<=p['max_physics'] and s['commands']==len(rows)+len(reps)+4<=p['max_commands']
 save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,discarded_historical_SGD=steps,all_saved_training_full_history_density_reward_rebuilt=True,compact_evaluation_NOT_full120_offline_reconstructed=True,all_replay_full_array_hashes_exact=True,physics=s['physics'],commands=s['commands'],scope=p['kind']));print('T542_CHILD_AUDIT_PASS',child,flush=True)
if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])
