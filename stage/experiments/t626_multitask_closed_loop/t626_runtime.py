from pathlib import Path
import os,sys,time,json,copy,io,subprocess,fcntl,traceback,shutil,gzip
import t626_learning as M
import t626_native as N
import t626_stream as S
L,torch,np=M.L,M.torch,M.np
ROOT,HERE,OUT,DOC=M.ROOT,M.HERE,M.OUT,M.DOC
read,save,sha,digest=M.read,M.save,M.sha,M.digest
sys.path.insert(0,str(ROOT/'stage/experiments/t608_current_policy_memory'));import t608_runtime as R
def used(folder):return sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())
def verify(p):
 assert p['authorization_approved'] and sha(HERE/'protocol.json')==sha(OUT/'protocol.json') and time.time()<p['absolute_all_stop'];assert used(OUT)<=p['max_bytes'] and used(ROOT/'stage/runs')-used(OUT)+p['max_bytes']+131072<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
 for f,h in p['inputs'].items():assert sha(f)==h,f
def append(name,row):
 with gzip.open(OUT/(name+'.gz'),'ab') as f:f.write((json.dumps(row,separators=(',',':'))+'\n').encode())
def read_rows(name):
 with gzip.open(OUT/(name+'.gz'),'rt') as f:return [json.loads(z) for z in f if z.strip()]
def setup(p):
 R.NIGHT=OUT;R.HERE=HERE;R.DOC=DOC;R.setup(OUT,'round');R.Base.verify=lambda *args:verify(p);R.P.frozen=L.frozen;N.restore(p)
class Runtime:
 def __init__(self,p):
  self.p=p;self.start=time.time();self.deadline=min(self.start+p['main_seconds'],p['absolute_native_stop']);self.counts=N.G.R.M.mp.get_context('spawn').RawArray('q',4);self.commands=3;self.pool=None;self.rows=[];self.replays=[];self.last=0.
 def guard(self,*args):
  assert time.time()<self.deadline-5 and sum(self.counts)<=self.p['max_physics'] and self.commands+3<=self.p['max_commands'];assert used(OUT)<=self.p['max_bytes']
  if time.time()-self.last>5:self.last=time.time();save(OUT/'status.json',dict(state='NATIVE_OR_STREAM',physics=sum(self.counts),commands=self.commands,pid=os.getpid(),epoch=time.time()))
 def request(self,jobs,model,role,seeds=None,references=None):
  values=[]
  for off in range(0,len(jobs),3):
   self.guard();js=jobs[off:off+3];assert sum(self.counts)+sum(j['case']['step_cap'] for j in js)<=self.p['max_physics'];assert self.commands+len(js)+3<=self.p['max_commands'];self.commands+=len(js);req=[dict(model=model['path'],job=j,seed=None if seeds is None else seeds[off+i],query=False,full_path=str(OUT/'batch_cache'/f'episode_{len(self.rows)+i:03d}.npz') if role=='FRESH_TRAIN4' else None) for i,j in enumerate(js)];returned=self.pool.request(req)
   # Pool.receive has already persisted every original request/native result
   # losslessly, before this derived envelope or any metric can throw.
   for i,((a,r),request,j) in enumerate(zip(returned,req,js)):
    r=R.Base.envelope(j,r,a,self.p['field_limit_m']);r.update(role=role,seed=request['seed'],request=request,loaded_model_sha256=model['sha256'],model_sha256=model['sha256'],full=True);assert r['policy_hash']==model['policy_hash']
    if references is not None:
     old=references[off+i]
     if not (r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256'] and r['field_detail']==old['field_detail']):
      path=OUT/f'exact_mismatch_{len(self.replays)}.npz';path.write_bytes(arrays_blob(a));raise AssertionError('Exact replay mismatch, raw saved')
     r.update(reference_path=old['path'],exact=True);self.replays.append(r);append('replays.jsonl',r)
    else:
     if role=='FINAL49':
      path=OUT/'evaluation'/f"{r['job_id']}.npz";path.parent.mkdir(exist_ok=True);blob=R.Base.pack(a);assert len(blob)<=self.p['compact_bytes'];path.write_bytes(blob);r.update(path=str(path),sha256=sha(path),full=False,full_arrays_generated_but_not_saved=True)
     else:
      path=Path(request['full_path']);assert path.exists() and path.stat().st_size<=self.p['fresh_bytes'];r.update(path=str(path),sha256=sha(path))
     self.rows.append(r);append('cases.jsonl',r)
    try:N.N.inspect(self.p,[(a,r)])
    except BaseException:
     if references is not None:(OUT/f'exact_contract_failure_{len(self.replays)}.npz').write_bytes(arrays_blob(a))
     raise
    values.append((a,r))
  return values
def arrays_blob(a):return N.N.blob(R.E.N.S.packed(a))
def stream(p,action,data,tag=None):
 cmd=[str(ROOT/'run_stage.sh'),str(HERE/'t626_stream.py'),action]+([] if tag is None else [tag]);c=subprocess.run(cmd,cwd=ROOT,input=data,capture_output=True,timeout=p['stream_seconds'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''));assert c.returncode==0,c.stderr.decode(errors='replace')[-1800:];return c.stdout
def batch_archive(p,index,vv):
 members={Path(r['path']).name:Path(r['path']).read_bytes() for a,r in vv};assert all(len(z)<=p['fresh_bytes'] for z in members.values());rows=[dict(r,sha256=S.digest(members[Path(r['path']).name])) for a,r in vv]
 blob=S.archive(members,dict(rows=rows,group=index,scope='four fresh student trajectories; actual full arrays; exact replay separately charged'));archived_bytes=sum(read(f)['bytes'] for f in OUT.glob('batch_*_receipt.json'));assert archived_bytes+len(blob)+p['max_bytes']+131072<=p['new_remote_raw_bytes_cap'];receipt=json.loads(stream(p,'upload',blob,f'T626_BATCH{index:02d}'));assert receipt['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED' and receipt['sha256']==S.digest(blob);save(OUT/f'batch_{index:02d}_receipt.json',receipt)
 # Evict only this new batch's byte-identical local cache after full remote
 # readback; original bytes remain in the manifest-addressed release asset.
 for r in rows:
  path=Path(r['path']);assert path.parent==OUT/'batch_cache' and sha(path)==r['sha256'];path.unlink();append('cache_evictions.jsonl',dict(path=str(path),sha256=r['sha256'],remote_receipt=f'batch_{index:02d}_receipt.json',full_remote_every_member_verified=True))
 return rows
def fallback(p,index,vv):
 for a,r in vv:assert Path(r['path']).exists() and sha(r['path'])==r['sha256']
 save(OUT/'NETWORK_STOP.json',dict(batch=index,raw_preserved=True,no_training_started=True))
def gate(p,groups,rows):
 new=set(p['training_job_ids'])-set(p['before_success_ids']);successnew=sum(R.safe(r) for group in rows for r in group if r['job_id'] in new);old=sum(R.safe(r) for group in rows for r in group if r['job_id'] in p['before_success_ids']);return dict(pass_gate=successnew>=1 and old>=1,new_task_exploration_successes=successnew,old_task_exploration_successes=old)
def main():
 p=M.plan();setup(p);verify(p);lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);rt=Runtime(p);groups=[];group_rows=[];error=None;result=None;obj=shadow=before=None
 try:
  rt.pool=N.Pool(dict(p,out=str(OUT),phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
  for index,job in enumerate(p['training_jobs']):
   jobs=[copy.deepcopy(job) for _ in range(4)];seeds=[[p['seed'],index,i] for i in range(4)];vv=rt.request(jobs,p['student_initial'],'FRESH_TRAIN4',seeds=seeds);ref=[r for a,r in vv]
   try:rows=batch_archive(p,index,vv)
   except BaseException:fallback(p,index,vv);raise
   rt.request(jobs,p['student_initial'],'EXACT_TRAIN4',seeds=seeds,references=ref);groups.append([a for a,r in vv]);group_rows.append(rows);save(OUT/'TRAINING_PROGRESS.json',dict(groups=len(groups),physics=sum(rt.counts),commands=rt.commands))
  coverage=gate(p,groups,group_rows);save(OUT/'DATA_ADMISSION.json',coverage);result=dict(coverage=coverage,accepted=False,candidate=None,final49=None,model=p['student_initial'])
  if coverage['pass_gate']:
   assert time.time()+p['learning_reserve_seconds']<p['absolute_learning_stop'];assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();gpu_lock=(ROOT/'stage/runs/.stage_gpu.lock').open('a+b');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);M.configure_gpu(p);obj=M.make(p);shadow=M.make(p);before=M.state(obj);assert digest(before)==digest(M.state(shadow));gpu_last=[0.]
   def learn_guard():
    rt.guard();assert time.time()<p['absolute_learning_stop']-5
    if time.time()-gpu_last[0]>5:
     jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip().splitlines();assert all(int(z)==os.getpid() for z in jobs if z.strip());gpu_last[0]=time.time()
   event=lambda who:(lambda stage,k:append('optimizer_calls.jsonl',[who,stage,k]));u=M.fit(obj,groups,p,learn_guard,event('live'));v=M.fit(shadow,groups,p,learn_guard,event('shadow'));assert u==v and digest(M.state(obj))==digest(M.state(shadow));mature=M.state(obj);mature_hash=digest(mature);M.load(obj,before);assert digest(M.state(obj))==digest(before);M.load(obj,mature);assert digest(M.state(obj))==mature_hash;obj[0].cpu();shadow[0].cpu();candidate=L.export(obj,p,OUT/'proposal.pt.gz');save(OUT/'LEARNING_COMPLETE.json',dict(update=u,candidate=candidate,mature_Actor_Adam_rollback_exact=True,epoch=time.time()));fcntl.flock(gpu_lock,fcntl.LOCK_UN);gpu_lock.close();result.update(update=u,candidate=candidate)
   if u['applied']>0:
    final=rt.request(p['evaluation_jobs'],candidate,'FINAL49');rows=[r for a,r in final]
    selected=[(a,r) for a,r in final if r['job_id'] in p['key_replay_ids']];rt.request([r['request']['job'] for a,r in selected],candidate,'EXACT_KEY4',references=[r for a,r in selected]);ids=sorted(r['job_id'] for r in rows if R.safe(r));lost=sorted(set(p['before_success_ids'])-set(ids));gained=sorted(set(ids)-set(p['before_success_ids']));accepted=not lost and bool(gained);result.update(final49=dict(success=len(ids),success_ids=ids,lost_prior=lost,gained_prior=gained),accepted=accepted,model=candidate if accepted else p['student_initial'])
    if not accepted:M.load(obj,before);M.load(shadow,before)
   else:
    M.load(obj,before);M.load(shadow,before);result['stop_reason']='NO_APPLIED_PARAMETER_UPDATE_NO_REPEAT_INITIAL49'
   result['retained_state_sha']=digest(M.state(obj))
 except BaseException:
  error=traceback.format_exc()
  if obj is not None and before is not None:M.load(obj,before)
  save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=3
  save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,groups=len(groups),result=result,new_Critic=0));fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
 assert error is None,error
def audit():
 p=M.plan();setup(p);verify(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE';g=read(OUT/'worker_generations.json');assert all(not Path('/proc',str(i)).exists() for z in g for i in [z['parent'],*z['workers']]);assert not Path('/proc',str(s['pid'])).exists();groups=[];rows=[]
 for index in range(s['groups']):
  receipt=read(OUT/f'batch_{index:02d}_receipt.json');data=stream(p,'download',json.dumps(receipt).encode());files,manifest=S.unpack(data);group=[]
  for r in manifest['rows']:
   b=files[Path(r['path']).name];assert S.digest(b)==r['sha256'];a=R.E.N.S.arrays(io.BytesIO(b))
   assert all(digest(v)==r['array_hashes'][k] for k,v in a.items());N.N.inspect(p,[(a,r)]);group.append(a);rows.append(r)
  groups.append(group)
 result=s['result'];assert gate(p,groups,[read(OUT/f'batch_{i:02d}_receipt.json')['manifest']['rows'] for i in range(len(groups))])==result['coverage']
 if result['candidate'] is not None:
  gpu_last=[0.]
  def audit_guard():
   assert time.time()<p['absolute_all_stop']-5
   if time.time()-gpu_last[0]>5:
    jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip().splitlines();assert all(int(z)==os.getpid() for z in jobs if z.strip());gpu_last[0]=time.time()
  def audit_event(stage,k):events.append(['live',stage,k]);append('historical_optimizer_calls.jsonl',[stage,k])
  M.configure_gpu(p);obj=M.make(p);before=M.state(obj);events=[];u=M.fit(obj,groups,p,audit_guard,audit_event);assert u==result['update'];obj[0].cpu();checkpoint=L.P.checkpoint(result['candidate']['path']);assert digest(M.state(obj))==checkpoint['joint_hash'] and obj[0].phash()==result['candidate']['policy_hash'];calls=read_rows('optimizer_calls.jsonl');assert [z for z in calls if z[0]=='live']==events and [['live',*z[1:]] for z in calls if z[0]=='shadow']==events
  if not result['accepted']:M.load(obj,before)
  assert digest(M.state(obj))==result['retained_state_sha']
 cases=read_rows('cases.jsonl');reps=read_rows('replays.jsonl');by={r['path']:r for r in cases}
 for r in reps:assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256']
 if result['final49'] is not None:
  final=[r for r in cases if r['role']=='FINAL49'];assert len(final)==49
  for r in final:
   assert sha(r['path'])==r['sha256'];a=R.E.N.S.arrays(r['path'])
   for key in ['actions','native_counts','end_flags']:assert digest(a[key])==r['array_hashes'][key]
  ids=sorted(r['job_id'] for r in final if R.safe(r));lost=sorted(set(p['before_success_ids'])-set(ids));gained=sorted(set(ids)-set(p['before_success_ids']));assert result['final49']==dict(success=len(ids),success_ids=ids,lost_prior=lost,gained_prior=gained) and result['accepted']==(not lost and bool(gained))
 assert s['physics']==sum(r['physics'] for r in cases+reps)<=p['max_physics'] and s['commands']==len(cases)+len(reps)+6<=p['max_commands'];save(OUT/'audit.json',dict(state='PASS',remote_raw_every_member_loaded_and_SHA_verified=True,actual_history_density_reward_all_memory_checked=True,live_shadow_and_independent_Actor_Adam_reconstructed=True,full49_compact_later_not_full_pose_reconstruction=True,new_native=0,historical_Actor_Adam=0 if result['candidate'] is None else result['update']['attempted_Adam']))
if __name__=='__main__':globals()[sys.argv[1]]()
