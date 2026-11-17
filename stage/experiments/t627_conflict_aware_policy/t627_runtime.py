from pathlib import Path
import sys,os,time,fcntl,traceback,copy,json,gzip,subprocess
import t627_learning as M
import t627_data as D
sys.path.insert(0,str(M.ROOT/'stage/experiments/t626_multitask_closed_loop'))
import t626_runtime as B
L,torch,np=M.L,M.torch,M.np
ROOT,HERE,OUT,DOC=M.ROOT,M.HERE,M.OUT,M.DOC
read,save,sha,digest=M.read,M.save,M.sha,M.digest
def setup(p):
 B.ROOT=ROOT;B.HERE=HERE;B.OUT=OUT;B.DOC=DOC;B.M=M;B.setup(p);B.N.restore(p)
 def stream(p,action,data,tag=None):
  cmd=[str(ROOT/'run_stage.sh'),str(HERE/'t627_stream.py'),action]+([] if tag is None else [tag.replace('T626','T627')]);c=subprocess.run(cmd,cwd=ROOT,input=data,capture_output=True,timeout=p['stream_seconds'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''));assert c.returncode==0,c.stderr.decode(errors='replace')[-1200:];return c.stdout
 B.stream=stream
def verify(p):return B.verify(p)
def fetch_all(p):
 groups=D.prior(p);receipt=read(OUT/'batch_08_receipt.json');files,manifest=D.download(p,receipt);group=[]
 for row in manifest['rows']:
  b=files[Path(row['path']).name];assert D.S.digest(b)==row['sha256'];a=B.R.E.N.S.arrays(__import__('io').BytesIO(b));assert all(digest(v)==row['array_hashes'][key] for key,v in a.items());B.N.N.inspect(p,[(a,row)]);group.append(a)
 groups.append(group);assert len(groups)==9;return groups
def main():
 p=M.plan();setup(p);verify(p);lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);rt=B.Runtime(p);error=None;results=[];gpu_lock=None
 try:
  groups=D.prior(p);rt.pool=B.N.Pool(dict(p,out=str(OUT),phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard);job=p['new_training_job'];jobs=[copy.deepcopy(job) for _ in range(4)];seeds=[[62720261117,8,i] for i in range(4)];vv=rt.request(jobs,p['student_initial'],'FRESH_TRAIN4',seeds=seeds);refs=[r for a,r in vv];newrows=B.batch_archive(p,8,vv);rt.request(jobs,p['student_initial'],'EXACT_TRAIN4',seeds=seeds,references=refs);groups.append([a for a,r in vv]);save(OUT/'DATA_ADMISSION.json',dict(groups=9,old32_reused_not_new=True,new4_successes=sum(B.R.safe(r) for r in newrows),new4_old_success_replay_not_teacher=True))
  gpu_lock=(ROOT/'stage/runs/.stage_gpu.lock').open('a+b');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for arm in p['arms']:
   rt.guard();assert time.time()+p['learning_reserve_seconds']<p['absolute_learning_stop'];M.configure_gpu(p);obj=M.make(p);shadow=M.make(p);before=M.state(obj);folder=OUT/arm;folder.mkdir();last=[0.]
   def guard():
    rt.guard();assert time.time()<p['absolute_learning_stop']-5
    if time.time()-last[0]>5:
     jobsGPU=__import__('subprocess').check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip().splitlines();assert all(int(z)==os.getpid() for z in jobsGPU if z.strip()),'Other GPU compute, yield';last[0]=time.time()
   def event(who):
    def emit(stage,k):
     with gzip.open(folder/'optimizer_calls.jsonl.gz','ab') as f:f.write((json.dumps([who,stage,k])+'\n').encode())
     save(folder/'status.json',dict(state=who,step=k,stage=stage,epoch=time.time()))
    return emit
   pc=dict(p,arm=arm);u=M.fit(obj,groups,pc,guard,event('live'));v=M.fit(shadow,groups,pc,guard,event('shadow'));assert u==v and digest(M.state(obj))==digest(M.state(shadow));mature=M.state(obj);mh=digest(mature);M.load(obj,before);assert digest(M.state(obj))==digest(before);M.load(obj,mature);assert digest(M.state(obj))==mh;obj[0].cpu();shadow[0].cpu();candidate=L.export(obj,p,folder/'proposal.pt.gz');result=dict(arm=arm,update=u,candidate=candidate,accepted=False,final49=None,model=p['student_initial']);save(folder/'LEARNING_COMPLETE.json',dict(update=u,candidate=candidate,epoch=time.time()))
   if u['applied']>0:
    B.OUT=folder
    try:
     # Two simultaneous evaluation partials fit the unchanged local cap;
     # collection still uses all three CPU workers before models exist.
     final=[]
     for off in range(0,49,2):final+=rt.request(p['evaluation_jobs'][off:off+2],candidate,'FINAL49')
     selected=[(a,r) for a,r in final if r['job_id'] in p['key_replay_ids']]
     for off in range(0,len(selected),2):
      sub=selected[off:off+2];rt.request([r['request']['job'] for a,r in sub],candidate,'EXACT_KEY4',references=[r for a,r in sub])
     ids=sorted(r['job_id'] for a,r in final if B.R.safe(r));lost=sorted(set(p['before_success_ids'])-set(ids));gained=sorted(set(ids)-set(p['before_success_ids']));accepted=not lost and bool(gained);result.update(accepted=accepted,final49=dict(success=len(ids),success_ids=ids,lost_prior=lost,gained_prior=gained),model=candidate if accepted else p['student_initial'])
    finally:B.OUT=OUT
   if not result['accepted']:M.load(obj,before);M.load(shadow,before)
   result['retained_state_sha']=digest(M.state(obj));save(folder/'summary.json',result);results.append(result);del obj,shadow;torch.cuda.empty_cache()
  fcntl.flock(gpu_lock,fcntl.LOCK_UN);gpu_lock.close();gpu_lock=None
 except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=3
  if gpu_lock:gpu_lock.close()
  save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,results=results,groups=9 if len(results) else None,new_Critic=0));fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
 assert error is None,error
def audit():
 p=M.plan();setup(p);verify(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE';assert not Path('/proc',str(s['pid'])).exists();workers=read(OUT/'worker_generations.json');assert all(not Path('/proc',str(i)).exists() for z in workers for i in [z['parent'],*z['workers']]);groups=fetch_all(p);cases=B.read_rows('cases.jsonl');reps=B.read_rows('replays.jsonl')
 for arm in p['arms']:
  B.OUT=OUT/arm
  if (B.OUT/'cases.jsonl.gz').exists():cases+=B.read_rows('cases.jsonl');reps+=B.read_rows('replays.jsonl')
 B.OUT=OUT;by={r['path']:r for r in cases}
 for r in reps:assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256']
 for result in s['results']:
  arm=result['arm'];folder=OUT/arm;M.configure_gpu(p);obj=M.make(p);before=M.state(obj);events=[]
  gpu_last=[0.]
  def guard():
   assert time.time()<p['absolute_all_stop']-5
   if time.time()-gpu_last[0]>5:
    jobsGPU=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip().splitlines();assert all(int(z)==os.getpid() for z in jobsGPU if z.strip()),'Other GPU compute during audit, yield';gpu_last[0]=time.time()
  def event(stage,k):
   events.append(['live',stage,k])
   with gzip.open(folder/'audit_optimizer_calls.jsonl.gz','ab') as f:f.write((json.dumps([stage,k])+'\n').encode())
  u=M.fit(obj,groups,dict(p,arm=arm),guard,event);assert u==result['update'];obj[0].cpu();c=L.P.checkpoint(result['candidate']['path']);assert digest(M.state(obj))==c['joint_hash'] and obj[0].phash()==result['candidate']['policy_hash']
  with gzip.open(folder/'optimizer_calls.jsonl.gz','rt') as f:calls=[json.loads(z) for z in f]
  assert [z for z in calls if z[0]=='live']==events and [['live',*z[1:]] for z in calls if z[0]=='shadow']==events
  if result['final49'] is not None:
   rows=[r for r in cases if r['role']=='FINAL49' and r['policy_hash']==result['candidate']['policy_hash']];assert len(rows)==49;ids=sorted(r['job_id'] for r in rows if B.R.safe(r));lost=sorted(set(p['before_success_ids'])-set(ids));gained=sorted(set(ids)-set(p['before_success_ids']));assert result['final49']==dict(success=len(ids),success_ids=ids,lost_prior=lost,gained_prior=gained) and result['accepted']==(not lost and bool(gained))
   for r in rows:
    path=Path(r['path']);assert sha(path)==r['sha256'];a=B.R.E.N.S.arrays(path)
    for key in ['actions','native_counts','end_flags']:assert digest(a[key])==r['array_hashes'][key]
  if not result['accepted']:M.load(obj,before)
  assert digest(M.state(obj))==result['retained_state_sha'];save(folder/'audit.json',dict(state='PASS',historical_policy_SGD_updates=u['attempted_optimizer'],full_task_gradients_CAGrad_dual_weights_optimizer_and_parameters_exact=True));del obj;torch.cuda.empty_cache()
 assert s['physics']==sum(r['physics'] for r in cases+reps)<=p['max_physics'] and s['commands']==len(cases)+len(reps)+6<=p['max_commands'];save(OUT/'audit.json',dict(state='PASS',old32_remote_raw_full_member_SHA_current_policy_density_checked=True,new4_raw_exact_and_history_checked=True,actual_optimizer_class='SGD_no_momentum',new_Critic=0,new_native=0))
if __name__=='__main__':globals()[sys.argv[1]]()
