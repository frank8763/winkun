from pathlib import Path
import sys,os,time,fcntl,traceback,copy,json,gzip,subprocess,io,shutil
import t628r_learning as M
import t628r_native as Native
sys.path.insert(0,str(M.ROOT/'stage/experiments/t626_multitask_closed_loop'))
import t626_runtime as B
L,torch,np=M.L,M.torch,M.np
ROOT,HERE,OUT,DOC=M.ROOT,M.HERE,M.OUT,M.DOC
read,save,sha,digest=M.read,M.save,M.sha,M.digest
BASE_INSPECT=B.N.N.inspect
def inspect(p,values):
 for a,r in values:
  q=M.frozen(p,r['request']['model']);pc=dict(p,raw_sigma=list(q.p['raw_sigma']));BASE_INSPECT(pc,[(a,r)])
def verify(p):
 assert p['authorization_approved'] and sha(HERE/'protocol.json')==sha(OUT/'protocol.json') and time.time()<p['absolute_all_stop'];used=lambda f:sum(x.stat().st_size for x in f.rglob('*') if x.is_file());assert used(OUT)<=p['max_bytes'] and used(ROOT/'stage/runs')-used(OUT)+p['max_bytes']+131072<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
 for f,h in p['inputs'].items():assert sha(f)==h,f
def setup(p):
 B.ROOT=ROOT;B.HERE=HERE;B.OUT=OUT;B.DOC=DOC;B.M=M;B.setup(p);B.verify=verify;B.R.Base.verify=lambda *args:verify(p);B.R.P.frozen=M.frozen;Native.install();B.N.worker=Native.worker;B.N.N.inspect=inspect;B.N.restore(p)
 def stream(p,action,data,tag=None):
  if action=='upload':
   committed=p.get('remote_preflight_bytes',0)+sum(read(f)['bytes'] for pattern in ['batch_*_receipt.json','*.remote.json'] for f in OUT.rglob(pattern))
   assert committed+len(data)+p['max_bytes']+131072<=p['new_remote_raw_bytes_cap'],'Whole experiment remote cap'
  cmd=[str(ROOT/'run_stage.sh'),str(HERE/'t628r_stream.py'),action]+([] if tag is None else [tag.replace('T626','T628R')]);c=subprocess.run(cmd,cwd=ROOT,input=data,capture_output=True,timeout=p['stream_seconds'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''));assert c.returncode==0,c.stderr.decode(errors='replace')[-1200:];return c.stdout
 B.stream=stream
def archive_model(p,model,tag):
 path=Path(model['path']);members={path.name:path.read_bytes()};blob=B.S.archive(members,dict(scope='sigma-labelled stage model, no optimizer application by export',model=model));receipt=json.loads(B.stream(p,'upload',blob,'T628R_MODEL_'+tag));assert receipt['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED';save(path.parent/(path.stem+'.remote.json'),receipt);model['remote_receipt']=str(path.parent/(path.stem+'.remote.json'))
def evict_model(model):
 path=Path(model['path']);receipt=path.parent/(path.stem+'.remote.json');r=read(receipt);member=next(z for z in r['manifest']['files'] if z['path']==path.name);assert sha(path)==model['sha256']==member['sha256'];path.unlink();return dict(model=model,receipt=str(receipt))
def gpu_check():
 jobsGPU=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip().splitlines();assert all(int(z)==os.getpid() for z in jobsGPU if z.strip()),'Other GPU compute, yield'
def ram_check(p,pids):
 import t628r_memory
 snapshot=t628r_memory.measure(pids,p['working_RAM_cap_bytes'],p['cgroup_unit']);snapshot['epoch']=time.time();save(OUT/'MEMORY_ACCOUNTING.json',snapshot)
def coverage_gate(p,groups,records):
 result=dict(old_successes=sum(B.R.safe(r) for r in records if r['job_id'] in p['old_sentinel_ids']),new_successes=sum(B.R.safe(r) for r in records if r['job_id'] in p['new_sentinel_ids']),return_spread_by_task=[float(np.ptp([float(a['rewards'].sum()) for a in g])) for g in groups])
 result['pass_gate']=result['old_successes']>0 and result['new_successes']>0 and any(x>1e-6 for x in result['return_spread_by_task']);return result
def sentinel_gate(p,rows):
 assert len(rows)==4 and sorted(r['job_id'] for r in rows)==sorted(p['old_sentinel_ids']+p['new_sentinel_ids'])
 return all(B.R.safe(r) for r in rows if r['job_id'] in p['old_sentinel_ids']) and any(B.R.safe(r) for r in rows if r['job_id'] in p['new_sentinel_ids'])
def final_metrics(p,rows):
 assert len(rows)==49 and sorted(r['job_id'] for r in rows)==sorted(r['id'] for r in p['evaluation_jobs'])
 ids=sorted(r['job_id'] for r in rows if B.R.safe(r));return dict(success=len(ids),success_ids=ids,lost_prior=sorted(set(p['before_success_ids'])-set(ids)),gained_prior=sorted(set(ids)-set(p['before_success_ids'])))
def compact_check(row):
 assert sha(row['path'])==row['sha256'];arrays=B.R.E.N.S.arrays(row['path'])
 for key in ['actions','native_counts','end_flags']:assert digest(arrays[key])==row['array_hashes'][key]
def main():
 p=M.plan();setup(p);verify(p);lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);rt=B.Runtime(p);error=None;results=[];gpu_lock=None
 original_guard=rt.guard;last_gpu_check=[0.]
 def runtime_guard(*args):
  original_guard(*args);assert sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())<=p['max_bytes']
  if time.time()-last_gpu_check[0]>5:gpu_check();ram_check(p,[os.getpid()]+([] if rt.pool is None else [c.pid for c in rt.pool.children]));last_gpu_check[0]=time.time()
 rt.guard=runtime_guard
 original_request=rt.request
 def capacity_request(jobs,model,role,*args,**kwargs):
  # Includes a full failed/current batch, or a full interrupted eval payload.
  pending=len(jobs)*(p['fresh_bytes'] if role in ['FRESH_TRAIN4','EXACT_TRAIN4'] else p['full_bytes'])
  assert sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())+pending+p['metadata_write_reserve']<=p['max_bytes'],'Local complete-batch admission'
  return original_request(jobs,model,role,*args,**kwargs)
 rt.request=capacity_request
 try:
  rt.pool=B.N.Pool(dict(p,out=str(OUT),phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard);gpu_lock=(ROOT/'stage/runs/.stage_gpu.lock').open('a+b');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for arm in p['arms']:
   obj=shadow=None;accepted_model=p['student_initial'];arm_results=[];arm_before=None;candidate=None;arm_stop_reason=None
   for rnd in range(p['rounds']):
    if rt.commands+40+53+3>p['max_commands']:
     arm_stop_reason='FULL_ROUND_AND_FINAL_COMMAND_RESERVE_INSUFFICIENT';break
    rt.guard();assert time.time()+p['round_learning_admission_seconds']<p['absolute_learning_stop'];folder=OUT/arm/f'round_{rnd}';folder.mkdir(parents=True);sigma=p['sigmas_by_arm'][arm][rnd];pc=dict(p,raw_sigma=list(sigma));gpu_check();M.configure_gpu(p)
    if obj is None:obj=M.make(p,[.2,.2]);shadow=M.make(p,[.2,.2]);arm_before=M.state(obj)
    M.set_sigma(obj,sigma);M.set_sigma(shadow,sigma)
    before=M.state(obj);assert digest(before)==digest(M.state(shadow));obj[0].cpu();start_model=M.export(obj,p,folder/'start.pt.gz');archive_model(p,start_model,arm+str(rnd)+'start');obj[0].cuda();B.OUT=folder;groups=[];records=[]
    if rnd>0:
     previous=OUT/arm/f'round_{rnd-1}'/'proposal.pt.gz'
     if previous.exists():evict_model(arm_results[rnd-1]['candidate'])
    for index,job in enumerate(p['training_jobs']):
     group=[]
     for half in range(2):
      jobs=[copy.deepcopy(job) for _ in range(2)];seeds=[[p['seed'],rnd,index,half*2+i] for i in range(2)];vv=rt.request(jobs,start_model,'FRESH_TRAIN4',seeds=seeds);refs=[r for a,r in vv]
      try:rows=B.batch_archive(pc,index*2+half,vv)
      except BaseException:B.fallback(p,index*2+half,vv);raise
      rt.request(jobs,start_model,'EXACT_TRAIN4',seeds=seeds,references=refs);group.extend(a for a,r in vv);records+=rows
     groups.append(group)
    coverage=coverage_gate(p,groups,records);save(folder/'DATA_ADMISSION.json',coverage)
    if not coverage['pass_gate']:
     M.load(obj,before);M.load(shadow,before);save(folder/'summary.json',dict(state='COVERAGE_INSUFFICIENT',round=rnd,sigma=sigma,coverage=coverage,optimizer_updates=0,model=start_model,retained_state_sha=digest(before)));arm_results.append(read(folder/'summary.json'));break
    last=[0.]
    def guard():
     rt.guard();verify(p);assert time.time()<p['absolute_learning_stop']-5
     if time.time()-last[0]>5:
      jobsGPU=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip().splitlines();assert all(int(z)==os.getpid() for z in jobsGPU if z.strip()),'Other GPU compute, yield';last[0]=time.time()
    def event(who):
     def emit(stage,k):
      with gzip.open(folder/'optimizer_calls.jsonl.gz','ab') as f:f.write((json.dumps([who,stage,k])+'\n').encode())
      save(folder/'status.json',dict(state=who,stage=stage,step=k,epoch=time.time()))
     return emit
    u=M.fit(obj,groups,pc,guard,event('live'));v=M.fit(shadow,groups,pc,guard,event('shadow'));assert u==v and digest(M.state(obj))==digest(M.state(shadow));mature=M.state(obj);mh=digest(mature);M.load(obj,before);assert digest(M.state(obj))==digest(before);M.load(obj,mature);assert digest(M.state(obj))==mh;obj[0].cpu();candidate=M.export(obj,p,folder/'proposal.pt.gz');archive_model(p,candidate,arm+str(rnd)+'candidate');obj[0].cuda();save(folder/'LEARNING_COMPLETE.json',dict(update=u,candidate=candidate,start_model=start_model,epoch=time.time(),before_state_sha=digest(before)))
    if u['applied']==0:
     M.load(obj,before);M.load(shadow,before);result=dict(state='NO_APPLIED_PARAMETER_UPDATE',round=rnd,sigma=sigma,coverage=coverage,update=u,candidate=candidate,start_model=start_model,passed_sentinel=False,retained_state_sha=digest(before));save(folder/'summary.json',result);arm_results.append(result);evict_model(start_model);break
    sent=[]
    for off in range(4):sent+=rt.request(p['sentinel_jobs'][off:off+1],candidate,'FINAL49')
    sr=[r for a,r in sent]
    for off in range(4):rt.request(p['sentinel_jobs'][off:off+1],candidate,'EXACT_SENTINEL4',references=sr[off:off+1])
    passed=sentinel_gate(p,sr);result=dict(state='COMPLETE',round=rnd,sigma=sigma,coverage=coverage,update=u,candidate=candidate,start_model=start_model,sentinel_rows=sr,passed_sentinel=passed,retained_state_sha=None)
    if not passed:M.load(obj,before);M.load(shadow,before)
    else:accepted_model=candidate
    result['retained_state_sha']=digest(M.state(obj));save(folder/'summary.json',result);arm_results.append(result);evict_model(start_model)
    if not passed:break
   B.OUT=OUT/arm;final=None;keep=False
   if accepted_model['path']!=p['student_initial']['path']:
    ensure_model(p,accepted_model)
    vals=[]
    for off in range(49):vals+=rt.request(p['evaluation_jobs'][off:off+1],accepted_model,'FINAL49')
    selected=[(a,r) for a,r in vals if r['job_id'] in p['key_replay_ids']]
    for a,r in selected:rt.request([r['request']['job']],accepted_model,'EXACT_KEY4',references=[r])
    final=final_metrics(p,[r for a,r in vals]);keep=not final['lost_prior'] and bool(final['gained_prior'])
   if obj is not None and not keep:M.load(obj,arm_before)
   elif obj is not None:M.set_sigma(obj,accepted_model['sigma'])
   ar=dict(arm=arm,stop_reason=arm_stop_reason,rounds=arm_results,final49=final,accepted=keep,model=accepted_model if keep else p['student_initial'],retained_state_sha=digest(M.state(obj)) if obj is not None else None);save(OUT/arm/'summary.json',ar);results.append(ar)
   # Archive before evicting old arm model, leave room for the next four cache.
   for item in arm_results:
    for key in ['candidate','start_model','model']:
     model=item.get(key)
     if model and Path(model['path']).exists():evict_model(model)
   del obj,shadow;torch.cuda.empty_cache();B.OUT=OUT
  fcntl.flock(gpu_lock,fcntl.LOCK_UN);gpu_lock.close();gpu_lock=None
 except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
 finally:
  B.OUT=OUT
  if rt.pool:rt.pool.close();rt.commands+=3
  if gpu_lock:gpu_lock.close()
  save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,results=results,new_Critic=0));fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
 assert error is None,error
def retrieve(p,receipt):
 files,manifest=B.S.unpack(B.stream(p,'download',json.dumps(receipt).encode()));return files,manifest
def ensure_model(p,model):
 path=Path(model['path'])
 if not path.exists():
  receipt=read(path.parent/(path.stem+'.remote.json'));files,meta=retrieve(p,receipt);blob=files[path.name];assert B.S.digest(blob)==model['sha256'];path.write_bytes(blob)
 assert sha(path)==model['sha256']
def audit():
 p=M.plan();setup(p);verify(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE';assert not Path('/proc',str(s['pid'])).exists();g=read(OUT/'worker_generations.json');assert all(not Path('/proc',str(i)).exists() for z in g for i in [z['parent'],*z['workers']]);cases=[];reps=[];updates=0
 for arm in s['results']:
  gpu_check();M.configure_gpu(p);obj=M.make(p,[.2,.2]);arm_before=M.state(obj);arm_cases=[];accepted_model=p['student_initial']
  for rnd,item in enumerate(arm['rounds']):
   folder=OUT/arm['arm']/f'round_{rnd}';B.OUT=folder;groups=[];records=[];sigma=p['sigmas_by_arm'][arm['arm']][rnd];assert item['sigma']==sigma;M.set_sigma(obj,sigma);before=M.state(obj)
   start=item.get('start_model',item.get('model'));ensure_model(p,start);allgroups=[[] for _ in range(4)]
   assert digest(before)==start['joint_hash'] and obj[0].phash()==start['policy_hash']
   for index in range(8):
    files,manifest=retrieve(p,read(folder/f'batch_{index:02d}_receipt.json'));group=allgroups[index//2]
    for r in manifest['rows']:
     b=files[Path(r['path']).name];assert B.S.digest(b)==r['sha256'];a=B.R.E.N.S.arrays(io.BytesIO(b));assert all(digest(v)==r['array_hashes'][k] for k,v in a.items());inspect(p,[(a,r)]);group.append(a);records.append(r)
   groups=allgroups;assert all(len(group)==4 for group in groups)
   current=B.read_rows('cases.jsonl');replays=B.read_rows('replays.jsonl');assert len([r for r in current if r['role']=='FRESH_TRAIN4'])==16 and len([r for r in replays if r['role']=='EXACT_TRAIN4'])==16;cases+=current;arm_cases+=current;reps+=replays
   assert coverage_gate(p,groups,records)==item['coverage']==read(folder/'DATA_ADMISSION.json')
   if item['state']=='COVERAGE_INSUFFICIENT':
    assert not item['coverage']['pass_gate'] and item['optimizer_updates']==0 and digest(M.state(obj))==item['retained_state_sha'];evict_model(start);continue
   candidate=item['candidate'];ensure_model(p,candidate);events=[]
   last=[0.]
   def guard():
    assert time.time()<p['absolute_all_stop']-5
    if time.time()-last[0]>5:gpu_check();ram_check(p,[os.getpid()]);last[0]=time.time()
   u=M.fit(obj,groups,dict(p,raw_sigma=sigma),guard,lambda stage,k:events.append(['live',stage,k]));assert u==item['update'];updates+=u['attempted_optimizer'];obj[0].cpu();assert digest(M.state(obj))==candidate['joint_hash']==M.L.P.checkpoint(candidate['path'])['joint_hash'] and obj[0].phash()==candidate['policy_hash'];obj[0].cuda()
   with gzip.open(folder/'optimizer_calls.jsonl.gz','rt') as f:calls=[json.loads(z) for z in f]
   assert [z for z in calls if z[0]=='live']==events and [['live',*z[1:]] for z in calls if z[0]=='shadow']==events
   if item['state']=='NO_APPLIED_PARAMETER_UPDATE':assert u['applied']==0 and not any(r['role']=='FINAL49' for r in current)
   else:
    sent=[r for r in current if r['role']=='FINAL49'];assert sent==item['sentinel_rows'] and sentinel_gate(p,sent)==item['passed_sentinel'];assert len([r for r in replays if r['role']=='EXACT_SENTINEL4'])==4
    for row in sent:compact_check(row)
   if not item['passed_sentinel']:M.load(obj,before)
   else:accepted_model=candidate
   assert digest(M.state(obj))==item['retained_state_sha'];evict_model(start);evict_model(candidate)
  B.OUT=OUT/arm['arm']
  if (B.OUT/'cases.jsonl.gz').exists():
   current=B.read_rows('cases.jsonl');arm_cases+=current;cases+=current;reps+=B.read_rows('replays.jsonl');assert sorted(r['job_id'] for r in B.read_rows('replays.jsonl'))==sorted(p['key_replay_ids'])
  if arm['final49'] is not None:
   final=[r for r in arm_cases if r['role']=='FINAL49' and r['path'].startswith(str(OUT/arm['arm']/'evaluation'))];assert final_metrics(p,final)==arm['final49'];assert arm['accepted']==(not arm['final49']['lost_prior'] and bool(arm['final49']['gained_prior']))
   for row in final:compact_check(row)
   assert accepted_model['path']!=p['student_initial']['path'] and all(r['model_sha256']==accepted_model['sha256'] for r in final)
  else:assert accepted_model['path']==p['student_initial']['path'] and not arm['accepted']
  if arm['accepted']:M.set_sigma(obj,accepted_model['sigma']);assert arm['model']==accepted_model
  else:M.load(obj,arm_before);assert arm['model']==p['student_initial']
  assert digest(M.state(obj))==arm['retained_state_sha']
  del obj;torch.cuda.empty_cache()
 B.OUT=OUT;by={r['path']:r for r in cases}
 for r in reps:assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256']
 assert updates<=p['historical_actor_updates_cap'] and s['physics']==sum(r['physics'] for r in cases+reps)<=p['max_physics'] and s['commands']==len(cases)+len(reps)+6<=p['max_commands'];save(OUT/'audit.json',dict(state='PASS',historical_Actor_SGD=updates,sigma_real_sampling_density_hash_and_export=True,full_round_history_Actor_optimizer_reconstructed=True,final_arm_full_state_reconstructed=True,coverage_sentinel_final49_gates_recomputed=True,raw_remote_all_member_SHA=True,new_native=0,new_Critic=0))
if __name__=='__main__':globals()[sys.argv[1]]()
