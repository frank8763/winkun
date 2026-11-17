"""Offline finite stages; every original training/history array remains local."""
from pathlib import Path
import sys,os,time,copy,json,gzip,io,fcntl,traceback,shutil,subprocess
import t629_learning as M
import t629_native as Native
sys.path.insert(0,str(M.ROOT/'stage/experiments/t626_multitask_closed_loop'))
import t626_runtime as B
ROOT,HERE,OUT,DOC=M.ROOT,M.HERE,M.OUT,M.DOC
read,save,sha,digest=M.read,M.save,M.sha,M.digest
ORIGINAL_ENVELOPE=B.R.Base.envelope
def envelope(job,row,arrays,limit):
 result=ORIGINAL_ENVELOPE(job,row,arrays,limit)
 if 'student_takeover_first' in arrays:result.update(forced_prefix_recovery_diagnostic=True,deployment_from_original_start=False,training_teacher_suffix_only=True,network_own_history_hidden_reconstructed=True)
 return result

def used(folder):return sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())
def verify(p):
 root=read(OUT/'protocol.json');assert root['authorization_approved'] and p['_root_sha256']==sha(OUT/'protocol.json');assert time.time()<root['absolute_all_stop'];assert used(OUT)<=root['max_bytes'] and used(ROOT/'stage/runs')-used(OUT)+root['max_bytes']+131072<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
 for f,h in root['inputs'].items():assert sha(f)==h,f

def setup(p,folder):
 B.ROOT=ROOT;B.HERE=HERE;B.OUT=folder;B.DOC=DOC;B.M=M;B.setup(p);B.verify=verify;B.R.Base.verify=lambda *args:verify(p);Native.install();B.N.worker=Native.worker;B.N.N.inspect=Native.inspect;B.N.restore(p);B.R.Base.envelope=envelope

def pchild(name):
 root=M.plan();p=dict(root,**read(OUT/name/'protocol.json'));p['_root_sha256']=sha(OUT/'protocol.json');return p

def safe(row):return B.R.safe(row)
def metrics(p,rows):
 assert len(rows)==len(p['evaluation_jobs']);assert sorted(r['job_id'] for r in rows)==sorted(j['id'] for j in p['evaluation_jobs']);ids=sorted(r['job_id'] for r in rows if safe(r));return dict(success=len(ids),success_ids=ids,lost_prior=sorted(set(p['before_success_ids'])-set(ids)),gained_prior=sorted(set(ids)-set(p['before_success_ids'])))
def accepted(result):return not result['lost_prior'] and bool(result['gained_prior'])
def load_values(rows):
 values=[]
 for row in rows:
  assert sha(row['path'])==row['sha256'];a=B.R.E.N.S.arrays(row['path']);assert all(digest(v)==row['array_hashes'][k] for k,v in a.items());Native.inspect(M.plan(),[(a,row)]);values.append((a,row))
 return values

def groups_from(p):
 rows=read(OUT/'collect/summary.json')['result']['training_rows'];groups=[];ids=[]
 for job in p['training_jobs']:
  selected=[r for r in rows if r['job_id']==job['id']];assert len(selected)==2;groups.append([a for a,r in load_values(selected)]);ids.append(job['id'])
 if p['kind']=='core':
  selected=[k for k,id in enumerate(ids) if id in p['core_ids']];groups=[groups[k] for k in selected];ids=[ids[k] for k in selected]
 old=sum(id in p['before_success_ids'] for id in ids);new=len(ids)-old;assert old and new;weights=[.5/(old if id in p['before_success_ids'] else new) for id in ids];return groups,weights

def gpu_guard(p):
 assert time.time()<p['absolute_learning_stop']-5
 processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip().splitlines();assert all(int(pid)==os.getpid() for pid in processes if pid.strip()),'Other GPU compute: yield'

def main(name):
 p=pchild(name);folder=OUT/name;setup(p,folder);verify(p);lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);rt=B.Runtime(p);error=None;result={};obj=None;before=None
 def request(jobs,model,role,seeds=None,refs=None,full=False):
  values=[]
  for off in range(0,len(jobs),2):
   js=jobs[off:off+2];assert used(OUT)+sum(p['full_bytes'] for _ in js)+131072<p['max_bytes'];mode='FRESH_TRAIN4' if full else 'FINAL49';mode=role if refs is not None else mode
   returned=rt.request(js,model,mode,seeds=None if seeds is None else seeds[off:off+2],references=None if refs is None else refs[off:off+2]);values+=returned
  return values
 def evaluate(model):
  vals=request(p['evaluation_jobs'],model,'EVALUATION49');rows=[r for a,r in vals];keys=[(a,r) for a,r in vals if r['job_id'] in p['key_replay_ids']];request([r['request']['job'] for a,r in keys],model,'EXACT_KEY4',refs=[r for a,r in keys]);m=metrics(p,rows);return dict(final49=m,evaluation_rows=rows,accepted=accepted(m),model=model if accepted(m) else (p['student_initial'] if p['kind']=='recovery_rl' else p['reference_model']))
 def fit(kind,groups=None,weights=None,corrections=None,anchors=None):
  nonlocal obj,before
  gpu_guard(p);M.configure_gpu(p);obj=M.make(p,p['raw_sigma']);shadow=M.make(p,p['raw_sigma']);before=M.state(obj);assert digest(before)==digest(M.state(shadow));event=lambda who:(lambda stage,k:B.append('optimizer_calls.jsonl',[who,stage,k]))
  apply=lambda target,who:M.fit_pg(target,groups,p,weights,lambda:(rt.guard(),gpu_guard(p)),event(who)) if kind=='pg' else M.fit_recovery(target,corrections,anchors,p,lambda:(rt.guard(),gpu_guard(p)),event(who))
  u=apply(obj,'live');v=apply(shadow,'shadow');assert u==v and digest(M.state(obj))==digest(M.state(shadow));obj[0].cpu();model=M.export(obj,p,folder/'proposal.pt.gz');save(folder/'LEARNING_COMPLETE.json',dict(update=u,proposal=model,epoch=time.time()));return u,model
 try:
  rt.pool=B.N.Pool(dict(p,out=str(folder),phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
  if p['kind']=='collect':
   training=[]
   for index,job in enumerate(p['training_jobs']):
    jobs=[copy.deepcopy(job) for _ in range(2)];seeds=[[p['seed'],0,index,i] for i in range(2)];vals=request(jobs,p['student_initial'],'FRESH_TRAIN4',seeds=seeds,full=True);rows=[r for a,r in vals];request(jobs,p['student_initial'],'EXACT_TRAIN2',seeds=seeds,refs=rows);training+=rows
   result=dict(training_rows=training,old_successes=sum(safe(r) for r in training if r['job_id'] in p['before_success_ids']),new_successes=sum(safe(r) for r in training if r['job_id'] in p['new_ids']),optimizer_updates=0)
  elif p['kind'] in ['core','broad','recovery_rl']:
   if p['kind']=='recovery_rl':
    groups=[]
    for index,job in enumerate(p['core_jobs']):
     jobs=[copy.deepcopy(job) for _ in range(2)];seeds=[[p['seed'],2,index,i] for i in range(2)];vv=request(jobs,p['student_initial'],'FRESH_TRAIN4',seeds=seeds,full=True);request(jobs,p['student_initial'],'EXACT_TRAIN2',seeds=seeds,refs=[r for a,r in vv]);groups.append([a for a,r in vv])
    weights=[.25]*4
   else:groups,weights=groups_from(p)
   u,proposal=fit('pg',groups,weights);result=dict(update=u,proposal=proposal,final49=None,accepted=False,model=p['reference_model'])
   if u['applied']>0:result.update(evaluate(proposal))
   if not result['accepted']:M.load(obj,before)
   result['retained_state_sha']=digest(M.state(obj))
  elif p['kind']=='recovery':
   model=p['student_initial'];jobs=p['recovery_source_jobs'];vv=request(jobs,model,'CANDIDATE_SOURCE',full=True);source=[r for a,r in vv];request(jobs,model,'EXACT_CANDIDATE_SOURCE',refs=source);source_by={r['job_id']:(a,r) for a,r in vv};anchors=[(a,model) for a,r in vv if r['job_id'] in p['new_ids'] and safe(r)];qualified=[];qualrows=[]
   for job in jobs:
    if job['id'] not in p['recovery_old_ids']:continue
    a,row=source_by[job['id']]
    for fraction in [.2,.4]:
     first=int(p['decision_stride_native']*0+job['case']['step_cap']*fraction/p['decision_stride_native']);qjob=dict(job,source_path=row['path'],student_takeover_first=first)
     if not (0<first<len(a['states'])-1):continue
     qr=request([qjob],p['reference_model'],'QUALIFIED_T608_RECOVERY',full=True);qrows=[r for arr,r in qr];request([qjob],p['reference_model'],'EXACT_TEACHER_RECOVERY',refs=qrows);qualrows+=qrows
     if safe(qrows[0]):qualified.append((qr[0][0],first,p['reference_model']))
   result=dict(candidate_source_rows=source,qualification_rows=qualrows,qualified_recoveries=sum(safe(r) for r in qualrows),new_success_anchors=len(anchors),update=None,proposal=None,accepted=False,final49=None,model=p['reference_model'])
   if qualified and len(anchors)==2:
    u,proposal=fit('recovery',corrections=qualified,anchors=anchors);result.update(update=u,proposal=proposal,correction_specs=[dict(path=r['path'],start=r['request']['job']['student_takeover_first'],model=p['reference_model']) for r in qualrows if safe(r)],anchor_specs=[dict(path=r['path'],model=model) for r in source if r['job_id'] in p['new_ids'] and safe(r)])
    if u['applied']>0:result.update(evaluate(proposal))
    if not result['accepted']:M.load(obj,before)
    result['retained_state_sha']=digest(M.state(obj))
  elif p['kind']=='profile':
   vals=request(p['profile_jobs'],p['student_initial'],'FROZEN_PROFILE407');rows=[r for a,r in vals];keys=[(a,r) for a,r in vals if p.get('exact_all_profile') or r['job_id'] in p['key_replay_ids']];request([r['request']['job'] for a,r in keys],p['student_initial'],'EXACT_PROFILE4',refs=[r for a,r in keys]);result=dict(model=p['student_initial'],rows=rows,success_ids=sorted(r['job_id'] for r in rows if safe(r)),no_selection=True,new_updates=0)
  else:raise ValueError(p['kind'])
 except BaseException:
  error=traceback.format_exc()
  if obj is not None and before is not None:M.load(obj,before)
  save(folder/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=3
  save(folder/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result));fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
 assert error is None,error

def audit(name):
 p=pchild(name);folder=OUT/name;setup(p,folder);verify(p);s=read(folder/'summary.json');assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists();g=read(folder/'worker_generations.json');assert all(not Path('/proc',str(pid)).exists() for row in g for pid in [row['parent'],*row['workers']]);cases=B.read_rows('cases.jsonl');replays=B.read_rows('replays.jsonl');by={r['path']:r for r in cases}
 for row in cases:
  assert sha(row['path'])==row['sha256'];a=B.R.E.N.S.arrays(row['path'])
  if row['full']:assert all(digest(v)==row['array_hashes'][k] for k,v in a.items());Native.inspect(p,[(a,row)])
  else:
   for key in ['actions','native_counts','end_flags']:assert digest(a[key])==row['array_hashes'][key]
 for row in replays:assert row['exact'] and row['array_hashes']==by[row['reference_path']]['array_hashes'] and row['integration_sha256']==by[row['reference_path']]['integration_sha256']
 r=s['result'];updates=0
 if r.get('update') is not None:
  gpu_guard(dict(p,absolute_learning_stop=p['absolute_all_stop']));M.configure_gpu(p);obj=M.make(p,p['raw_sigma']);before=M.state(obj);events=[];guard=lambda:gpu_guard(dict(p,absolute_learning_stop=p['absolute_all_stop']));emit=lambda kind,k:events.append(['live',kind,k])
  if p['kind']=='recovery':
   corrections=[(B.R.E.N.S.arrays(x['path']),x['start'],x['model']) for x in r['correction_specs']];anchors=[(B.R.E.N.S.arrays(x['path']),x['model']) for x in r['anchor_specs']];u=M.fit_recovery(obj,corrections,anchors,p,guard,emit)
  else:
   if p['kind']=='recovery_rl':groups=[[a for a,row in load_values([r for r in cases if r['job_id']==job['id'] and r['full']])] for job in p['core_jobs']];weights=[.25]*4
   else:groups,weights=groups_from(p)
   u=M.fit_pg(obj,groups,p,weights,guard,emit)
  assert u==r['update'];updates=u['attempted_optimizer'];obj[0].cpu();assert digest(M.state(obj))==r['proposal']['joint_hash'] and obj[0].phash()==r['proposal']['policy_hash'];calls=B.read_rows('optimizer_calls.jsonl');assert [e for e in calls if e[0]=='live']==events and [['live',*e[1:]] for e in calls if e[0]=='shadow']==events
  if r['final49'] is not None:assert metrics(p,r['evaluation_rows'])==r['final49'] and r['accepted']==accepted(r['final49'])
  if not r['accepted']:M.load(obj,before)
  assert digest(M.state(obj))==r['retained_state_sha']
 if p['kind']=='recovery':
  assert r['qualified_recoveries']==sum(safe(row) for row in r['qualification_rows'])
  assert all(row['forced_prefix_recovery_diagnostic'] and not row['deployment_from_original_start'] for row in r['qualification_rows'])
 if p['kind']=='collect':
  assert cases==r['training_rows'] and len(cases)==72 and len(replays)==72
  assert r['old_successes']==sum(safe(row) for row in cases if row['job_id'] in p['before_success_ids']) and r['new_successes']==sum(safe(row) for row in cases if row['job_id'] in p['new_ids'])
 if p['kind']=='profile':
  assert cases==r['rows'] and r['success_ids']==sorted(row['job_id'] for row in cases if safe(row))
  assert sorted(row['job_id'] for row in cases)==sorted(j['id'] for j in p['profile_jobs'])
 assert sum(row['physics'] for row in cases+replays)==s['physics']<=p['max_physics'] and len(cases)+len(replays)+6==s['commands']<=p['max_commands'];save(folder/'audit.json',dict(state='PASS',independent_historical_Actor=updates,new_native=0,full_history_density_reward_model_and_rollback_reconstructed=True,all_saved_bytes_SHA=True,all_exact_replays_counted=True))
if __name__=='__main__':globals()[sys.argv[1]](sys.argv[2])
