"""Bounded two-arm experiment. No network dependency during science."""
from pathlib import Path
import sys,os,time,fcntl,traceback,shutil
import t620_learning as L
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t620_recurrent_prior_td3';DOC=ROOT/'stage/docs/day_20261117/t620'
sys.path.insert(0,str(ROOT/'stage/experiments/t608_current_policy_memory'));import t608_runtime as R
import t620_native as N
read,save,sha,digest=L.read,L.save,L.sha,L.digest

def used(p):return sum(f.stat().st_size for f in Path(p).rglob('*') if f.is_file())
def verify(p,full=False):
 root=read(OUT/'protocol.json');assert root['authorization_approved'] and sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
 assert time.time()<root['absolute_all_stop'] and used(OUT)<=root['max_bytes']
 assert used(ROOT/'stage/runs')-used(OUT)+root['max_bytes']+131072<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not L.torch.cuda.is_initialized() and set(os.sched_getaffinity(0))<={14,15}
 for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f
def plan(arm):
 root=read(OUT/'protocol.json');child=read(OUT/arm/'protocol.json');assert child['_root_sha256']==sha(OUT/'protocol.json')
 p=read(root['actor_configuration']);p.update(root);p.update(child);p.update(out=str(OUT/arm),child_bytes=root['max_bytes']);return p
def setup(folder):
 R.NIGHT=OUT;R.HERE=HERE;R.DOC=DOC;R.setup(folder,'round');R.verify=verify;R.Base.verify=verify;R.P.frozen=L.frozen
def safe(row):return R.safe(row)

def main(arm):
 p=plan(arm);folder=Path(p['out']);setup(folder);verify(p,True);R.E.N.offline();R.E.L.install_stop_handlers();rt=R.Base.Runtime(p);state='FAILED';error=None;result=None;obj=None;before=None
 try:
  obj=L.make(p);shadow=L.make(p);before=L.state(obj);assert obj['actor'].phash()==p['student_initial']['policy_hash'];assert digest(before)==digest(L.state(shadow))
  rt.pool=N.Pool(dict(p,native_stop_epoch=rt.deadline,phase='student'),rt.counts,rt.guard)
  jobs=p['training_jobs_by_arm'][arm];vv=rt.request(jobs,p['student_initial'],'NEW_ONLINE3',seeds=p['training_seeds'],full=True);N.inspect(p,vv);online=[r for a,r in vv]
  assert all(Path(r['path']).stat().st_size<=p['fresh_file_bytes'] for r in online)
  rt.request(jobs,p['student_initial'],'EXACT_NEW_ONLINE3',seeds=p['training_seeds'],reference=online)
  data=L.dataset(p,online);event=lambda who:(lambda which,stage,k:R.Base.append('optimizer_calls.jsonl',[who,which,stage,k]))
  rt.role='LEARN_LIVE_AND_SHADOW';u=L.fit(obj,data,p,lambda:rt.guard(True),event('live'));v=L.fit(shadow,data,p,lambda:rt.guard(True),event('shadow'));assert u==v and digest(L.state(obj))==digest(L.state(shadow))
  mature=L.state(obj);mh=digest(mature);L.load(obj,before);assert digest(L.state(obj))==digest(before);L.load(obj,mature);assert digest(L.state(obj))==mh
  model=L.export((obj['actor'],obj['actor_opt']),p,folder/'proposal.pt.gz');joint=L.save_joint(obj,p,folder/'learner_state.pt.gz');save(folder/'LEARNING_COMPLETE.json',dict(update=u,proposal=model,joint=joint,mature_rollback_exact=True,epoch=time.time()))
  result=dict(arm=arm,update=u,proposal=model,joint=joint,online_rows=online,online_accounting=L.online_accounting(online),accepted=False,model=p['student_initial'],check=None,new_navigation_gain=False)
  critical=[]
  for index,job in enumerate(p['critical_jobs']):
   vv=rt.request([job],model,'CRITICAL_ORIGINAL_START',full=index==0);N.inspect(p,vv);critical.append(vv[0][1])
   if index==0:assert Path(critical[-1]['path']).stat().st_size<=p['fresh_file_bytes']
  result['critical']=critical;save(folder/'CRITICAL_PROGRESS.json',dict(rows=critical));rt.request(p['critical_jobs'],model,'EXACT_CRITICAL',reference=critical)
  hh=[]
  if all(safe(r) for r in critical):
   jobs=[r['request']['job'] for r in p['recovery_rows']];vv=rt.request(jobs,model,'RECOVERY3');N.inspect(p,vv);hh=[r for a,r in vv];rt.request(jobs,model,'EXACT_RECOVERY3',reference=hh)
   result['recoveries']=hh;save(folder/'RECOVERY_PROGRESS.json',dict(rows=hh))
   if all(safe(r) for r in hh):
    jobs=[j for j in p['evaluation_jobs'] if j['id'] not in p['critical_ids']];vv=rt.request(jobs,model,'REMAINING47');N.inspect(p,vv);rows=critical+[r for a,r in vv];by={r['job_id']:r for r in rows};success=sorted(r['job_id'] for r in rows if safe(r));lost=sorted(set(p['before_success_ids'])-set(success));gain=sorted(set(success)-set(p['before_success_ids']))
    result.update(check=dict(success=len(success),valid=len(rows),success_ids=success,lost_prior=lost,gained_prior=gain,field_success=sum(safe(r) for r in rows if r['job_id'].startswith('t502_field_'))),accepted=not lost and bool(gain),new_navigation_gain=not lost and bool(gain))
    keys=[j for j in p['evaluation_jobs'] if j['id'] in p['key_replay_ids']];rt.request(keys,model,'EXACT_KEY4',reference=[by[j['id']] for j in keys])
  if not result['accepted']:L.load(obj,before);L.load(shadow,before)
  result.update(model=model if result['accepted'] else p['student_initial'],retained_state_sha=digest(L.state(obj)),full407=None,fresh16=None,reload78=None,globally_promoted=False)
  state='COMPLETE'
 except BaseException:
  error=traceback.format_exc()
  if obj is not None and before is not None:L.load(obj,before)
  save(folder/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=2
  save(folder/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
 assert state=='COMPLETE',error

def audit(arm):
 p=plan(arm);folder=Path(p['out']);setup(folder);verify(p,True);R.G.stopped(folder);assert not (folder/'audit.json').exists();s=read(folder/'summary.json');assert s['state']=='COMPLETE';result=s['result'];rows=R.G.rows(folder);replays=R.G.rows(folder,'replays.jsonl');by={r['path']:r for r in rows}
 for r in rows:
  assert sha(r['path'])==r['sha256'];a=R.E.N.S.arrays(r['path'])
  if r['full']:assert all(digest(v)==r['array_hashes'][k] for k,v in a.items());N.inspect(p,[(a,r)])
 for r in replays:assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256']
 assert L.online_accounting(result['online_rows'])==result['online_accounting']
 obj=L.make(p);before=L.state(obj);events=[];u=L.fit(obj,L.dataset(p,result['online_rows']),p,lambda:None,lambda which,stage,k:events.append(['live',which,stage,k]));assert u==result['update']
 assert digest(L.state(obj))==result['joint']['state_sha'] and sha(result['joint']['path'])==result['joint']['sha256'];assert obj['actor'].phash()==result['proposal']['policy_hash']
 calls=R.G.rows(folder,'optimizer_calls.jsonl');assert [r for r in calls if r[0]=='live']==events and [['live',*r[1:]] for r in calls if r[0]=='shadow']==events
 if result['check'] is not None:
  original=[r for r in rows if r['role'] in ['CRITICAL_ORIGINAL_START','REMAINING47']];ids=sorted(r['job_id'] for r in original if safe(r));lost=sorted(set(p['before_success_ids'])-set(ids));gained=sorted(set(ids)-set(p['before_success_ids']));assert len(original)==49 and ids==result['check']['success_ids'];assert result['accepted']==(not lost and bool(gained))
 else:assert not result['accepted']
 if not result['accepted']:L.load(obj,before)
 assert digest(L.state(obj))==result['retained_state_sha'];assert s['physics']==sum(r['physics'] for r in rows+replays)<=p['max_physics'];assert s['commands']==len(rows)+len(replays)+4<=p['max_commands'];assert read(folder/'LEARNING_COMPLETE.json')['epoch']<p['absolute_learning_stop']
 save(folder/'audit.json',dict(state='PASS',historical_actor_updates=u['actor_steps'],historical_critic_updates=u['critic_steps'],actor_critic_and_both_Adam_and_targets_reconstructed=True,new_native=0,physics=s['physics'],commands=s['commands'],all_saved_bytes_SHA=True,full_native_training_histories_checked=True,compact_evaluations_not_full_offline_reconstruction=True))

if __name__=='__main__':globals()[sys.argv[1]](sys.argv[2])
