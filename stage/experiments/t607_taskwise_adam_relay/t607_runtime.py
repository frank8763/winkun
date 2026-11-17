"""Offline child execution with one shared immutable root and thin child plans."""
from pathlib import Path
import sys,os,time,shutil,fcntl,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
NIGHT=ROOT/'stage/runs/t607_taskwise_adam_relay';DOC=ROOT/'stage/docs/overnight_20261116/t607'
import t607_learning as L
sys.path.insert(0,str(ROOT/'stage/experiments/t598_independent_recurrent_recovery'))
import t598_runtime as Legacy
Base,F,G,E,P,torch,np=Legacy.Base,Legacy.F,Legacy.G,Legacy.E,Legacy.P,L.torch,L.np
read,save,sha,digest=L.read,L.save,L.sha,L.digest
ACTIVE=NIGHT;ORIGINAL_APPEND=Base.append

def used(folder):return sum(f.stat().st_size for f in Path(folder).rglob('*') if f.is_file())
def plan(child):
 r=read(NIGHT/'protocol.json');c=read(NIGHT/child/'protocol.json');assert c['_root_sha256']==sha(NIGHT/'protocol.json')
 p=read(r['actor_configuration']);p.update(r);p.update(c);p['out']=str(NIGHT/child);return p

def verify(p,full=False):
 r=read(NIGHT/'protocol.json');assert r['authorization_approved'] and sha(HERE/'protocol.json')==sha(NIGHT/'protocol.json')
 assert p.get('_root_sha256',sha(NIGHT/'protocol.json'))==sha(NIGHT/'protocol.json')
 assert time.time()<r['absolute_all_stop'] and used(NIGHT)<=r['max_bytes']
 assert used(ROOT/'stage/runs')-used(NIGHT)+r['max_bytes']+131072<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized() and set(os.sched_getaffinity(0))<={14,15}
 for f,h in (r['inputs'] if full else r['protected_files']).items():assert sha(f)==h,f

def slim(r):
 keys=['complete','success','collision','timeout','administrative_censor','physics','physics_steps','remaining_m','job_id','policy_hash','role','seed','path','sha256','array_digest','integration_sha256','field_detail']
 return {k:r[k] for k in keys if k in r}

def setup(folder,kind):
 global ACTIVE
 ACTIVE=folder;Legacy.OUT=folder;Legacy.HERE=HERE;Legacy.DOC=DOC;Legacy.R.verify=verify;Legacy.setup(folder);Base.verify=verify;P.frozen=L.frozen
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 def append(name,value):
  if kind=='stress' and name=='cases.jsonl':value=slim(value)
  return ORIGINAL_APPEND(name,value)
 Base.append=append

def safe(r):return Legacy.R.safe(r)
def keep_check(p,rows,hh):
 check=Legacy.R.assess(p,rows);ids=set(check['success_ids']);prior=set(p['before_success_ids'])
 return dict(check=check,lost_prior=sorted(prior-ids),gained_prior=sorted(ids-prior),all3_safe=all(safe(r) for r in hh),
  retained=prior<=ids and all(safe(r) for r in hh),original_T55039_restored=not check['lost_initial'])

def final_counts(rows):
 def part(values):return dict(native=G.R.M.tally(values),qualified_safe=sum(safe(r) for r in values),valid=sum(bool(r['complete']) and not r.get('administrative_censor',False) for r in values))
 return dict(kind='final',old275=part(rows[:275]),six54=part(rows[275:329]),eight54=part(rows[329:383]),field24=part(rows[383:]),development407_all_safe=all(safe(r) for r in rows),fresh16=None,reload78=None,globally_promoted=False)


def main(child):
 p=plan(child);folder=Path(p['out']);setup(folder,p['kind']);verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=Base.Runtime(p)
 state='FAILED';error=None;result=None;obj=shadow=before=None;pool_started=False
 try:
  if p['kind']=='round':
   data=L.dataset(p);obj=L.make(p);shadow=L.make(p);before=L.state(obj);training=[];arrays=None
   if p['arm']=='rl':
    pool_started=True;rt.pool=Legacy.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard);pool_started=True
    vv=rt.request([p['training_job']]*4,p['student_initial'],'FRESH_CURRENT_POLICY4',seeds=p['training_seeds'],full=True);Legacy.inspect(p,vv);arrays=[a for a,r in vv];training=[r for a,r in vv]
    rt.request([p['training_job']]*4,p['student_initial'],'EXACT_FRESH_CURRENT4',seeds=p['training_seeds'],reference=training)
   ev=lambda who:(lambda s,k:Base.append('optimizer_calls.jsonl',[who,s,k]))
   rt.role='LEARN_'+p['arm'];u=L.fit(obj,data,p,lambda:rt.guard(True),ev('live'),arrays);v=L.fit(shadow,data,p,lambda:rt.guard(True),ev('shadow'),arrays)
   assert u==v and digest(L.state(obj))==digest(L.state(shadow));model=L.export(obj,p,folder/'proposal.pt.gz');assert L.frozen(p,model['path']).phash()==obj[0].phash()
   save(folder/'LEARNING_COMPLETE.json',dict(epoch=time.time(),update=u,proposal=model))
   result=dict(kind='round',arm=p['arm'],update=u,proposal=model,training_rows=training,handoff=[],check=None,accepted=False,model=p['student_initial'],full407=None,globally_promoted=False)
   if u['applied_parameter_steps']:
    if not rt.pool:pool_started=True;rt.pool=Legacy.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard);pool_started=True
    hh=[]
    for k,row in enumerate(p['recovery_rows']):
     job=row['request']['job'];vv=rt.request([job],model,'RECOVERY_'+str(k));Legacy.inspect(p,vv);hh.append(vv[0][1]);save(folder/'RECOVERY_PROGRESS.json',dict(rows=hh))
     rt.request([job],model,'EXACT_RECOVERY_'+str(k),reference=[hh[-1]])
    result['handoff']=hh;save(folder/'AFTER_RECOVERY.json',result)
    if all(safe(r) for r in hh):
     sentinel=[j for j in p['evaluation_jobs'] if j['id'] in p['sentinel_ids']];vv=rt.request(sentinel,model,'SENTINELS15');Legacy.inspect(p,vv);ss=[r for a,r in vv];result['sentinels']=ss;save(folder/'AFTER_SENTINELS.json',result)
     if all(safe(r) for r in ss if r['job_id'] in p['before_success_ids']):
      rest=[j for j in p['evaluation_jobs'] if j['id'] not in p['sentinel_ids']];vv=rt.request(rest,model,'REMAINING34');Legacy.inspect(p,vv);by={r['job_id']:r for r in ss+[r for a,r in vv]};ordered=[by[j['id']] for j in p['evaluation_jobs']]
      result.update(keep_check(p,ordered,hh));save(folder/'AFTER49.json',result)
      keys=[j for j in p['evaluation_jobs'] if j['id'] in p['key_replay_ids']];rt.request(keys,model,'EXACT_KEY4',reference=[by[j['id']] for j in keys])
      result['accepted']=result['retained']
   if not result['accepted']:L.load(obj,before);L.load(shadow,before)
   result.update(model=model if result['accepted'] else p['student_initial'],retained_joint_hash=digest(L.state(obj)),new_navigation_gain=bool(result.get('gained_prior')) if result['accepted'] else False)
  else:
   pool_started=True;rt.pool=Legacy.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard);pool_started=True
   if p['kind']=='final':
    vv=rt.request(p['final_jobs'],p['student_initial'],'FINAL_SAME_POLICY407');Legacy.inspect(p,vv);rows=[r for a,r in vv]
    result=final_counts(rows)
   elif p['kind']=='stress':
    runs=[]
    for mi,m in enumerate(p['stress_models']):
     vv=rt.request(p['stress_jobs'],m,'STRESS_MODEL_'+str(mi),seeds=p['stress_seeds']);Legacy.inspect(p,vv);runs.append(vv)
    pairs=[]
    for i,(aa,bb) in enumerate(zip(*runs)):
     a,ra=aa;b,rb=bb;n=min(len(a['innovations']),len(b['innovations']));assert np.array_equal(a['innovations'][:n],b['innovations'][:n])
     pairs.append(dict(job_id=p['stress_jobs'][i]['id'],seed=p['stress_seeds'][i],model0=slim(ra),model1=slim(rb),common_innovation_prefix_sha=digest(a['innovations'][:n]),common_decisions=n))
    result=dict(kind='stress',pairs=pairs,selection_or_training_from_stress=False,new_Adam=0,deterministic_deployment_gate=False,independent_native_replay=False,compact_live_history_checked_only=True)
   elif p['kind']=='engineering':
    vv=rt.request([p['probe_job']]*4,p['student_initial'],'ENGINEERING_CURRENT_IID4',seeds=p['training_seeds'],full=True);Legacy.inspect(p,vv);rr=[r for a,r in vv];rt.request([p['probe_job']]*4,p['student_initial'],'EXACT_ENGINEERING4',seeds=p['training_seeds'],reference=rr)
    _,diag=L.Credit.peer_batch((L.frozen(p,p['student_initial']['path']),None),[a for a,r in vv],p)
    result=dict(kind='engineering',density=diag,new_Adam=0,short_probe_not_navigation_success=True)
   else:raise ValueError(p['kind'])
  state='COMPLETE'
 except BaseException:
  error=traceback.format_exc()
  if before is not None:L.load(obj,before);L.load(shadow,before)
  save(folder/'failure.json',dict(error=error,rollback=before is not None,raw_preserved=True,no_restart=True))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=2
  elif pool_started:rt.commands+=2
  if not pool_started:assert sum(rt.counts)==0 and rt.commands==2;rt.commands=0
  save(folder/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result))
  fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
 assert state=='COMPLETE',error

def audit(child):
 p=plan(child);folder=Path(p['out']);setup(folder,p['kind']);verify(p,True);G.stopped(folder);assert not (folder/'audit.json').exists();s=read(folder/'summary.json');assert s['state']=='COMPLETE';result=s['result'];rows=G.rows(folder);reps=G.rows(folder,'replays.jsonl');by={r['path']:r for r in rows}
 for r in rows:
  assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert int(a['native_counts'].sum())==r['physics']
  if p['kind']!='stress':assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
  if r.get('full'):Legacy.inspect(p,[(a,r)])
 for r in reps:assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256']
 historical=0
 if p['kind']=='round':
  obj=L.make(p);before=L.state(obj);events=[];training=result['training_rows'];arrays=[E.N.S.arrays(r['path']) for r in training] if training else None
  u=L.fit(obj,L.dataset(p),p,lambda:None,lambda t,k:events.append(['live',t,k]),arrays);historical=u['attempted_Adam']
  assert u==result['update'] and digest(L.state(obj))==result['proposal']['joint_hash'] and sha(result['proposal']['path'])==result['proposal']['sha256']
  calls=G.rows(folder,'optimizer_calls.jsonl');assert [v for v in calls if v[0]=='live']==events and [['live',t,k] for w,t,k in calls if w=='shadow']==events
  if u['applied_parameter_steps']:
   hh=[r for r in rows if r['role'].startswith('RECOVERY_')];assert hh==result['handoff'] and len(hh)==3
  if result['check'] is not None:
   original=[r for r in rows if r['role'] in ['SENTINELS15','REMAINING34']];ordered=[next(r for r in original if r['job_id']==j['id']) for j in p['evaluation_jobs']]
   reconstructed=keep_check(p,ordered,result['handoff']);assert all(result[k]==v for k,v in reconstructed.items());assert result['accepted']==reconstructed['retained']
  else:assert not result['accepted']
  if not result['accepted']:L.load(obj,before)
  assert digest(L.state(obj))==result['retained_joint_hash']
  assert read(folder/'LEARNING_COMPLETE.json')['epoch']<p['absolute_learning_stop']
 elif p['kind']=='engineering':
  _,diag=L.Credit.peer_batch((L.frozen(p,p['student_initial']['path']),None),[E.N.S.arrays(r['path']) for r in rows],p);assert diag==result['density'] and len(rows)==len(reps)==4
 elif p['kind']=='final':assert len(rows)==407 and len(reps)==0 and final_counts(rows)==result
 elif p['kind']=='stress':
  assert len(rows)==2*len(p['stress_jobs']) and not reps
  assert all(z['common_decisions']>0 for z in result['pairs'])
  for pair in result['pairs']:
   for k in ['model0','model1']:assert by[pair[k]['path']]==pair[k]
 assert s['physics']==sum(r['physics'] for r in rows+reps)<=p['max_physics']
 assert s['commands']==(len(rows)+len(reps)+4 if rows else 0)<=p['max_commands']
 save(folder/'audit.json',dict(state='PASS',new_native=0,historical_discarded_Adam=historical,physics=s['physics'],commands=s['commands'],all_saved_bytes_SHA=True,available_full_training_history_checked=True,compact_not_full_offline_trajectory_reconstruction=True,stress_common_innovations_live_checked_only=p['kind']=='stress'))

if __name__=='__main__':globals()[sys.argv[1]](sys.argv[2])
