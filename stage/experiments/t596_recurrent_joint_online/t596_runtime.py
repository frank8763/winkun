"""One bounded fresh closed-loop recurrent update, then all49+3handoff acceptance."""
from pathlib import Path
import sys,time,os,fcntl,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t596_recurrent_joint_online';DOC=ROOT/'stage/docs/day_20261116/t596'
import t596_learning as L
sys.path.insert(0,str(ROOT/'stage/experiments/t591_joint_representation_pair'));import t591_runtime as R
Base,F,G,E,P,torch,np=R.Base,R.F,R.G,R.E,R.P,L.torch,L.np
read,save,sha,digest=L.read,L.save,L.sha,L.digest
def setup(folder):
 R.OUT=OUT;R.HERE=HERE;R.DOC=DOC;R.setup(folder);Base.verify=verify;P.frozen=L.frozen
def verify(p,full=False):return R.verify(p,full)
def worker(pipe,index,p,counts):
 G.T.Session=R.Session;G.B.native_setup=F.field_setup;P.frozen=L.frozen;E.payload=R.payload;G.cpu_worker(pipe,index,p,counts)
class Pool(R.Pool):
 def __init__(self,p,counts,guard):
  self.p=p;self.guard,self.children,self.pipes=guard,[],[];ctx=G.R.M.mp.get_context('spawn')
  try:
   for i in range(2):
    a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
   self.ready=[self.receive(i) for i in range(2)];assert all(x['state']=='READY' for x in self.ready)
   f=Path(p['out'])/'worker_generations.json';save(f,(read(f) if f.exists() else [])+[dict(parent=os.getpid(),workers=[x.pid for x in self.children],epoch=time.time(),CPU=[14,15],GPU=False)])
  except BaseException:self.close();raise
def inspect(p,vv):return R.inspect(p,vv)
def main():
 p=dict(read(OUT/'protocol.json'),_root_sha256=sha(OUT/'protocol.json'));setup(OUT);verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=Base.Runtime(p);status='FAILED';error=None;result=None;obj=shadow=before=None
 try:
  rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
  vv=rt.request(p['handoff_jobs'],p['base_model'],'INITIAL_T550_HANDOFF');inspect(p,vv);baseline=[r for a,r in vv];rt.request(p['handoff_jobs'],p['base_model'],'EXACT_INITIAL_HANDOFF',reference=baseline)
  vv=rt.request(p['training_jobs'],p['base_model'],'FRESH_OLD_AND_HARD_TASKS',seeds=p['training_seeds'],full=True);inspect(p,vv);training=[r for a,r in vv];rt.request(p['training_jobs'],p['base_model'],'EXACT_FRESH_TRAINING',seeds=p['training_seeds'],full=True,reference=training);rt.pool.close();rt.pool=None;rt.commands+=2
  data=L.references(p);arrays=[E.N.S.arrays(r['path']) for r in training];obj=L.make(p);shadow=L.make(p);before=L.state(obj);rt.role='RECURRENT_JOINT_LEARNING';event=lambda who:(lambda t,k:Base.append('optimizer_calls.jsonl',[who,t,k]))
  u=L.fit(obj,arrays,data,p,lambda:rt.guard(True),event('live'));v=L.fit(shadow,arrays,data,p,lambda:rt.guard(True),event('shadow'));assert u==v and digest(L.state(obj))==digest(L.state(shadow));model=L.export(obj,p,OUT/'proposal.pt.gz');assert L.frozen(p,model['path']).phash()==obj[0].phash();save(OUT/'LEARNING_COMPLETE.json',dict(epoch=time.time(),live_steps=u['Actor_Adam'],learning_stop=rt.learnstop))
  rt.commands+=2;rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
  vv=rt.request(p['evaluation_jobs'],model,'PURE_STUDENT_ORIGINAL49');inspect(p,vv);rows=[r for a,r in vv];check=R.assess(p,rows);keys=[j for j in p['evaluation_jobs'] if j['id'] in p['key_replay_ids']];by={r['job_id']:r for r in rows};rt.request(keys,model,'EXACT_KEY4_COMPACT',reference=[by[j['id']] for j in keys])
  vv=rt.request(p['handoff_jobs'],model,'PURE_STUDENT_RECOVERY3');inspect(p,vv);hh=[r for a,r in vv];rt.request(p['handoff_jobs'],model,'EXACT_RECOVERY3_COMPACT',reference=hh)
  field_success={r['job_id'] for r in rows if r['job_id'].startswith('t502_field_') and R.safe(r)};lost=set(p['protected_field15'])-field_success;gained=field_success-set(p['protected_field15']);all3=all(R.safe(r) for r in hh)
  accepted=not check['lost_initial'] and not lost and all3 and bool(gained)
  result=dict(update=u,training=G.R.M.tally(training),training_rows=training,baseline_handoff=baseline,proposal=model,check=check,handoff=hh,field_success_ids=sorted(field_success),lost_field15=sorted(lost),gained_field_ids=sorted(gained),all3_recovery_safe=all3,accepted=accepted,model=model if accepted else p['base_model'],full407=None,fresh16=None,reload78=None,globally_promoted=False)
  if not accepted:L.load(obj,before);L.load(shadow,before)
  result['retained_joint_hash']=digest(L.state(obj));status='COMPLETE'
 except BaseException:
  error=traceback.format_exc()
  if before is not None:L.load(obj,before);L.load(shadow,before)
  save(OUT/'failure.json',dict(error=error,whole_ActorAdam_rollback=before is not None,no_restart=True,raw_preserved=True))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=2
  save(OUT/'summary.json',dict(state=status,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
 assert status=='COMPLETE',error
def audit():
 p=dict(read(OUT/'protocol.json'),_root_sha256=sha(OUT/'protocol.json'));setup(OUT);verify(p,True);G.stopped(OUT);assert not (OUT/'audit.json').exists();s=read(OUT/'summary.json');assert s['state']=='COMPLETE';r=s['result'];rows=G.rows(OUT);reps=G.rows(OUT,'replays.jsonl');assert len(rows)==63 and len(reps)==18
 for x in rows:
  assert sha(x['path'])==x['sha256'];a=E.N.S.arrays(x['path']);assert int(a['native_counts'].sum())==x['physics'] and all(digest(v)==x['array_hashes'][k] for k,v in a.items())
  if x['full']:inspect(p,[(a,x)])
 by={x['path']:x for x in rows}
 for x in reps:assert x['exact'] and x['array_hashes']==by[x['reference_path']]['array_hashes'] and x['integration_sha256']==by[x['reference_path']]['integration_sha256']
 assert rows[3:11]==r['training_rows'];obj=L.make(p);before=L.state(obj);events=[];u=L.fit(obj,[E.N.S.arrays(x['path']) for x in r['training_rows']],L.references(p),p,lambda:None,lambda t,k:events.append(['live',t,k]))
 assert u==r['update'] and digest(L.state(obj))==r['proposal']['joint_hash'] and sha(r['proposal']['path'])==r['proposal']['sha256'];ev=G.rows(OUT,'optimizer_calls.jsonl');assert [x for x in ev if x[0]=='live']==events and [['live',t,k] for w,t,k in ev if w=='shadow']==events
 assert R.assess(p,rows[11:60])==r['check'] and rows[60:]==r['handoff'];field={x['job_id'] for x in rows[11:60] if x['job_id'].startswith('t502_') and R.safe(x)};assert sorted(field-set(p['protected_field15']))==r['gained_field_ids'] and sorted(set(p['protected_field15'])-field)==r['lost_field15']
 assert r['accepted']==(not r['check']['lost_initial'] and not r['lost_field15'] and all(R.safe(x) for x in r['handoff']) and bool(r['gained_field_ids']))
 if not r['accepted']:L.load(obj,before)
 assert digest(L.state(obj))==r['retained_joint_hash'];assert s['commands']==89 and s['physics']==sum(x['physics'] for x in rows+reps)<=p['max_physics'];assert read(OUT/'LEARNING_COMPLETE.json')['epoch']<p['absolute_learning_stop']
 save(OUT/'audit.json',dict(state='PASS',new_native=0,discarded_historical_Adam=u['Actor_Adam'],all_live_shadow_ActorAdam_reconstruct_and_rollback_exact=True,all8_training_full_history_and_native_replays=True,other55_compact_live_checked_not_full_pose_reconstruction=True,all18_exact_replays=True,physics=s['physics'],commands=s['commands']));print('T596_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
