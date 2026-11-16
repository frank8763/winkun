"""Finite qualified recovery learning with paired autonomous handoff controls."""
from pathlib import Path
import fcntl
import os
import shutil
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
import t591_learning as L
OUT=ROOT/'stage/runs/t591_joint_representation_pair';DOC=ROOT/'stage/docs/day_20261116/t591'
sys.path.insert(0,str(ROOT/'stage/experiments/t577_offline_recovery_night'))
import t577_runtime as R
read,save,sha,digest,torch,np=L.read,L.save,L.sha,L.digest,L.torch,L.np
Base,F,G,E,P=R.Base,R.F,R.G,R.E,R.P


def used(path):return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())


def verify(p,full=False):
    root=read(OUT/'protocol.json');assert root['authorization_approved'] is True;assert p['_root_sha256']==sha(OUT/'protocol.json')
    assert time.time()<root['absolute_all_stop'] and used(OUT)<=root['max_bytes']
    assert used(ROOT/'stage/runs')-used(OUT)+root['max_bytes']+131072<2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    assert set(os.sched_getaffinity(0))<={14,15}
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f


def setup(folder):
    global ACTIVE
    ACTIVE=folder
    R.HERE=HERE;R.verify=verify;R.L.frozen=L.frozen;R.setup(folder);Base.verify=verify;P.frozen=L.frozen


def worker(pipe,index,p,counts):
    R.L.frozen=L.frozen;R.worker(pipe,index,p,counts)


class Pool(R.Pool):
    def __init__(self,p,counts,guard):
        self.p=p;self.guard,self.children,self.pipes=guard,[],[];ctx=G.R.M.mp.get_context('spawn')
        try:
            for i in range(2):
                a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
            self.ready=[self.receive(i) for i in range(2)];assert all(x['state']=='READY' for x in self.ready)
            save(Path(p['out'])/'worker_generations.json',[dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time(),CPU=[14,15],GPU=False)])
        except BaseException:self.close();raise





sys.path.insert(0,str(ROOT/'stage/experiments/t581_preserved_t550_rl'));import t581_runtime as Previous
assess=Previous.assess
NORMAL=R.NORMAL
class Session(F.FieldSession):
 def __init__(self,job,*args,**kw):
  super().__init__(job,*args,**kw);self.first=int(job.get('student_takeover_first',-1));self.source=E.N.S.arrays(job['source_path']) if self.first>=0 else None;self.prefix_actions=[]
 def step(self,forced=None,owner=0,actor_active=True):
  assert forced is None and owner==0
  if self.first<0:return super().step()
  k=len(self.rows);self.q.sample=False;use=k<self.first;u=self.source['actions'][k] if use else np.zeros(2,np.float32);self.prefix_actions.append(u.copy())
  return super().step(u if use else None,owner=int(use),actor_active=True)
def payload(s,tape=None):
 a=NORMAL(s,tape)
 if s.first>=0:a.update(source_prefix_actions=np.asarray(s.prefix_actions[:len(a['states'])],np.float32),student_takeover_first=np.asarray(s.first,np.int64))
 return a
def worker(pipe,index,p,counts):
 G.T.Session=Session;G.B.native_setup=F.field_setup;P.frozen=L.frozen;E.payload=payload;G.cpu_worker(pipe,index,p,counts)
def handoff_audit(p,r,a):
 job=r['request']['job'];b=E.N.S.arrays(job['source_path']);k=int(a['student_takeover_first']);assert k==job['student_takeover_first'] and len(a['states'])>k
 for name in ['states','next_states','actions','native_counts','poses','next_poses','history_poses','history_goals','history_times']:assert np.array_equal(a[name][:k],b[name][:k]),name
 for name in ['states','poses','history_poses','history_goals','history_times']:assert np.array_equal(a[name][k],b[name][k]),name
 assert np.array_equal(a['initial_native_integration'],b['initial_native_integration']) and not a['stochastic'].any()
 assert a['ownership'][:k].all() and not a['ownership'][k:].any();assert np.array_equal(a['actions'][:k],b['actions'][:k]);assert np.array_equal(a['states'][1:,117:119],a['actions'][:-1])
 proposals=np.clip(a['raw_actions'],-1,1).astype(np.float32);assert np.array_equal(a['actions'][k:],proposals[k:])
 F.M.audit_full_history(p,dict(r,teacher=False),dict(a,actions=proposals,ownership=np.zeros(len(a['states']),np.uint8)))
def inspect(p,vals):
 for a,r in vals:
  if 'student_takeover_first' in a:handoff_audit(p,r,a)
  else:F.M.audit_full_history(p,r,a)

def safe(r):return bool(r['success'] and not r['collision'] and r['field_detail']['field_qualified_success'])

def main():
 p=dict(read(OUT/'protocol.json'),_root_sha256=sha(OUT/'protocol.json'));setup(OUT);verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=Base.Runtime(p);rt.commands=0;results=[];prepared=[];baseline=None;error=None;status='FAILED'
 try:
  data=L.dataset(p);initial=L.export((L.initial(p),None),dict(p,model_bytes=p['initial_model_bytes']),OUT/'initial.pt.gz');rt.commands+=2;rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
  vv=rt.request(p['handoff_jobs'],initial,'INITIAL_SMOOTH_STUDENT_HANDOFF_COMPACT');inspect(p,vv);baseline=[r for a,r in vv];rt.request(p['handoff_jobs'],initial,'EXACT_INITIAL_HANDOFF_COMPACT',reference=baseline);rt.pool.close();rt.pool=None;rt.commands+=2;save(OUT/'baseline.json',baseline)
  # Finish both fixed learning arms before either full evaluation can consume
  # the shared learning window. Models still start independently and identically.
  for method in p['methods']:
   folder=OUT/method;folder.mkdir();pc=dict(p,method=method,out=str(folder));setup(folder);obj=L.make(pc);shadow=L.make(pc);before=L.state(obj);assert obj[0].phash()==initial['policy_hash'];event=lambda who:(lambda kind,k:Base.append('optimizer_calls.jsonl',[who,kind,k]));rt.role='LEARN_'+method
   try:
    start=time.time();u=L.fit(obj,data,pc,lambda:rt.guard(True),event('live'));live_seconds=time.time()-start;v=L.fit(shadow,data,pc,lambda:rt.guard(True),event('shadow'));assert u==v and digest(L.state(obj))==digest(L.state(shadow))
    model=L.export(obj,pc,folder/'proposal.pt.gz');assert L.frozen(pc,model['path']).phash()==obj[0].phash();save(folder/'LEARNING_COMPLETE.json',dict(epoch=time.time(),live_seconds=live_seconds,descriptive_fit=u['after']['joint_descriptive_fit']))
    result=dict(method=method,update=u,proposal=model,accepted=False,model=p['base_model'],retained_joint_hash=digest(before),handoff=None,check=None,full407=None,fresh16=None,reload78=None,globally_promoted=False)
    prepared.append((folder,pc,obj,shadow,before,result))
   except BaseException:L.load(obj,before);L.load(shadow,before);raise
  for folder,pc,obj,shadow,before,result in prepared:
   setup(folder);model=result['proposal']
   # Native handoff is mandatory for every finite candidate. Offline fit is
   # descriptive here, never a silently relaxed historical safety gate.
   rt.commands+=2;rt.pool=Pool(dict(pc,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard);vv=rt.request(p['handoff_jobs'],model,'TRAINED_STUDENT_HANDOFF_FULL',full=True);inspect(pc,vv);hh=[r for a,r in vv];rt.request(p['handoff_jobs'],model,'EXACT_TRAINED_HANDOFF',full=True,reference=hh);result['handoff']=hh
   safe_ids={r['job_id'] for r in hh if safe(r)};baseline_ids={r['job_id'] for r in baseline if safe(r)};result['new_handoff_success_ids']=sorted(safe_ids-baseline_ids)
   if safe_ids:
    vv=rt.request(p['evaluation_jobs'],model,'ORIGINAL_START49');inspect(pc,vv);by={r['job_id']:r for a,r in vv};check=assess(p,[by[j['id']] for j in p['evaluation_jobs']]);keys=[j for j in p['evaluation_jobs'] if j['id'] in p['key_replay_ids']];rt.request(keys,model,'EXACT_KEY4_COMPACT',reference=[by[j['id']] for j in keys]);result['check']=check
    result['accepted']=not check['lost_initial'] and bool(result['new_handoff_success_ids'] or check['gained'])
   rt.pool.close();rt.pool=None;rt.commands+=2
   if result['accepted']:result.update(model=model,retained_joint_hash=digest(L.state(obj)))
   else:L.load(obj,before);L.load(shadow,before)
   assert digest(L.state(obj))==result['retained_joint_hash'];save(folder/'result.json',result);results.append(result)
  status='COMPLETE'
 except BaseException:
  for folder,pc,obj,shadow,before,result in prepared:
   if not result.get('accepted'):L.load(obj,before);L.load(shadow,before)
  error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,no_restart=True,raw_preserved=True,unaccepted_ActorAdam_rollback=True))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=2
  save(OUT/'summary.json',dict(state=status,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,baseline=baseline,result=results,new_RL=0,globally_promoted=False));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
 assert status=='COMPLETE',error

def audit_rows(p,folder):
 rows=G.rows(folder);reps=G.rows(folder,'replays.jsonl');by={r['path']:r for r in rows}
 for r in rows:
  assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert int(a['native_counts'].sum())==r['physics'] and all(digest(v)==r['array_hashes'][k] for k,v in a.items())
  if r['full']:inspect(p,[(a,r)])
 for r in reps:assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256']
 for w in read(folder/'worker_generations.json'):assert all(not Path('/proc',str(pid)).exists() for pid in w['workers'])
 return rows,reps

def audit():
 p=dict(read(OUT/'protocol.json'),_root_sha256=sha(OUT/'protocol.json'));setup(OUT);verify(p,True);G.stopped(OUT);assert not (OUT/'audit.json').exists();s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and len(s['result'])==len(p['methods'])==2;data=L.dataset(p);br,bp=audit_rows(p,OUT);assert len(br)==len(bp)==3 and br==s['baseline'];total=sum(r['physics'] for r in br+bp);commands=10;history=0
 for method,result in zip(p['methods'],s['result']):
  folder=OUT/method;pc=dict(p,method=method,out=str(folder));setup(folder);obj=L.make(pc);before=L.state(obj);events=[];u=L.fit(obj,data,pc,lambda:None,lambda t,k:events.append(['live',t,k]));history+=u['steps']
  assert u==result['update'] and digest(L.state(obj))==result['proposal']['joint_hash'] and sha(result['proposal']['path'])==result['proposal']['sha256'];assert read(folder/'LEARNING_COMPLETE.json')['epoch']<p['absolute_learning_stop']
  ev=G.rows(folder,'optimizer_calls.jsonl');assert [v for v in ev if v[0]=='live']==events and [['live',t,k] for w,t,k in ev if w=='shadow']==events
  rows,reps=audit_rows(pc,folder);assert len(rows)==(52 if result['check'] is not None else 3) and len(reps)==(7 if result['check'] is not None else 3);assert rows[:3]==result['handoff']
  new={r['job_id'] for r in rows[:3] if safe(r)}-{r['job_id'] for r in br if safe(r)};assert sorted(new)==result['new_handoff_success_ids']
  if result['check'] is not None:
   rr={r['job_id']:r for r in rows[3:]};assert assess(p,[rr[j['id']] for j in p['evaluation_jobs']])==result['check'];assert result['accepted']==(not result['check']['lost_initial'] and bool(new or result['check']['gained']))
  if not result['accepted']:L.load(obj,before)
  assert digest(L.state(obj))==result['retained_joint_hash'];total+=sum(r['physics'] for r in rows+reps);commands+=len(rows)+len(reps)+4
 assert total==s['physics']<=p['max_physics'] and commands==s['commands']<=p['max_commands']
 save(OUT/'audit.json',dict(state='PASS',new_native=0,discarded_historical_Adam=history,all_ActorAdam_shadow_rollback_exact=True,baseline_compact_exact_3=True,trained_handoff_full_exact_6=True,all_current_student_hidden_history_verified=True,original49compact_not_full_pose_reconstruction=True,physics=total,commands=commands));print('T591_AUDIT_PASS',flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()

