"""Finite qualified recovery learning with paired autonomous handoff controls."""
from pathlib import Path
import fcntl
import os
import shutil
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
import t587_learning as L
OUT=ROOT/'stage/runs/t587_verified_short_correction';DOC=ROOT/'stage/docs/day_20261116/t587'
sys.path.insert(0,str(ROOT/'stage/experiments/t577_offline_recovery_night'))
import t577_runtime as R
read,save,sha,digest,torch,np=L.read,L.save,L.sha,L.digest,L.torch,L.np
Base,F,G,E,P=R.Base,R.F,R.G,R.E,R.P


def used(path):return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())


def verify(p,full=False):
    root=read(OUT/'protocol.json');assert root['authorization_approved'] is True;assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert time.time()<root['absolute_all_stop'] and used(OUT)<=root['max_bytes']
    assert used(ROOT/'stage/runs')-used(OUT)+root['max_bytes']+131072<2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    assert set(os.sched_getaffinity(0))<={14,15}
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f


def setup():
    R.HERE=HERE;R.verify=verify;R.L.frozen=L.frozen;R.setup(OUT);Base.verify=verify;P.frozen=L.frozen


def worker(pipe,index,p,counts):
    R.L.frozen=L.frozen;R.worker(pipe,index,p,counts)


class Pool(R.Pool):
    def __init__(self,p,counts,guard):
        self.p=p;self.guard,self.children,self.pipes=guard,[],[];ctx=G.R.M.mp.get_context('spawn')
        try:
            for i in range(2):
                a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
            self.ready=[self.receive(i) for i in range(2)];assert all(x['state']=='READY' for x in self.ready)
            save(OUT/'worker_generations.json',[dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time(),CPU=[14,15],GPU=False)])
        except BaseException:self.close();raise





sys.path.insert(0,str(ROOT/'stage/experiments/t581_preserved_t550_rl'));import t581_runtime as Previous
sys.path.insert(0,str(ROOT/'stage/experiments/t550_offline_obstacle_relay'));import t550_teacher as Teacher
assess=Previous.assess;NORMAL=R.NORMAL
class Session(F.FieldSession):
 def __init__(self,job,*args,**kw):
  super().__init__(job,*args,**kw);self.first=int(job.get('gap_recovery_first',-1));self.stop=self.first+int(job.get('correction_decisions',20));self.handoff=int(job.get('student_takeover_first',-1));self.source=E.N.S.arrays(job['source_path']) if self.handoff>=0 else None;self.teacher_actions=[];self.scans=[];self.prefix=[]
  self.parameters=np.asarray([self.env.speed_gain,self.env.max_wheel,self.env.wheelbase,self.env.yaw_gain,self.env.max_steer],np.float64)
 def step(self,forced=None,owner=0,actor_active=True):
  assert forced is None and owner==0
  if self.first>=0:
   native=E.N.K.R.integration(self.env).copy();scan=self.env._lidar48().copy();u=Teacher.control('GAP',self.env._pose()[:3],self.goal,scan,self.parameters);assert np.array_equal(native,E.N.K.R.integration(self.env));self.teacher_actions.append(u.copy());self.scans.append(scan);k=len(self.rows);own=self.first<=k<self.stop
   if k>=self.first:self.q.sample=False
   return super().step(u if own else None,owner=int(own),actor_active=True)
  if self.handoff>=0:
   k=len(self.rows);self.q.sample=False;use=k<self.handoff;u=self.source['actions'][k] if use else np.zeros(2,np.float32);self.prefix.append(u.copy());return super().step(u if use else None,owner=int(use),actor_active=True)
  return super().step()
def payload(s,tape=None):
 a=NORMAL(s,tape);n=len(a['states'])
 if s.first>=0:a.update(teacher_actions=np.asarray(s.teacher_actions[:n],np.float32),teacher_scans=np.asarray(s.scans[:n],np.float32),teacher_vehicle_parameters=s.parameters.copy(),teacher_first_declared=np.asarray(s.first,np.int64),teacher_stop_declared=np.asarray(s.stop,np.int64))
 if s.handoff>=0:a.update(source_prefix_actions=np.asarray(s.prefix[:n],np.float32),student_takeover_first=np.asarray(s.handoff,np.int64))
 return a
def worker(pipe,index,p,counts):
 G.T.Session=Session;G.B.native_setup=F.field_setup;P.frozen=L.frozen;E.payload=payload;G.cpu_worker(pipe,index,p,counts)
def receive(self,i):
 value=F.Pool.receive(self,i)
 if isinstance(value,tuple):
  a,r=value;r['teacher']='teacher_first_declared' in a
  if r['teacher']:r.update(teacher_kind='FIXED_GAP_4S_THEN_DETERMINISTIC_STUDENT',teacher_first=int(a['teacher_first_declared']),teacher_stop=int(a['teacher_stop_declared']),training_only=True)
 return value
Pool.receive=receive
PREFIX=['states','next_states','actions','native_counts','poses','next_poses','hidden_before','history_poses','history_goals','history_times','locations','innovations','raw_actions','atom','logp','stochastic']
def actor_tape_audit(p,r,a):
 proposals=np.clip(a['raw_actions'],-1,1).astype(np.float32);F.M.audit_full_history(p,dict(r,teacher=False),dict(a,actions=proposals,ownership=np.zeros(len(a['states']),np.uint8)));return proposals

def inspect(p,vals):
 for a,r in vals:
  job=r['request']['job']
  if 'teacher_first_declared' in a:
   k=int(a['teacher_first_declared']);stop=int(a['teacher_stop_declared']);b=E.N.S.arrays(job['source_path']);assert stop-k==p['correction_decisions'] and len(a['states'])>k
   assert all(np.array_equal(a[n][:k],b[n][:k]) for n in PREFIX);assert all(np.array_equal(a[n][k],b[n][k]) for n in ['states','poses','hidden_before','history_poses','history_goals','history_times']);assert np.array_equal(a['initial_native_integration'],b['initial_native_integration'])
   own=(np.arange(len(a['states']))>=k)&(np.arange(len(a['states']))<stop);assert np.array_equal(a['ownership'],own) and a['stochastic'][:k].all() and not a['stochastic'][k:].any()
   for i,(pose,scan) in enumerate(zip(a['poses'],a['teacher_scans'])):assert np.array_equal(Teacher.control('GAP',pose,a['task_goal_xy'],scan,a['teacher_vehicle_parameters']),a['teacher_actions'][i])
   assert np.array_equal(a['actions'][own],a['teacher_actions'][own]);pr=actor_tape_audit(p,r,a);assert np.array_equal(a['actions'][~own],pr[~own])
  elif 'student_takeover_first' in a:
   k=int(a['student_takeover_first']);b=E.N.S.arrays(job['source_path']);assert all(np.array_equal(a[n][:k],b[n][:k]) for n in ['states','next_states','actions','native_counts','poses','next_poses','hidden_before','history_poses','history_goals','history_times']);assert all(np.array_equal(a[n][k],b[n][k]) for n in ['states','poses','hidden_before','history_poses','history_goals','history_times']);assert a['ownership'][:k].all() and not a['ownership'][k:].any() and not a['stochastic'].any();pr=actor_tape_audit(p,r,a);assert np.array_equal(a['actions'][k:],pr[k:])
  else:F.M.audit_full_history(p,r,a)
  assert np.array_equal(a['states'][1:,117:119],a['actions'][:-1])
def safe(r):return bool(r['success'] and not r['collision'] and r['field_detail']['field_qualified_success'])
def main():
 p=read(OUT/'protocol.json');setup();verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=Base.Runtime(p);obj=shadow=before=None;result=None;error=None;state='FAILED'
 try:
  rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
  vv=rt.request(p['qualification_jobs'],p['initial'],'FIXED_4S_TEACHER_THEN_ORIGINAL_STUDENT',seeds=p['qualification_seeds'],full=True);inspect(p,vv);qr=[r for a,r in vv];rt.request(p['qualification_jobs'],p['initial'],'EXACT_4S_HAND_BACK',seeds=p['qualification_seeds'],full=True,reference=qr);qualified=[r for r in qr if safe(r)];save(OUT/'qualification.json',dict(all_rows=qr,qualified=qualified))
  result=dict(qualification=qr,qualified_jobs=[r['job_id'] for r in qualified],new_Actor_Adam=0,new_RL=0,accepted=False,model=p['base_model'],post_handoff=None,check=None,full407=None,fresh16=None,reload78=None,globally_promoted=False)
  if qualified:
   obj=L.make(p);shadow=L.make(p);before=L.state(obj);assert digest(before)==p['initial']['joint_hash'];data=L.dataset(p,qualified);event=lambda w:(lambda t,k:Base.append('optimizer_calls.jsonl',[w,t,k]));rt.role='LEARN_VERIFIED_SHORT_CORRECTION'
   u=L.fit(obj,data,p,lambda:rt.guard(True),event('live'));v=L.fit(shadow,data,p,lambda:rt.guard(True),event('shadow'));assert u==v and digest(L.state(obj))==digest(L.state(shadow));model=L.export(obj,p,OUT/'proposal.pt.gz');assert L.frozen(p,model['path']).phash()==obj[0].phash();save(OUT/'LEARNING_COMPLETE.json',dict(epoch=time.time()));result.update(update=u,proposal=model,new_Actor_Adam=u['Actor_Adam'])
   vv=rt.request(p['handoff_jobs'],model,'PURE_STUDENT_AT_SAME_ACTUAL_PREFIX',full=True);inspect(p,vv);hr=[r for a,r in vv];rt.request(p['handoff_jobs'],model,'EXACT_LEARNED_HANDOFF',full=True,reference=hr);new=[r['job_id'] for r in hr if safe(r)];result.update(post_handoff=hr,new_handoff_success_ids=new)
   if new:
    keys=[j for j in p['evaluation_jobs'] if j['id'] in p['key_replay_ids']];others=[j for j in p['evaluation_jobs'] if j['id'] not in p['key_replay_ids']];vv=rt.request(keys,model,'ORIGINAL_START_STUDENT49',full=True);keyrows=[r for a,r in vv];vv+=rt.request(others,model,'ORIGINAL_START_STUDENT49');inspect(p,vv);by={r['job_id']:r for a,r in vv};check=assess(p,[by[j['id']] for j in p['evaluation_jobs']]);rt.request(keys,model,'EXACT_KEY4',full=True,reference=keyrows);result.update(check=check,accepted=not check['lost_initial'])
   if not result['accepted']:
    L.load(obj,before);L.load(shadow,before)
   result.update(retained_joint_hash=digest(L.state(obj)),model=model if result['accepted'] else p['base_model'])
  state='COMPLETE'
 except BaseException:
  error=traceback.format_exc()
  if before is not None:
   for z in [obj,shadow]:L.load(z,before)
  save(OUT/'failure.json',dict(error=error,no_restart=True,raw_preserved=True,whole_ActorAdam_rollback=True))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=2
  save(OUT/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
 assert state=='COMPLETE',error

def audit():
 p=read(OUT/'protocol.json');setup();verify(p,True);G.stopped(OUT);assert not (OUT/'audit.json').exists();s=read(OUT/'summary.json');assert s['state']=='COMPLETE';v=s['result'];rows=G.rows(OUT);reps=G.rows(OUT,'replays.jsonl')
 for r in rows:
  a=E.N.S.arrays(r['path']);assert sha(r['path'])==r['sha256'] and int(a['native_counts'].sum())==r['physics'] and all(digest(w)==r['array_hashes'][k] for k,w in a.items())
  if r['full']:inspect(p,[(a,r)])
 by={r['path']:r for r in rows}
 for r in reps:assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256']
 qualified=[r for r in rows[:3] if safe(r)];assert [r['job_id'] for r in qualified]==v['qualified_jobs'];hist=0
 if qualified:
  obj=L.make(p);before=L.state(obj);events=[];u=L.fit(obj,L.dataset(p,qualified),p,lambda:None,lambda t,k:events.append(['live',t,k]));hist=u['Actor_Adam'];assert u==v['update'] and digest(L.state(obj))==v['proposal']['joint_hash'] and sha(v['proposal']['path'])==v['proposal']['sha256']
  record=G.rows(OUT,'optimizer_calls.jsonl');assert [r for r in record if r[0]=='live']==events and [['live',t,k] for w,t,k in record if w=='shadow']==events
  if v['new_handoff_success_ids']:
   assert len(rows)==55 and len(reps)==10 and s['commands']==69;rr={r['job_id']:r for r in rows[6:]};assert assess(p,[rr[j['id']] for j in p['evaluation_jobs']])==v['check']
  else:assert len(rows)==len(reps)==6 and s['commands']==16 and v['check'] is None
  if not v['accepted']:L.load(obj,before)
  assert digest(L.state(obj))==v['retained_joint_hash'];assert read(OUT/'LEARNING_COMPLETE.json')['epoch']<p['absolute_learning_stop']
 else:assert len(rows)==len(reps)==3 and s['commands']==10 and not (OUT/'optimizer_calls.jsonl').exists()
 assert s['physics']==sum(r['physics'] for r in rows+reps)<=p['max_physics'] and s['commands']<=p['max_commands']
 save(OUT/'audit.json',dict(state='PASS',new_native=0,discarded_historical_Adam=hist,all_recorded_histories_and_exact_replays_verified=True,actual_source_prefix_and_teacher_window_verified=True,all_ActorAdam_shadow_rollback_rebuilt_if_executed=True,physics=s['physics'],commands=s['commands'],full407=None));print('T587_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
