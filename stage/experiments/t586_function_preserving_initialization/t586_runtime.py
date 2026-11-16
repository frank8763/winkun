"""Finite qualified recovery learning with paired autonomous handoff controls."""
from pathlib import Path
import fcntl
import os
import shutil
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
import t586_learning as L
OUT=ROOT/'stage/runs/t586_function_preserving_initialization';DOC=ROOT/'stage/docs/day_20261116/t586'
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
 for name in ['states','next_states','actions','native_counts','poses','next_poses','history_poses','history_goals','history_times','hidden_before']:assert np.array_equal(a[name][:k],b[name][:k]),name
 for name in ['states','poses','history_poses','history_goals','history_times','hidden_before']:assert np.array_equal(a[name][k],b[name][k]),name
 assert np.array_equal(a['initial_native_integration'],b['initial_native_integration']) and not a['stochastic'].any()
 assert a['ownership'][:k].all() and not a['ownership'][k:].any();assert np.array_equal(a['actions'][:k],b['actions'][:k]);assert np.array_equal(a['states'][1:,117:119],a['actions'][:-1])
 proposals=np.clip(a['raw_actions'],-1,1).astype(np.float32);assert np.array_equal(a['actions'][k:],proposals[k:])
 F.M.audit_full_history(p,dict(r,teacher=False),dict(a,actions=proposals,ownership=np.zeros(len(a['states']),np.uint8)))
def inspect(p,vals):
 for a,r in vals:
  if 'student_takeover_first' in a:handoff_audit(p,r,a)
  else:F.M.audit_full_history(p,r,a)
def main():
 p=read(OUT/'protocol.json');setup();verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=Base.Runtime(p);rt.commands=0;obj=shadow=before=None;result=None;error=None;state='FAILED'
 try:
  obj=L.make(p);shadow=L.make(p);before=L.state(obj);data=L.dataset(p);initial=L.export(obj,p,OUT/'initial.pt.gz');event=lambda w:(lambda t,k:Base.append('optimizer_calls.jsonl',[w,t,k]))
  rt.role='FINITE_ANCHORED_RECOVERY_LEARNING';u=L.fit(obj,data,p,lambda:rt.guard(True),event('live'));v=L.fit(shadow,data,p,lambda:rt.guard(True),event('shadow'));assert u==v and digest(L.state(obj))==digest(L.state(shadow));model=L.export(obj,p,OUT/'proposal.pt.gz');assert L.frozen(p,model['path']).phash()==obj[0].phash();gate=L.offline_gate(obj[0],data,p);save(OUT/'LEARNING_COMPLETE.json',dict(epoch=time.time(),gate=gate))
  result=dict(update=u,initial=initial,proposal=model,offline_gate=gate,historical_A=p['historical_A_model'],new_RL=0,new_Critic=0,accepted=False,retained_joint_hash=digest(before),model=p['base_model'],handoff=None,check=None,full407=None,fresh16=None,reload78=None,globally_promoted=False)
  if gate['passed']:
   rt.commands+=2;rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
   vv=rt.request(p['handoff_jobs'],model,'PURE_STUDENT_AFTER_FIXED_SOURCE_PREFIX',full=True);inspect(p,vv);handoffs=[r for a,r in vv];rt.request(p['handoff_jobs'],model,'EXACT_LEARNED_HANDOFF',full=True,reference=handoffs)
   keys=[j for j in p['evaluation_jobs'] if j['id'] in p['key_replay_ids']];others=[j for j in p['evaluation_jobs'] if j['id'] not in p['key_replay_ids']]
   vv=rt.request(keys,model,'PURE_STUDENT_FROM_ORIGINAL_START',full=True);keyrows=[r for a,r in vv];vv+=rt.request(others,model,'PURE_STUDENT_FROM_ORIGINAL_START');inspect(p,vv);by={r['job_id']:r for a,r in vv};ordered=[by[j['id']] for j in p['evaluation_jobs']];rt.request(keys,model,'EXACT_KEY4',full=True,reference=keyrows);check=assess(p,ordered)
   new=[r['job_id'] for r in handoffs if r['success'] and not r['collision'] and r['field_detail']['field_qualified_success']];accepted=not check['lost_initial'] and bool(new or check['gained']);result.update(handoff=handoffs,new_handoff_success_ids=new,check=check,accepted=accepted)
   if accepted:result.update(model=model,retained_joint_hash=digest(L.state(obj)))
  if not result['accepted']:
   L.load(obj,before);L.load(shadow,before)
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
 p=read(OUT/'protocol.json');setup();verify(p,True);G.stopped(OUT);assert not (OUT/'audit.json').exists();s=read(OUT/'summary.json');assert s['state']=='COMPLETE';v=s['result'];rows=G.rows(OUT) if (OUT/'cases.jsonl').exists() else [];reps=G.rows(OUT,'replays.jsonl') if (OUT/'replays.jsonl').exists() else []
 for r in rows:
  a=E.N.S.arrays(r['path']);assert sha(r['path'])==r['sha256'] and int(a['native_counts'].sum())==r['physics'];assert all(digest(w)==r['array_hashes'][k] for k,w in a.items())
  if r['full']:inspect(p,[(a,r)])
 by={r['path']:r for r in rows}
 for r in reps:assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256']
 obj=L.make(p);before=L.state(obj);data=L.dataset(p);events=[];u=L.fit(obj,data,p,lambda:None,lambda t,k:events.append(['live',t,k]));assert u==v['update'] and digest(L.state(obj))==v['proposal']['joint_hash'];assert sha(v['proposal']['path'])==v['proposal']['sha256'] and L.offline_gate(obj[0],data,p)==v['offline_gate']
 if v['offline_gate']['passed']:
  assert len(rows)==52 and len(reps)==7 and s['commands']==63;rr={r['job_id']:r for r in rows[3:]};assert assess(p,[rr[j['id']] for j in p['evaluation_jobs']])==v['check']
 else:assert len(rows)==len(reps)==s['commands']==s['physics']==0 and v['check'] is None
 if not v['accepted']:L.load(obj,before)
 assert digest(L.state(obj))==v['retained_joint_hash'];record=G.rows(OUT,'optimizer_calls.jsonl');assert [r for r in record if r[0]=='live']==events and [['live',t,k] for w,t,k in record if w=='shadow']==events
 assert s['physics']==sum(r['physics'] for r in rows+reps)<=p['max_physics'] and s['commands']<=p['max_commands'];assert read(OUT/'LEARNING_COMPLETE.json')['epoch']<p['absolute_learning_stop']
 save(OUT/'audit.json',dict(state='PASS',new_native=0,discarded_historical_Adam=u['Actor_Adam'],all_ActorAdam_shadow_and_rollback_exact=True,offline_gate_recomputed=True,full7_history_and7_native_exact_if_executed=True,compact45_not_full_pose_reconstruction=True,physics=s['physics'],commands=s['commands']))
 print('T586_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
