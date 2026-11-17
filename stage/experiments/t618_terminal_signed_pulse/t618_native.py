"""Training-only fixed steering pulse; the deployed Actor is unchanged."""
from pathlib import Path
import sys,os,time,copy
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t608_current_policy_memory'))
import t608_native as Previous
L,G,E,P,F,torch,np=Previous.L,Previous.G,Previous.E,Previous.P,Previous.F,Previous.torch,Previous.np

def pulse(action,loc,delta):
 out=action.clone();out[0,1]=(loc[0,1]+delta).clamp(-1,1).float();return out

class Session(F.FieldSession):
 def __init__(self,job,*args,**kwargs):
  super().__init__(job,*args,**kwargs);self.first=-1;self.start=int(self.q.p['teacher_first']);self.duration=int(self.q.p['pulse_decisions']);self.delta=float(job['steering_delta']);self.source=E.N.S.arrays(self.q.p['candidate_prefix_source'])
 def step(self,forced=None,owner=0,actor_active=True):
  assert forced is None and owner==0 and actor_active;k=len(self.rows);x=self.packet().copy()
  if k<=self.start:assert np.array_equal(x,self.source['states'][k]),('actual_prefix',k)
  active=self.delta!=0 and self.start<=k<self.start+self.duration;original=self.q.step
  def selected(*args,**kwargs):
   action,loc,hidden,extra=original(*args,**kwargs)
   return (pulse(action,loc,self.delta) if active else action),loc,hidden,extra
  self.q.step=selected
  try:return super().step(owner=int(active),actor_active=True)
  finally:self.q.step=original

def payload(s,tape=None):
 a=Previous.payload(s,tape);a.update(pulse_start_decision=np.asarray(s.start,np.int64),pulse_duration_decisions=np.asarray(s.duration,np.int64),pulse_steering_delta=np.asarray(s.delta,np.float64));return a

def inspect(p,values):
 for a,r in values:
  k=p['teacher_first'];delta=r['request']['job']['steering_delta'];source=E.N.S.arrays(p['candidate_prefix_source']);n=len(a['states']);active=(np.arange(n)>=k)&(np.arange(n)<k+p['pulse_decisions'])&(delta!=0)
  assert np.array_equal(a['ownership'],active.astype(np.uint8)) and not a['stochastic'].any()
  for key in ['states','next_states','actions','native_counts','poses','next_poses','history_poses','history_goals','history_times','hold_sha256']:assert np.array_equal(a[key][:k],source[key][:k]),key
  for key in ['states','poses','history_poses','history_goals','history_times']:assert np.array_equal(a[key][k],source[key][k]),key
  assert np.array_equal(a['initial_native_integration'],source['initial_native_integration'])
  expected=a['locations'].copy();expected[active,1]+=delta;expected=expected.clip(-1,1).astype(np.float32);assert np.array_equal(expected,a['actions'])
  # Existing double-memory audit reconstructs the original Actor's proposal,
  # then checks forced actions separately; no teacher hidden is invented.
  Previous.inspect(p,[(dict(a,teacher_actions=a['actions']),dict(r,teacher=True))])

def worker(pipe,index,p,counts):
 G.T.Session=Session;G.B.native_setup=F.field_setup;P.frozen=L.frozen;E.payload=payload;G.cpu_worker(pipe,index,p,counts)

class Pool(Previous.Pool):
 def __init__(self,p,counts,guard):
  self.p=p;self.guard,self.children,self.pipes=guard,[],[];ctx=G.R.M.mp.get_context('spawn');self.serial=0
  try:
   for i in range(2):
    a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
   self.ready=[self.receive(i) for i in range(2)];assert all(v['state']=='READY' for v in self.ready)
   L.save(Path(p['out'])/'worker_generations.json',[dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time(),CPU=[14,15],GPU=False)])
  except BaseException:self.close();raise
 def request(self,items):
  self.pending_requests=items;self.serial+=1;return super().request(items)
 def receive(self,i):
  value=super().receive(i)
  if isinstance(value,tuple):
   a,r=value;request=self.pending_requests[i]
   # Persist unannotated results before any derived metadata can throw.
   L.save(Path(self.p['out'])/f'worker_return_{self.serial}_{i}.json',dict(request=request,native_result=copy.deepcopy(r)))
   r.update(teacher=bool(request['job']['steering_delta']!=0),fixed_training_pulse=True,deployment_evaluation=False,rollout_is_single_actor=False,signed_steering_delta=request['job']['steering_delta'],pulse_start=self.p['teacher_first'],pulse_decisions=self.p['pulse_decisions'])
  return value
