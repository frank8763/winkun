"""One Actor with an optional recorded successful guide prefix in training."""
from pathlib import Path
import sys,os,time,copy
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t608_current_policy_memory'))
import t608_native as Previous
L,G,E,P,F,torch,np=Previous.L,Previous.G,Previous.E,Previous.P,Previous.F,Previous.torch,Previous.np

class Session(F.FieldSession):
 def __init__(self,job,*args,**kw):
  super().__init__(job,*args,**kw);self.first=-1;self.guide_first=int(job.get('guide_first',job.get('student_takeover_first',-1)));path=job.get('guide_source',job.get('source_path'));self.source=E.N.S.arrays(path) if self.guide_first>=0 else None
 def step(self,forced=None,owner=0,actor_active=True):
  assert forced is None and owner==0 and actor_active;k=len(self.rows)
  if self.guide_first<0:return super().step()
  if k<=self.guide_first:assert np.array_equal(self.packet(),self.source['states'][k]),('guide_prefix_actual120',k)
  use=k<self.guide_first
  return super().step(self.source['actions'][k] if use else None,owner=int(use),actor_active=True)

def payload(s,tape=None):
 a=Previous.payload(s,tape)
 if s.guide_first>=0:a['recorded_guide_first']=np.asarray(s.guide_first,np.int64)
 return a

def inspect(p,values):
 for a,r in values:
  job=r['request']['job'];k=int(job.get('guide_first',job.get('student_takeover_first',-1)));n=len(a['states'])
  expected=(np.arange(n)<k) if k>=0 else np.zeros(n,dtype=bool)
  assert np.array_equal(a['ownership'],expected.astype(np.uint8))
  if 'recorded_guide_first' in a:assert int(a['recorded_guide_first'])==k
  proposals=a['raw_actions'].clip(-1,1).astype(np.float32)
  if k>=0:
   source=E.N.S.arrays(job.get('guide_source',job.get('source_path')))
   for key in ['states','next_states','actions','native_counts','poses','next_poses','history_poses','history_goals','history_times']:assert np.array_equal(a[key][:k],source[key][:k]),key
   for key in ['states','poses','history_poses','history_goals','history_times']:assert np.array_equal(a[key][k],source[key][k]),key
   assert np.array_equal(a['initial_native_integration'],source['initial_native_integration'])
   assert np.array_equal(a['actions'][k:],proposals[k:])
  else:assert np.array_equal(a['actions'],proposals)
  assert np.array_equal(a['states'][1:,117:119],a['actions'][:-1])
  # Historical handoff markers dispatch to an older forced-prefix validator.
  # The actual prefix was checked above; this derived view is only for Actor
  # computation and must not be misinterpreted as another handoff episode.
  view={key:value for key,value in a.items() if key not in ['student_takeover_first','source_prefix_actions']}
  view.update(actions=proposals,ownership=np.zeros(n,np.uint8))
  Previous.inspect(p,[(view,dict(r,teacher=False))])

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
 def request(self,items):self.pending=items;self.serial+=1;return super().request(items)
 def receive(self,i):
  value=super().receive(i)
  if isinstance(value,tuple):
   a,r=value;request=self.pending[i];L.save(Path(self.p['out'])/f'worker_return_{self.serial}_{i}.json',dict(request=request,native_result=copy.deepcopy(r)));k=int(request['job'].get('guide_first',request['job'].get('student_takeover_first',-1)))
   r.update(teacher=k>=0,recorded_guide_prefix=k>=0,deployment_evaluation=k<0,guide_first=k,mixed_behavior_not_student_onpolicy=k>=0)
  return value
