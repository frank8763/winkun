"""Single fixed early handover, with teacher memory computed on actual observations."""
from pathlib import Path
import sys,os,time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t608_current_policy_memory'));import t608_native as Previous
L,G,E,P,F,torch,np=Previous.L,Previous.G,Previous.E,Previous.P,Previous.F,Previous.torch,Previous.np

class TeacherTrace:
 def __init__(self,p):
  m=p['handover_teacher'];assert L.sha(m['path'])==m['sha256'];self.q=L.frozen(p,m['path']);assert self.q.phash()==m['policy_hash'];self.q.sample=False;self.q.reset_noise();self.hidden=None;self.rows=[]
 def propose(self,x):
  hb=np.zeros(64) if self.hidden is None else self.hidden[0,0].numpy().copy()
  with torch.no_grad():command,loc,self.hidden,_=self.q.step(torch.tensor(x[None]),self.hidden)
  z=self.q.tape[-1];self.rows.append(dict(teacher_actions=command[0].numpy().copy(),teacher_locations=loc[0].numpy().copy(),teacher_hidden_before=hb,teacher_extra_hidden_before=z['extra_hidden_before'].copy(),teacher_anchor_hidden_before=z['anchor_hidden_before'].copy()));return command[0].numpy().copy()
 def arrays(self):
  out={k:np.asarray([r[k] for r in self.rows]) for k in self.rows[0]}
  out.update(teacher_last_hidden_after=self.hidden[0,0].numpy().copy(),teacher_last_extra_hidden_after=self.q.extra_hidden[0,0].numpy().copy(),teacher_last_anchor_hidden_after=self.q.anchor_hidden[0,0].numpy().copy());return out

class Session(F.FieldSession):
 def __init__(self,job,*args,**kwargs):
  super().__init__(job,*args,**kwargs);self.first=-1;self.handover_first=int(self.q.p['teacher_first']);self.teacher_trace=TeacherTrace(self.q.p);self.source=E.N.S.arrays(self.q.p['candidate_prefix_source']);assert len(self.source['states'])>self.handover_first;self.boundary=None
 def step(self,forced=None,owner=0,actor_active=True):
  assert forced is None and owner==0 and actor_active;k=len(self.rows);x=self.packet().copy();before=E.N.K.R.integration(self.env).copy();u=self.teacher_trace.propose(x);assert np.array_equal(before,E.N.K.R.integration(self.env))
  if k<=self.handover_first:assert np.array_equal(x,self.source['states'][k]),('actual_prefix120',k)
  if k==self.handover_first:
   self.boundary=dict(native_steps=self.steps,actual120_sha=L.digest(x),teacher_hidden_sha=L.digest(self.teacher_trace.rows[-1]['teacher_hidden_before']),teacher_extra_sha=L.digest(self.teacher_trace.rows[-1]['teacher_extra_hidden_before']),candidate_hidden_sha=L.digest(self.hidden[0,0].numpy()))
  return super().step(u if k>=self.handover_first else None,owner=int(k>=self.handover_first),actor_active=True)

def payload(s,tape=None):
 a=Previous.payload(s,tape);assert len(s.teacher_trace.rows)==len(a['states']);a.update(s.teacher_trace.arrays());a['teacher_handover_first']=np.asarray(s.handover_first,np.int64);return a

def inspect(p,values):
 for a,r in values:
  assert r['teacher'] and r['mixed_teacher_diagnostic'] and not r['deployment_evaluation'];k=p['teacher_first'];source=E.N.S.arrays(p['candidate_prefix_source']);assert len(a['states'])>k
  for key in ['states','next_states','actions','native_counts','poses','next_poses','history_poses','history_goals','history_times','hold_sha256']:
   assert np.array_equal(a[key][:k],source[key][:k]),key
  for key in ['states','poses','history_poses','history_goals','history_times']:assert np.array_equal(a[key][k],source[key][k]),key
  assert np.array_equal(a['initial_native_integration'],source['initial_native_integration']) and int(a['native_counts'][:k].sum())==3800
  assert not a['ownership'][:k].any() and a['ownership'][k:].all() and not a['stochastic'].any();assert np.array_equal(a['actions'][k:],a['teacher_actions'][k:])
  Previous.inspect(p,[(a,r)])
  trace=TeacherTrace(p)
  for x in a['states']:trace.propose(x)
  rebuilt=trace.arrays();assert all(np.array_equal(v,a[key]) for key,v in rebuilt.items())
  assert trace.q.phash()==p['handover_teacher']['policy_hash']

def worker(pipe,index,p,counts):
 G.T.Session=Session;G.B.native_setup=F.field_setup;P.frozen=L.frozen;E.payload=payload;G.cpu_worker(pipe,index,p,counts)

class Pool(Previous.Pool):
 def __init__(self,p,counts,guard):
  self.p=p;self.guard,self.children,self.pipes=guard,[],[];ctx=G.R.M.mp.get_context('spawn')
  try:
   for i in range(2):
    a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
   self.ready=[self.receive(i) for i in range(2)];assert all(x['state']=='READY' for x in self.ready)
   L.save(Path(p['out'])/'worker_generations.json',[dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time(),CPU=[14,15],GPU=False)])
  except BaseException:self.close();raise
 def receive(self,i):
  value=super().receive(i)
  if isinstance(value,tuple):
   a,r=value;r.update(teacher=True,teacher_first=self.p['teacher_first'],mixed_teacher_diagnostic=True,deployment_evaluation=False,rollout_is_single_actor=False,candidate_policy_hash=r['policy_hash'],teacher_policy_hash=self.p['handover_teacher']['policy_hash'],teacher_model_sha256=self.p['handover_teacher']['sha256'],policy_hash_is_recorded_actor_NOT_entire_executed_controller=True)
  return value
