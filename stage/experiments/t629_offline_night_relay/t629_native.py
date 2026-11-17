"""Validated native IO with optional fixed candidate-prefix recovery diagnostic."""
from pathlib import Path
import os,sys
import t629_learning as M
sys.path.insert(0,str(M.ROOT/'stage/experiments/t626_multitask_closed_loop'))
import t626_native as N
sys.path.insert(0,str(M.ROOT/'stage/experiments/t591_joint_representation_pair'))
import t591_runtime as Handoff
class Session(N.Session):
 def __init__(self,job,*args,**kw):
  super().__init__(job,*args,**kw);self.first=int(job.get('student_takeover_first',-1));self.source=N.E.N.S.arrays(job['source_path']) if self.first>=0 else None;self.prefix_actions=[]
 def step(self,forced=None,owner=0,actor_active=True):
  assert forced is None and owner==0
  if self.first<0:return super().step()
  k=len(self.rows);self.q.sample=False;use=k<self.first;u=self.source['actions'][k] if use else N.np.zeros(2,N.np.float32);self.prefix_actions.append(u.copy());return super().step(u if use else None,owner=int(use),actor_active=True)
BASE_PAYLOAD=N.N.payload
BASE_INSPECT=N.N.inspect

def payload(s,tape=None):
 a=BASE_PAYLOAD(s,tape)
 if s.first>=0:a.update(source_prefix_actions=N.np.asarray(s.prefix_actions[:len(a['states'])],N.np.float32),student_takeover_first=N.np.asarray(s.first,N.np.int64))
 return a

def inspect(p,values):
 for a,r in values:
  q=M.frozen(p,r['request']['model']);pc=dict(p,raw_sigma=list(q.p['raw_sigma']));BASE_INSPECT(pc,[(a,r)])
  if 'student_takeover_first' in a:
   assert r['request']['seed'] is None;Handoff.handoff_audit(pc,r,a)

def install():N.L.frozen=M.frozen;N.N.L.frozen=M.frozen;N.P.frozen=M.frozen

def worker(pipe,index,p,counts):
 os.environ['CUDA_VISIBLE_DEVICES']='';assert not N.torch.cuda.is_initialized();install();real=N.G._real_affinity;cpus=p['stage_cpus']
 def pin(pid,mask):return real(pid,{cpus[index]}) if pid==0 and set(mask)=={index} else real(pid,mask)
 os.sched_setaffinity=pin;N.G.T.Session=Session;N.G.B.native_setup=N.F.field_setup;N.E.payload=payload;N.G.T.worker(pipe,index,p,counts)
