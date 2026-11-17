from pathlib import Path
import sys,os,time,copy
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t620_recurrent_prior_td3'))
import t620_native as Old
L,G,E,P,F,torch,np=Old.L,Old.G,Old.E,Old.P,Old.F,Old.torch,Old.np

class Session(F.FieldSession):
 def __init__(self,job,*args,**kw):
  super().__init__(job,*args,**kw);self.first=-1;self.source=E.N.S.arrays(self.q.p['prefix_source']);self.k=self.q.p['branch_decision'];self.delta=job['steering_delta']
 def step(self,forced=None,owner=0,actor_active=True):
  assert forced is None and owner==0 and actor_active;i=len(self.rows)
  if i<=self.k:assert np.array_equal(self.packet(),self.source['states'][i]),i
  original=self.q.step
  def selected(*args,**kwargs):
   action,loc,h,extra=original(*args,**kwargs)
   if i==self.k:
    action=action.clone();action[0,1]=(loc[0,1]+self.delta).clamp(-1,1).float()
   return action,loc,h,extra
  self.q.step=selected
  try:return super().step(self.source['actions'][i] if i<self.k else None,owner=int(i<self.k or i==self.k and self.delta!=0),actor_active=True)
  finally:self.q.step=original

def inspect(p,values):
 for a,r in values:
  k=p['branch_decision'];source=E.N.S.arrays(p['prefix_source']);n=len(a['states']);delta=r['request']['job']['steering_delta'];mask=np.arange(n)<k
  if delta!=0:mask[k]=True
  assert np.array_equal(a['ownership'],mask.astype(np.uint8))
  for key in ['states','next_states','actions','native_counts','poses','next_poses','history_poses','history_goals','history_times','hold_sha256']:assert np.array_equal(a[key][:k],source[key][:k]),key
  for key in ['states','poses','history_poses','history_goals','history_times']:assert np.array_equal(a[key][k],source[key][k]),key
  assert np.array_equal(a['initial_native_integration'],source['initial_native_integration'])
  expected=a['raw_actions'].clip(-1,1).astype(np.float32);expected[:k]=source['actions'][:k];expected[k,1]=np.float32(np.clip(a['locations'][k,1]+delta,-1,1));assert np.array_equal(expected,a['actions'])
  proposals=a['raw_actions'].clip(-1,1).astype(np.float32);view=dict(a,actions=proposals,ownership=np.zeros(n,np.uint8));Old.Previous.inspect(p,[(view,dict(r,teacher=False))])

def worker(pipe,index,p,counts):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
 real=os.sched_setaffinity;cpus=p['stage_cpus']
 def pin(pid,mask):return real(pid,{cpus[index]}) if pid==0 and set(mask)=={index} else real(pid,mask)
 os.sched_setaffinity=pin;G.T.Session=Session;G.B.native_setup=F.field_setup;P.frozen=L.frozen;E.payload=Old.Previous.payload;G.T.worker(pipe,index,p,counts)

class Pool(Old.Previous.Pool):
 def __init__(self,p,counts,guard):
  self.p=p;self.guard,self.children,self.pipes=guard,[],[];self.serial=0;ctx=G.R.M.mp.get_context('spawn')
  try:
   for i in range(3):
    a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
   self.ready=[self.receive(i) for i in range(3)];assert all(v['state']=='READY' for v in self.ready);L.save(Path(p['out'])/'worker_generations.json',[dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time(),CPU=p['stage_cpus'],GPU=False)])
  except BaseException:self.close();raise
 def request(self,items):self.pending=items;self.serial+=1;return super().request(items)
 def receive(self,i):
  value=super().receive(i)
  if isinstance(value,tuple):
   a,r=value;req=self.pending[i];L.save(Path(self.p['out'])/f'worker_return_{self.serial}_{i}.json',dict(request=req,native_result=copy.deepcopy(r)));r.update(teacher=True,deployment_evaluation=False,fixed_common_continuation=True)
  return value
