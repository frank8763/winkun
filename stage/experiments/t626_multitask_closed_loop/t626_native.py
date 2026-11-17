from pathlib import Path
import sys,os,time,copy,gzip,json
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t608_current_policy_memory'))
import t608_native as N
L,G,E,P,F,torch,np=N.L,N.G,N.E,N.P,N.F,N.torch,N.np
class Session(F.FieldSession):
 def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.first=-1
def restore(p):
 real=G._real_affinity;os.sched_setaffinity=real;real(0,set(p['stage_cpus']))
def worker(pipe,index,p,counts):
 os.environ['CUDA_VISIBLE_DEVICES']='';assert not torch.cuda.is_initialized();real=G._real_affinity;cpus=p['stage_cpus']
 def pin(pid,mask):return real(pid,{cpus[index]}) if pid==0 and set(mask)=={index} else real(pid,mask)
 os.sched_setaffinity=pin;G.T.Session=Session;G.B.native_setup=F.field_setup;P.frozen=L.frozen;E.payload=N.payload;G.T.worker(pipe,index,p,counts)
class Pool(N.Pool):
 def __init__(self,p,counts,guard):
  self.p=p;self.guard,self.children,self.pipes=guard,[],[];ctx=G.R.M.mp.get_context('spawn');self.serial=0
  try:
   for i in range(3):
    a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
   self.ready=[self.receive(i) for i in range(3)];assert all(v['state']=='READY' for v in self.ready);actual=[sorted(os.sched_getaffinity(c.pid)) for c in self.children];assert actual==[[8],[10],[12]],actual;L.save(Path(p['out'])/'worker_generations.json',[dict(parent=os.getpid(),workers=[c.pid for c in self.children],actual_affinity=actual,epoch=time.time())])
  except BaseException:self.close();raise
 def request(self,items):self.pending=items;self.serial+=1;return super().request(items)
 def receive(self,i):
  v=super().receive(i)
  if isinstance(v,tuple):
   a,r=v;path=Path(self.p['out'])/f'worker_return_{self.serial}_{i}.json.gz';blob=gzip.compress(json.dumps(dict(request=self.pending[i],native_result=copy.deepcopy(r)),separators=(',',':')).encode(),mtime=0)
   with path.open('xb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
  return v
