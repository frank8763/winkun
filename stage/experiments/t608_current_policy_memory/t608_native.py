"""Architecture-aware Stage payload. Missing memory is never filled with fake zeros."""
from pathlib import Path
import sys,os,io
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t598_independent_recurrent_recovery'))
import t598_runtime as Legacy
L,G,E,P,F,torch,np=Legacy.L,Legacy.G,Legacy.E,Legacy.P,Legacy.F,Legacy.torch,Legacy.np

def blob(arrays):
 buf=io.BytesIO();np.savez_compressed(buf,**arrays);return buf.getvalue()

def payload(s,tape=None):
 a=Legacy.R.payload(s,tape);t=s.q.tape if tape is None else tape
 if isinstance(s.q,L.Actor):
  assert len(t)==len(a['states']) and len(t)>0
  for k in ['extra_hidden_before','anchor_hidden_before']:
   assert all(k in row for row in t),k;a[k]=np.asarray([z[k] for z in t])
  a['last_extra_hidden_after']=s.q.extra_hidden[0,0].detach().numpy().copy()
  a['last_anchor_hidden_after']=s.q.anchor_hidden[0,0].detach().numpy().copy()
 else:
  assert not any('extra_hidden_before' in z or 'anchor_hidden_before' in z for z in t)
 return a

def inspect(p,values):
 for a,r in values:
  model=L.frozen(p,r['request']['model'])
  if isinstance(model,L.Actor):Legacy.inspect(p,[(a,r)])
  else:
   assert not any(k in a for k in ['extra_hidden_before','anchor_hidden_before','last_extra_hidden_after','last_anchor_hidden_after'])
   Legacy.R.inspect(p,[(a,r)])

def worker(pipe,index,p,counts):
 G.T.Session=Legacy.R.Session;G.B.native_setup=F.field_setup;P.frozen=L.frozen;E.payload=payload;G.cpu_worker(pipe,index,p,counts)

class Pool(Legacy.Pool):
 def __init__(self,p,counts,guard):
  self.p=p;self.guard,self.children,self.pipes=guard,[],[];ctx=G.R.M.mp.get_context('spawn')
  try:
   for i in range(2):
    a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
   self.ready=[self.receive(i) for i in range(2)];assert all(x['state']=='READY' for x in self.ready)
   Legacy.save(Path(p['out'])/'worker_generations.json',[dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=__import__('time').time(),CPU=[14,15],GPU=False)])
  except BaseException:self.close();raise
