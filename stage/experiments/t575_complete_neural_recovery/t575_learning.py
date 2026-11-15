"""Fixed-basis complete verified recovery plus own-policy rehearsal; standard ridge."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t571_self_replay_continual'));import t571_learning as Prior
P,G,torch,np=Prior.P,Prior.G,Prior.torch,Prior.np
read,save,sha,digest=Prior.read,Prior.save,Prior.sha,Prior.digest
frozen,state,load,export=Prior.frozen,Prior.state,Prior.load,Prior.export

def make(p):
 q=frozen(p,p['initial']['path']);assert sha(p['initial']['path'])==p['initial']['sha256'];q.requires_grad_(False)
 return q,torch.optim.SGD(q.continual[-1].parameters(),lr=1.)
def fixed(q):return digest({k:v for k,v in q.state_dict().items() if not k.startswith('continual.2.')})
def basis(q,x):
 f,b=Prior.feature(q,x);b=b+q.continual(f);h=q.continual[1](q.continual[0](f));return torch.cat([h,torch.ones(len(h),1,dtype=h.dtype)],-1),b

def dataset(p,memory):
 q=make(p)[0];data=[]
 with torch.no_grad():
  for kind,rows in [('keep',memory),('repair',[p['repair_row']])]:
   for row in rows:
    assert sha(row['path'])==row['sha256'];a=G.E.N.S.arrays(row['path']);H,b=basis(q,torch.tensor(a['states']))
    if kind=='keep':
     y=torch.tensor(a['locations']);assert float((b-y).abs().max())<1e-9;mask=torch.ones(len(H),dtype=torch.bool);target=y
    else:
     mask=torch.tensor(a['ownership'].astype(bool));assert mask[:89].sum()==0 and mask[89:].all();y=torch.tensor(a['teacher_actions'],dtype=torch.float64);target=torch.where(y>=1,torch.maximum(b,torch.ones_like(b)),torch.where(y<=-1,torch.minimum(b,-torch.ones_like(b)),y))
    data.append(dict(id=row['job_id'],kind=kind,H=H[mask],base=b[mask],target=target[mask]))
 assert len(data)==23;return data

def solve(data,p):
 Q=torch.zeros((33,33),dtype=torch.float64);c=torch.zeros((33,2),dtype=torch.float64)
 for g in data:
  H=g['H'];w=1/((22 if g['kind']=='keep' else 1)*len(H));Q+=w*H.T@H;c-=w*H.T@(g['target']-g['base'])
 scale=float(Q.diag().mean());Q=Q/scale+p['ridge_relative']*torch.eye(33,dtype=torch.float64);c/=scale;w=torch.cholesky_solve(-c,torch.linalg.cholesky(Q));res=float((Q@w+c).abs().max());assert res<1e-7 and torch.isfinite(w).all()
 return w,dict(Q=digest(Q),c=digest(c),coefficients=digest(w),stationarity=res,rows=[dict(id=g['id'],kind=g['kind'],n=len(g['H']),RMS_before=float((g['base']-g['target']).square().mean().sqrt()),RMS_after=float((g['base']+g['H']@w-g['target']).square().mean().sqrt())) for g in data])
def apply(obj,w):
 body=fixed(obj[0])
 with torch.no_grad():obj[0].continual[-1].weight.add_(w[:-1].T);obj[0].continual[-1].bias.add_(w[-1])
 assert fixed(obj[0])==body
