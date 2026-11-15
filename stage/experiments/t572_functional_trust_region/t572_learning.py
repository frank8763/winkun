"""Exact finite-memory output constraints on a frozen nonlinear policy basis.

A: unconstrained ridge; B: same strictly convex objective with two-sided
linear constraints, solved by bounded custom ADMM. Not OSQP or an RL update.
"""
from pathlib import Path
import sys,copy,gzip,io,time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t571_self_replay_continual'));import t571_learning as Prior
P,G,torch,np=Prior.P,Prior.G,Prior.torch,Prior.np
read,save,sha,digest=Prior.read,Prior.save,Prior.sha,Prior.digest
ARCH=Prior.ARCH

def make(p):
 q,opt=Prior.make(p);s=P.checkpoint(p['feature_source']['path']);assert sha(p['feature_source']['path'])==p['feature_source']['sha256']
 with torch.no_grad():
  for key,val in q.continual[0].state_dict().items():val.copy_(s['actor']['continual.0.'+key])
  q.continual[-1].weight.zero_();q.continual[-1].bias.zero_()
 q.requires_grad_(False);return q,torch.optim.SGD(q.continual[-1].parameters(),lr=1.)
def fixed(q):return digest({k:v for k,v in q.state_dict().items() if not k.startswith('continual.2.')})
def state(obj):return Prior.state(obj)
def load(obj,s):Prior.load(obj,s)
def frozen(p,path):return Prior.frozen(p,path)
def export(obj,p,path):return Prior.export(obj,p,path)

def dataset(p):
 q=make(p)[0];data=[]
 with torch.no_grad():
  for kind,rows in [('keep',p['memory_rows']),('repair',p['repair_rows'])]:
   for row in rows:
    assert sha(row['path'])==row['sha256'];a=G.E.N.S.arrays(row['path']);x=torch.tensor(a['states']);f,b=Prior.feature(q,x);h=q.continual[1](q.continual[0](f));h=torch.cat([h,torch.ones(len(h),1,dtype=h.dtype)],-1);mask=torch.tensor(a['ownership'].astype(bool)) if kind=='repair' and row['teacher_suffix_only'] else torch.ones(len(h),dtype=torch.bool)
    if kind=='keep':
     y=torch.tensor(a['locations']);assert float((q.sequence(x)-y).abs().max())<1e-9;target=y
    else:
     y=torch.tensor(a['teacher_actions'] if row['teacher_suffix_only'] else a['actions'],dtype=torch.float64)
     # Canonical nearest feasible deterministic location for boundary commands.
     target=torch.where(y>=1,torch.maximum(b,torch.ones_like(b)),torch.where(y<=-1,torch.minimum(b,-torch.ones_like(b)),y))
    data.append(dict(id=row['job_id'],kind=kind,H=h[mask],base=b[mask],target=target[mask],action_target=y[mask],source_sha256=row['sha256']))
 assert len(data)==23 and sum(g['kind']=='keep' for g in data)==21
 return data

def matrices(data,p):
 D=data[0]['H'].shape[1];Q=torch.zeros((D,D),dtype=torch.float64);c=torch.zeros((D,2),dtype=torch.float64)
 for kind,count in [('keep',21),('repair',2)]:
  for g in data:
   if g['kind']!=kind:continue
   H=g['H'];delta=g['target']-g['base'];w=1/(count*len(H));Q+=w*H.T@H;c-=w*H.T@delta
 scale=float(Q.diag().mean());assert scale>0;Q=Q/scale+p['ridge_relative']*torch.eye(D,dtype=torch.float64);c=c/scale;A=torch.cat([g['H'] for g in data if g['kind']=='keep']);norm=A.norm(dim=1);assert float(norm.min())>0
 return Q,c,A,norm

def certificate(Q,c,A,b,w,lam):
 v=A@w;upper=(v-b).clamp(min=0);lower=(-v-b).clamp(min=0);station=Q@w+c+A.T@lam;lp=lam.clamp(min=0);ln=(-lam).clamp(min=0)
 return dict(primal_inf=float(torch.maximum(upper,lower).max()),stationarity_inf=float(station.abs().max()),complementarity_inf=max(float((lp*(b-v)).abs().max()),float((ln*(b+v)).abs().max())),finite=bool(torch.isfinite(w).all() and torch.isfinite(lam).all()))

def solve(Q,c,rawA,norm,arm,p,guard):
 started=time.time();D=len(Q);zero=torch.zeros_like(c);L=torch.linalg.cholesky(Q);unconstrained=torch.cholesky_solve(-c,L);trace=[]
 if arm=='unconstrained':
  residual=float((Q@unconstrained+c).abs().max());passed=residual<=p['kkt_tolerance'];w=unconstrained;lam=torch.zeros((len(rawA),2),dtype=Q.dtype);info=dict(iterations=0,closed_form_linear_solves=1,certificate_pass=passed,certificate=dict(stationarity_inf=residual,finite=bool(torch.isfinite(w).all())),trace=trace)
 else:
  assert arm=='bounded';A=rawA/norm[:,None];b=torch.full((len(A),1),p['drift_limit'],dtype=Q.dtype)/norm[:,None];rho=D/len(A);K=Q+rho*A.T@A;factor=torch.linalg.cholesky(K);w=zero.clone();z=torch.zeros((len(A),2),dtype=Q.dtype);u=z.clone();passed=False;cert=None
  for k in range(p['admm_max_iterations']):
   if k%10==0:guard()
   w=torch.cholesky_solve(-c+rho*A.T@(z-u),factor);aw=A@w;relaxed=p['over_relaxation']*aw+(1-p['over_relaxation'])*z;newz=torch.minimum(torch.maximum(relaxed+u,-b),b);u=u+relaxed-newz;z=newz
   if (k+1)%10==0 or k+1==p['admm_max_iterations']:
    cert=certificate(Q,c,A,b,w,rho*u)
    if (k+1)%50==0:trace.append([k+1,cert['primal_inf'],cert['stationarity_inf'],cert['complementarity_inf']])
    passed=cert['finite'] and max(cert['primal_inf'],cert['stationarity_inf'],cert['complementarity_inf'])<=p['kkt_tolerance'] and float((rawA@w).abs().max())<=p['drift_limit']+p['raw_drift_tolerance']
    if passed:break
  lam=rho*u;info=dict(iterations=k+1,closed_form_linear_solves=k+2,certificate_pass=passed,certificate=cert,rho=rho,trace=trace)
 info.update(arm=arm,raw_memory_location_drift_max=float((rawA@w).abs().max()),coefficient_norm=float(w.norm()),objective=float(.5*(w*(Q@w)).sum()+(c*w).sum()),new_Adam=0,new_RL=0)
 assert torch.isfinite(w).all();return w,lam,info

def measure(w,data):
 return [dict(id=g['id'],kind=g['kind'],location_RMS=float((g['base']+g['H']@w-g['target']).square().mean().sqrt()),command_RMS=float(((g['base']+g['H']@w).clamp(-1,1).float().double()-g['action_target'].clamp(-1,1)).square().mean().sqrt()),location_drift_max=float((g['H']@w).abs().max())) for g in data]
def apply(obj,w):
 body=fixed(obj[0])
 with torch.no_grad():obj[0].continual[-1].weight.copy_(w[:-1].T);obj[0].continual[-1].bias.copy_(w[-1])
 assert fixed(obj[0])==body and not obj[1].state
