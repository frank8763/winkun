"""Fixed feature subspace and explicit episodic Gaussian score gradient."""
import os
os.sched_setaffinity(0,{int(os.environ.get('STAGE_PROCESS_CPU','0'))})
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t207_parameter_pg'
sys.path.insert(0,str(ROOT/'stage/experiments/t203_native_exploration'))
import runtime as R
C=R.C;np=R.np;torch=R.torch;Policy=R.Policy

def direction_basis(plan,p):
 with np.load(plan['failure_trace']) as z:x=torch.tensor(z['states']);steps=z['steps']
 with np.load(plan['retention_targets']) as z:old=torch.tensor(z['states']);targets=torch.tensor(z['actions']);offsets=z['offsets'].copy()
 with torch.no_grad():f=torch.cat([p.body.net[:4](x),torch.ones((len(x),1))],1).double();g=torch.cat([p.body.net[:4](old),torch.ones((len(old),1))],1).double()
 lo,hi=plan['failure_step_window'];focus=f[(steps>=lo)&(steps<=hi)];A=focus.T@focus/len(focus);B=sum(g[offsets[j]:offsets[j+1]].T@g[offsets[j]:offsets[j+1]]/(offsets[j+1]-offsets[j]) for j in range(7))/7;ridge=float(B.trace()/len(B))*plan['ridge_multiplier'];v,u=torch.linalg.eigh(B+ridge*torch.eye(len(B),dtype=torch.float64));W=(u*v.rsqrt())@u.T;ev,vec=torch.linalg.eigh(W@A@W);basis=[]
 for i in range(plan['rank']):
  d=W@vec[:,-1-i];d/=torch.sqrt(torch.mean((focus@d)**2))
  if float((focus@d).sum())<0:d=-d
  basis.append(d)
 return torch.stack(basis,1).numpy(),old,targets,offsets,ev[-plan['rank']:].numpy()

def set_coefficients(p,parent,basis,coeff):
 p.load_state_dict(parent)
 delta=np.asarray(coeff).reshape(2,-1)@basis.T
 with torch.no_grad():p.direct_head.weight.add_(torch.tensor(delta[:,:-1],dtype=torch.float32));p.direct_head.bias.add_(torch.tensor(delta[:,-1],dtype=torch.float32))
 return p

def score_gradient(fplus,fminus,epsilon,sigma):
 return np.mean((np.asarray(fplus)-np.asarray(fminus))[:,None]*np.asarray(epsilon),axis=0)/(2*np.asarray(sigma))
