"""Mathematical and exact-initial-function gates, no optimizer or physics."""
from support import *
import json

def main():
 assert not (OUT/'contracts.json').exists();plan=json.loads((OUT/'protocol.json').read_text());torch.set_num_threads(1)
 p=Policy().eval().requires_grad_(False);parent=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['policy'];p.load_state_dict(parent);basis,old,targets,offsets,ev=direction_basis(plan,p)
 set_coefficients(p,parent,basis,np.zeros(6));assert all(torch.equal(v,p.state_dict()[k]) for k,v in parent.items())
 with torch.no_grad():assert torch.equal(p(old),targets)
 epsilon=np.arange(24,dtype=float).reshape(4,6)/13-.8;sigma=np.asarray(plan['sigma']).ravel();mu=np.arange(6,dtype=float)/10
 f=lambda x:np.sin(x.sum())-.2*np.dot(x,x)
 plus=np.asarray([f(mu+sigma*e) for e in epsilon]);minus=np.asarray([f(mu-sigma*e) for e in epsilon]);g=score_gradient(plus,minus,epsilon,sigma)
 eta=np.vstack([mu+sigma*epsilon,mu-sigma*epsilon]);returns=np.r_[plus,minus];reference=np.mean(returns[:,None]*(eta-mu)/sigma**2,axis=0);assert np.max(abs(g-reference))<1e-12
 change=.4*sigma*g*sigma/np.linalg.norm(g*sigma);kl=.5*np.sum((change/sigma)**2);assert abs(kl-.08)<1e-12
 np.savez_compressed(OUT/'basis.npz',basis=basis,eigenvalues=ev)
 C.save(OUT/'contracts.json',{'state':'PASS','basis_shape':list(basis.shape),'rank':3,'zero_coefficient_full_parameter_equality':True,'initial_retention_action_maxerror':0.,'mirrored_score_vs_explicit_gaussian_score_maxerror':float(np.max(abs(g-reference))),'declared_step_KL':float(kl),'optimizer_updates':0,'physics_steps':0,'basis_sha256':C.sha(OUT/'basis.npz')})
 print('T207 contracts PASS')
if __name__=='__main__':main()
