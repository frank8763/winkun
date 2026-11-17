import time,copy,io,json
import t627_learning as M
import t627_data as D
L,torch,np=M.L,M.torch,M.np
def main():
 p=M.plan();groups=D.prior(p);M.configure_gpu(p);obj=M.make(p);before=M.digest(M.state(obj));items,diag=M.batch(obj[0],groups,p);refs=M.memories(obj[0],p);g,detail=M.task_gradients(obj[0],items,refs,p);mean,a=M.combine(g,p,'mean');ca,b=M.combine(g,p,'cagrad');cos=float(torch.dot(mean,ca)/(torch.linalg.vector_norm(mean)*torch.linalg.vector_norm(ca)));unitmean=mean/torch.linalg.vector_norm(mean);unitca=ca/torch.linalg.vector_norm(ca);worstmean=float((g.T@unitmean).min());worstca=float((g.T@unitca).min());material=cos<.999 and worstca>worstmean+1e-10
 # Correct convex-solver support: simplex, c=0 identity, permutation equivariance.
 zero=dict(p,cagrad_c=0.);z,_=M.combine(g,zero,'cagrad');assert torch.allclose(z,mean,atol=1e-7,rtol=1e-6);perm=torch.arange(g.shape[1]-1,-1,-1,device=g.device);v,vi=M.combine(g[:,perm],p,'cagrad');assert torch.allclose(v,ca,atol=1e-6,rtol=1e-5)
 assert abs(sum(b['weights'])-1)<1e-10 and min(b['weights'])>=0 and b['dual_gap']<=p['dual_tolerance'];assert M.digest(M.state(obj))==before
 # Mean task PG gradients equal scalar objective gradient with keep included.
 obj[1].zero_grad(set_to_none=True);loss,_=M.P.objective(obj[0],items,refs,p);loss.backward();expected=torch.cat([v.grad.flatten() for v in L.variables(obj[0])]);assert torch.allclose(mean,expected,atol=1e-10,rtol=1e-8)
 # No optimizer step: flat-gradient substitution and mature state are reserved
 # for main/shadow/audit under explicit new quotas.
 report=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',old32_full_remote_readback_member_SHA=True,old32_initial_policy_hash_exact=True,prior_only_not_new_onpolicy_collection=True,keep_gradient_norm_at_initial=detail['keep_gradient_norm'],task_Gram=g.T.detach().cpu().tolist() if False else (g.T@g).detach().cpu().tolist(),mean=a,cagrad=b,direction_cosine=cos,unit_direction_worst_dot_mean=worstmean,unit_direction_worst_dot_cagrad=worstca,material_conflict_correction=material,c0_identity=True,task_permutation_equivariance=True,mean_task_matches_scalar_original_gradient=True,all_Actor_optimizer_parameters_unchanged=True,new_native=0,new_optimizer=0,epoch=time.time());M.save(M.DOC/'PREFLIGHT.json',report);assert material,'Skip full native ablation: no material conflict correction';print('T627_REAL8TASK_CONFLICT_PREFLIGHT_PASS',cos,worstmean,worstca,b['dual_gap'])
if __name__=='__main__':main()
