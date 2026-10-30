"""Local gradient and function-preservation audit, no native steps/optimizers."""
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
from common import *

def main():
    assert not (OUT/'status.json').exists()
    plan=json.loads((HERE/'protocol.json').read_text())
    for p,h in plan['input_hashes'].items():assert sha(p)==h,p
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    with np.load(plan['common_states']) as z:common=torch.tensor(z['states'],dtype=torch.float32)
    bounds=[]
    for name,path in plan['checkpoints'].items():
        p=OriginalPolicy().eval();p.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['policy'])
        with torch.no_grad():raw=ParentPolicy.forward(p,common);saturated=raw.abs()>=.999999
        grad=None
        if saturated[:,0].any():
            loss=p.latent_mean(common[saturated[:,0]])[:,0].sum();grad=torch.autograd.grad(loss,p.native_head.bias)[0].tolist();assert grad==[0.,0.]
        bounds.append({'model':name,'checkpoint_sha256':sha(path),'common_state_count':len(common),'saturated_fraction_per_action':saturated.float().mean(0).tolist(),'raw_native_min':raw.min(0).values.tolist(),'raw_native_max':raw.max(0).values.tolist(),'local_throttle_latent_gradient_native_bias_on_saturated_states':grad})
    parent=torch.load(plan['checkpoints']['T194'],map_location='cpu',weights_only=False)['policy'];old=OriginalPolicy().eval();old.load_state_dict(parent)
    native=Policy('native').eval();native.load_parent(parent);latent=Policy('latent').eval();latent.load_parent(parent)
    arrays=[common.numpy()]
    for path in plan['verification_states']:
        with np.load(path) as z:arrays.append(z['states'])
    states=torch.tensor(np.concatenate(arrays)[:4096],dtype=torch.float32)
    with torch.no_grad():errors={name:max(float((p(x)-old(x)).abs().max()) for x in states.split(64)) for name,p in [('native',native),('latent',latent)]}
    assert max(errors.values())==0
    mask=ParentPolicy.forward(old,common).detach()[:,0]>=.999999;assert mask.any();selected=common[mask]
    noise=torch.tensor(np.random.default_rng(plan['seed']).normal(size=(8,len(selected),2)),dtype=torch.float32)
    gradients={}
    for name,p in [('native',native),('latent',latent)]:
        mu=p.latent_mean(selected)[None,:,:];sigma=.5;z=mu+sigma*noise
        logp=torch.distributions.Normal(mu,sigma).log_prob(z)-2*(math.log(2)-z-torch.nn.functional.softplus(-2*z))
        loss=logp[:,:,0].mean();g=torch.autograd.grad(loss,p.extra_head.weight)[0]
        gradients[name]=float(g[0].norm())
    assert gradients['native']==0 and gradients['latent']>0
    for p in [native,latent]:
        for k,v in parent.items():assert torch.equal(p.state_dict()[k],v)
        assert torch.count_nonzero(p.extra_head.weight)==0 and torch.count_nonzero(p.extra_head.bias)==0
    checkpoint=OUT/'initial_residual_policy.pt';torch.save({'policy':latent.state_dict(),'parent_sha256':sha(plan['checkpoints']['T194']),'optimizer_updates':0,'architecture':'SACResidualPlacementPolicy','initial_output_errors':errors},checkpoint)
    result={'state':'COMPLETE','common_state_audit':bounds,'initial_function_errors':errors,'verification_states':len(states),'saturated_state_entropy_gradient_extra_steering_throttle_head':gradients,'initial_checkpoint_sha256':sha(checkpoint),'native_physics_steps':0,'optimizer_updates':0,'automatic_promotion':False,'incumbent':'T159','scope':'Exact local gradient blockage from inverse-squash clipping. New post-atanh residual restores a local mean-gradient path with same initial behavior. Native and latent controls have equal1202 newparameters. Shared oldparameters can still move clipped states indirectly; not proof old model permanently untrainable or blockage alone caused LEFTfailure. No online new-head evidence yet.'}
    save(OUT/'summary.json',result);save(OUT/'status.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
