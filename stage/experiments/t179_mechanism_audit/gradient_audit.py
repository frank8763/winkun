"""Read-only first/last-batch diagnostics; gradients computed but never applied."""
from pathlib import Path
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import sys,json,math,hashlib
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[3];OUT=REPO/'stage/runs/t179_mechanism_audit'
sys.path.insert(0,str(REPO/'stage/experiments/t177_frontier_onpolicy'))
import common as C
torch.set_num_threads(1)
assert not (OUT/'gradient_audit.json').exists()
initial=torch.load(REPO/'stage/runs/t177_frontier_onpolicy/initial_policy.pt',weights_only=False,map_location='cpu')['policy']
plan0=json.loads((REPO/'stage/experiments/t177_frontier_onpolicy/protocol.json').read_text())
with np.load(plan0['retention_demonstrations']) as z:states=z['states'].copy();actions=z['actions'].copy();offsets=z['episode_offsets'].copy();counts=z['native_counts'].copy()
actions[:,0]=.5*(actions[:,0]+1);clocks=[]
for j in range(6):
 lo,hi=offsets[j:j+2];clocks.extend(1-np.r_[0,np.cumsum(counts[lo:hi-1])]/plan0['primary_cases'][j]['step_cap'])
anchors=torch.tensor(np.column_stack([states,np.asarray(clocks,dtype=np.float32)]));targets=torch.tensor(actions)
rows=[]
for name in ['t177_frontier_onpolicy','t178_exploration_time']:
 root=REPO/'stage/runs'/name;plan=json.loads((REPO/'stage/experiments'/name/'protocol.json').read_text());summary=json.loads((root/'summary.json').read_text())
 for arm,spec in summary['training'].items():
  assert C.sha(spec['checkpoint'])==spec['sha256']
  final=torch.load(spec['checkpoint'],weights_only=False,map_location='cpu')['policy']
  for when,number,weights in [('initial',0,initial),('final',5,final)]:
   policy=C.Policy();policy.load_state_dict(weights)
   if when=='initial':
    with torch.no_grad():policy.logstd.copy_(torch.log(torch.tensor(plan['initial_std'])))
   with np.load(root/arm/f'batch_{number}.npz') as z:x={k:torch.from_numpy(z[k].copy()) for k in z.files}
   rho=math.exp(-.2/(plan['arms'][arm].get('noise_correlation_s',1.)))
   mean=policy(x['states'])+rho*(x['previous_latents']-policy(x['previous_states']))*x['previous_mask'][:,None]
   std=policy.logstd.clamp(-1.6,0).exp()[None,:]*torch.where(x['previous_mask'][:,None]>0,math.sqrt(1-rho*rho),1.)
   dist=torch.distributions.Normal(mean,std);delta=dist.log_prob(x['latents']).sum(-1)-x['old_logp'];ratio=delta.exp()
   if when=='initial':assert float(delta.detach().abs().max())<2e-4
   rl=(-torch.minimum(ratio*x['advantages'],ratio.clamp(.9,1.1)*x['advantages'])*x['policy_weights']).mean()
   anchor_error=(policy(anchors)-targets).square();ret=10*torch.stack([anchor_error[offsets[j]:offsets[j+1]].mean() for j in range(6)]).mean()
   entropy=-.001*dist.entropy().sum(-1).mean()
   params=list(policy.named_parameters())
   def grad(loss):
    gg=torch.autograd.grad(loss,[v for _,v in params],retain_graph=True,allow_unused=True);data=[]
    for (n,p),g in zip(params,gg):
     g=torch.zeros_like(p) if g is None else g.clone()
     if n=='body.net.0.weight':g[:,[57,58,74,75,86,87,98,99,110,111]]=0
     data.append(g.flatten())
    return torch.cat(data)
   g,h,e=grad(rl),grad(ret),grad(entropy);ng=float(g.norm());nh=float(h.norm());cos=float(torch.dot(g,h)/(g.norm()*h.norm()+1e-30))
   with torch.no_grad():
    raw=policy.body(x['states']);native=policy(x['states']);init=C.Policy();init.load_state_dict(initial);d=(native-init(x['states'])).abs()
    stats={'experiment':name,'arm':arm,'point':when,'batch':number,'n_decisions':len(raw),'rl_gradient_norm':ng,'weighted_retention_gradient_norm':nh,
     'retention_to_rl_norm':nh/max(ng,1e-30),'rl_retention_gradient_cosine':cos,'entropy_gradient_norm':float(e.norm()),
     'tanh_saturation_fraction_abs_gt099':(raw.abs()>.99).float().mean(0).tolist(),'tanh_derivative_median':(1-raw.square()).median(0).values.tolist(),
     'mean_native_action_change_vs_initial':d.mean(0).tolist(),'max_native_action_change_vs_initial':d.max(0).values.tolist(),
     'anchor_native_action_rmse':float(anchor_error.mean().sqrt()),'advantage_std':float(x['advantages'].std()),'advantage_abs_mean':float(x['advantages'].abs().mean()),
     'full_batch_rl_loss':float(rl),'weighted_retention_loss':float(ret),'max_behavior_logp_difference':float(delta.abs().max())}
   rows.append(stats)
   assert all(torch.equal(v,weights[k] if not(k=='logstd' and when=='initial') else torch.log(torch.tensor(plan['initial_std']))) for k,v in policy.state_dict().items())
C.save(OUT/'gradient_audit.json',{'state':'COMPLETE','optimizer_updates':0,'native_physics_steps':0,'rows':rows,
 'scope':'Full recorded first batch at exact behavior initialization and last batch at post-update final weights; not historical minibatch-gradient telemetry. Retention expectation balanced over all six anchor courses; gradient norms alone do not prove causal dominance. All weights and input data unchanged.'})
print(json.dumps(rows,indent=2))
