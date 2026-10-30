"""No optimizer or native physics steps: head bounds and function-preserving audit."""
from pathlib import Path
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import json,hashlib,numpy as np,torch
from policy import C,DirectFeaturePolicy
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=ROOT/'stage/runs/t188_direct_action_head'

def main():
 assert not (OUT/'summary.json').exists()
 plan=json.loads((HERE/'protocol.json').read_text())
 for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 bounds=[]
 for name,path in plan['bound_checkpoints'].items():
  state=torch.load(path,weights_only=False,map_location='cpu')['policy'];w=state['native_head.weight'];b=state['native_head.bias'];lo=b-w.abs().sum(1);hi=b+w.abs().sum(1)
  native_min=float(lo[0].clamp(-1,1));bounds.append({'model':name,'checkpoint_sha256':C.sha(path),'mean_native_action_box_lower':lo.tolist(),'upper':hi.tolist(),'minimum_wheel_command_lower_bound':5*((native_min+1)/2)**2,'scope':'For fixed saved head parameters and bodyTanh outputs in[-1,1]^2. This is an analytic bound, not an observed minimum or proof head could never learn another range.'})
 parent=torch.load(plan['parent_checkpoint'],weights_only=False,map_location='cpu')['policy'];old=C.Policy().eval();old.load_state_dict(parent);new=DirectFeaturePolicy().eval();new.load_parent(parent)
 states=[]
 with np.load(plan['anchor_states']) as z:states.append(np.column_stack([z['states'],np.ones(len(z['states']),dtype=np.float32)]))
 for path in plan['batch_files']:
  with np.load(path) as z:states.append(z['states'][::max(1,len(z['states'])//256)])
 x=torch.tensor(np.concatenate(states)[:4096],dtype=torch.float32);maximum=0.
 with torch.no_grad():
  for a,b in zip(x.split(64),x.split(64)):maximum=max(maximum,float((old(a)-new(b)).abs().max()))
 assert maximum==0.,maximum
 for k,v in parent.items():assert torch.equal(new.state_dict()[k],v),k
 original_bias=new.direct_head.bias.detach().clone();targets=[]
 for i in [0,len(x)//2,len(x)-1]:
  with torch.no_grad():
   new.direct_head.bias.copy_(original_bias);base=new(x[i:i+1])[0].clone();new.direct_head.bias[0]+=-1-float(base[0]);output=new(x[i:i+1])[0];clipped=C.action_clip(output.numpy(),'full_range')
  assert abs(float(clipped[0])+1)<1e-6
  targets.append({'sample':i,'native_throttle':float(clipped[0]),'wheel_command':5*((float(clipped[0])+1)/2)**2})
 with torch.no_grad():new.direct_head.bias.copy_(original_bias)
 new.zero_grad();loss=(new(x[:16])[:,0]+1).square().mean();loss.backward();grad=float(new.direct_head.weight.grad.norm());assert grad>0 and torch.isfinite(new.direct_head.weight.grad).all()
 assert torch.count_nonzero(new.direct_head.weight)==0 and torch.count_nonzero(new.direct_head.bias)==0
 for k,v in parent.items():assert torch.equal(new.state_dict()[k],v),k
 path=OUT/'initial_direct_policy.pt';torch.save({'policy':new.state_dict(),'parent_sha256':C.sha(plan['parent_checkpoint']),'architecture':'DirectFeaturePolicy','optimizer_updates':0,'initial_max_output_error':maximum},path)
 result={'state':'COMPLETE','bounds':bounds,'input_states':len(x),'initial_max_native_output_error':maximum,'zero_wheel_command_representability_checks':targets,'direct_head_gradient_norm_for_zero_drive_target':grad,'native_physics_steps':0,'optimizer_updates':0,'initial_checkpoint_sha256':C.sha(path),'automatic_promotion':False,'incumbent':'T159','scope':'Architecture/interface audit only. Directhead is internal to one network, zero initialized and unbounded before native action clipping. Shows expressibility and nonzero gradient, not learned stopping, safe braking, 5m success or algorithm novelty.'}
 C.save(OUT/'summary.json',result);C.save(OUT/'status.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
