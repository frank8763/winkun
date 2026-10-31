"""Meaningful causal/reset/gradient checks; no simulator or optimizer steps."""
import os
os.sched_setaffinity(0, {0})
from pathlib import Path
import json,copy,time,hashlib,fcntl
import numpy as np
import torch
from model import TemporalActor, TemporalValue, PredictionHeads, packet, C, PACKET_DIM, HIDDEN
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t209_temporal_actor'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(module):return float(torch.sqrt(sum((p.grad.square().sum() if p.grad is not None else torch.zeros(())) for p in module.parameters())))
def main():
 lock=(OUT/'contracts.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'summary.json').exists(),'No duplicate architecture contract'
 plan=json.loads((OUT/'protocol.json').read_text());start=time.monotonic()
 for p,h in plan['input_hashes'].items():assert sha(p)==h,p
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(2026103909)
 parent=torch.load(plan['parent'],map_location='cpu',weights_only=False)['policy'];actor=TemporalActor(parent,recurrent=True).eval();control=TemporalActor(parent,recurrent=False).eval();control.load_state_dict(actor.state_dict());original={k:v.clone() for k,v in actor.state_dict().items()}
 with np.load(plan['retention_targets']) as z:ret=torch.tensor(z['states']);target=torch.tensor(z['actions'])
 extras=torch.zeros((len(ret),2));rp=packet(ret,extras,extras,torch.zeros(len(ret)))
 with torch.no_grad():error=float((actor.step(rp)['mode']-target).abs().max())
 assert error==0
 with np.load(plan['causal_source']) as z:
  states=torch.tensor(z['states']);steps=z['steps'];previous=np.vstack([np.zeros((1,2),dtype=np.float32),z['actions'][:-1]]);steer=z['steer_before'];delta=np.r_[0.,np.diff(steps)*.002]
 previous=torch.tensor(previous,dtype=torch.float32);assert float((states[:,57:59]-previous).abs().max())<1e-6
 x=packet(states,previous,torch.tensor(steer,dtype=torch.float32),torch.tensor(delta,dtype=torch.float32))[:32,None];reset=torch.zeros(x.shape[:2],dtype=torch.bool);reset[0]=True
 # Native-domain likelihood bypasses the historical inverse-tanh gradient clip.
 row=actor.step(x[:16,0]);action=torch.zeros((16,2));loss=actor.log_prob(row,action).mean();loss.backward();base_grad=norm(actor.base.body.net[:4]);core_initial_grad=norm(actor.core);residual_grad=norm(actor.residual)
 assert base_grad>1e-8 and residual_grad>1e-8 and core_initial_grad==0
 actor.zero_grad(set_to_none=True)
 # A nonzero synthetic head is an analytical witness, never a trained/deployed model.
 witness=copy.deepcopy(actor)
 with torch.no_grad():witness.residual.weight.normal_(0,.03)
 xr=x[:16].clone().requires_grad_(True);out=witness.sequence(xr,reset[:16]);witness.log_prob(out,torch.zeros((16,1,2))).mean().backward();conv_grad=norm(witness.core.encoder.rays);recurrent_grad=norm(witness.core.cell);context_grad=norm(witness.core.encoder.context)
 assert min(conv_grad,recurrent_grad,context_grad)>1e-8
 assert all(torch.equal(original[k],v) for k,v in actor.state_dict().items())
 # Compare streaming and complete-sequence computation with identical weights.
 with torch.no_grad():
  full=witness.sequence(x,reset);hidden=None;stream=[]
  for i in range(len(x)):
   row=witness.step(x[i],hidden,reset[i]);stream.append(row['raw']);hidden=row['hidden']
  stream_error=float((torch.stack(stream)-full['raw']).abs().max());assert stream_error<5e-6
  altered=x.clone();altered[16:,:,:48]=1-altered[16:,:,:48];other=witness.sequence(altered,reset);prefix_error=float((full['raw'][:16]-other['raw'][:16]).abs().max());assert prefix_error==0
  # Same current packet, different earlier measurements; reset removes leakage.
  earlier=x.clone();earlier[:12,:,:48]=1-earlier[:12,:,:48];changed=witness.sequence(earlier,reset)
  memory_effect=float((full['raw'][-1]-changed['raw'][-1]).abs().max());hidden_effect=float((full['hidden']-changed['hidden']).abs().max());assert memory_effect>1e-8 and hidden_effect>1e-8
  off=copy.deepcopy(witness);off.core.recurrent=False;memoryoff_error=float((off.sequence(x,reset)['raw'][-1]-off.sequence(earlier,reset)['raw'][-1]).abs().max());assert memoryoff_error==0
  resets=reset.clone();resets[12]=True;a=witness.sequence(x,resets);b=witness.sequence(earlier,resets);episode_reset_error=float((a['raw'][12:]-b['raw'][12:]).abs().max());assert episode_reset_error==0
  value=TemporalValue(recurrent=True).eval();baseline,hidden=value.sequence(x,reset);assert baseline.shape==x.shape[:2] and bool(torch.isfinite(baseline).all())
  auxiliary=PredictionHeads().eval()(full['hidden']);assert all(v.shape==(1,2) for v in auxiliary.values())
 assert set(map(id,actor.parameters())).isdisjoint(map(id,value.parameters()))
 assert sum(p.numel() for p in actor.parameters())==sum(p.numel() for p in control.parameters())
 assert all(p.requires_grad for name,p in actor.named_parameters() if name!='base.logstd') and not actor.base.logstd.requires_grad
 for p,h in plan['input_hashes'].items():assert sha(p)==h,p
 assert time.monotonic()-start<plan['max_wall_seconds']
 result={'state':'ARCHITECTURE_CONTRACT_PASS','physics_steps':0,'optimizer_updates':0,'online_training_started':False,'navigation_episodes':0,'Actor_total_parameters':sum(p.numel() for p in actor.parameters()),'Actor_trainable_parameters':sum(p.numel() for p in actor.parameters() if p.requires_grad),'memoryoff_parameter_count_equal':True,'initial_old7_max_action_error':error,'initial_parent_feature_gradient_norm':base_grad,'initial_residual_head_gradient_norm':residual_grad,'initial_zero_head_blocks_new_core_gradient_as_expected':core_initial_grad,'synthetic_nonzero_head_gradient_norms':{'ray_CNN':conv_grad,'GRU':recurrent_grad,'context':context_grad},'stream_sequence_raw_max_error':stream_error,'future_to_past_raw_error':prefix_error,'history_effect_with_identical_current_packet':memory_effect,'history_hidden_effect':hidden_effect,'memoryoff_history_error':memoryoff_error,'episode_reset_leak_error':episode_reset_error,'causal_packet_dim':PACKET_DIM,'sequence_decisions_checked':len(x),'decision_step_seconds':.2,'tested_source_span_seconds':float(steps[len(x)-1]-steps[0])*.002,'actor_value_parameter_storage_separate':True,'trained_or_exported_model':False,'auxiliary_heads_enabled_in_policy':False,'elapsed_s':time.monotonic()-start,'scope':'Only implementation and gradient-path checks, using stored real Stage observations/steering and synthetic copies. No navigation success, improved learning, scientific novelty, simulator interaction or full training claimed.'}
 (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(OUT/'status.json').write_text(json.dumps(result,indent=2)+'\n');assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<plan['max_bytes'];assert sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file())<1024**3;print(json.dumps(result,indent=2))
if __name__=='__main__':main()
