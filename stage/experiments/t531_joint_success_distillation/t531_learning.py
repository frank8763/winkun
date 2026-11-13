"""Joint distillation of twenty verified successful causal histories."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t529_bidirectional_recovery'));import t529_learning as B
G,P,A,torch,np,digest,sha=B.G,B.P,B.A,B.torch,B.np,B.digest,B.sha
make,variables=B.make,B.variables
def references(p):return B.references(p,p['new_control_row'])
def fit(obj,p,refs,guard,event,steps=None):
 q,v,ao,vo=obj;vh=digest(v.state_dict());voh=digest(vo.state_dict());previous=[int(ao.state[z]['step']) for z in variables(q)];trace=[]
 for k in range(p['epochs'] if steps is None else steps):
  guard();ao.zero_grad(set_to_none=True);loss=B.preserve(q,refs);norm=float(torch.nn.utils.clip_grad_norm_(variables(q),1.,error_if_nonfinite=True));event('BEFORE',k);ao.step();event('AFTER',k)
  assert digest(v.state_dict())==vh and digest(vo.state_dict())==voh;trace.append(dict(epoch=k,three_family_success_distillation_loss=loss,gradient_norm=norm))
 n=len(trace);assert [int(ao.state[z]['step']) for z in variables(q)]==[s+n for s in previous]
 return dict(Actor_Adam=n,Value_Adam=0,new_policy_gradient_updates=0,trace=trace,full_history_BPTT=True,all_targets_from_actual_successful_source_policy_locations=True,three_families_equal_weight=True,recovery_teacher_queries=0)
