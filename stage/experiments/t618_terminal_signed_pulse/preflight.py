"""Actual Session.step and result-receive regression without native or Adam."""
from pathlib import Path
import ast,sys,io,time,types,unittest.mock as mock
import t618_runtime as M
N=M.Native;R=M.R;torch,np=R.torch,R.np

def main():
 p=M.plan();M.setup();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 for f in M.HERE.glob('*.py'):ast.parse(f.read_text())
 a=R.E.N.S.arrays(p['candidate_prefix_source']);k=p['teacher_first'];checks=[]
 q=R.L.frozen(p,p['candidate_model']['path']);ph=q.phash();h=None;q.sample=False;q.reset_noise()
 with torch.no_grad():
  for x in a['states'][:k]:_,_,h,_=q.step(torch.tensor(x[None]),h)
 # Each real Session wrapper calls the actual Actor exactly once. Native stepping
 # alone is replaced; recorded source state and three memory carries are real.
 for delta in [0.,.25,-.25]:
  actor=R.L.frozen(p,p['candidate_model']['path']);actor.sample=False;actor.reset_noise();hidden=None
  with torch.no_grad():
   for x in a['states'][:k]:_,_,hidden,_=actor.step(torch.tensor(x[None]),hidden)
  initial=R.L.digest(actor.state_dict());session=object.__new__(N.Session);session.rows=[None]*k;session.q=actor;session.hidden=hidden;session.start=k;session.duration=30;session.delta=delta;session.source=a;session.packet=lambda:a['states'][k].copy();called=[]
  def no_native(s,forced=None,owner=0,actor_active=True):
   with torch.no_grad():action,loc,new,_=s.q.step(torch.tensor(s.packet()[None]),s.hidden)
   called.append(owner);expected=loc.clamp(-1,1).float();expected[0,1]=(loc[0,1]+delta).clamp(-1,1).float();assert torch.equal(action,expected);assert float(action[0,0])==float(loc[0,0].clamp(-1,1).float());return action
  original=actor.step
  with mock.patch.object(N.F.FieldSession,'step',new=no_native):N.Session.step(session)
  assert called==[int(delta!=0)] and len(actor.tape)==k+1 and actor.step==original and R.L.digest(actor.state_dict())==initial
  checks.append(dict(delta=delta,real_actor_step_called_once=True,full_prefix_memory_real=True,drive_unchanged=True,step_method_restored=True))
 # Original T617 failed because no native result carries a request yet. Exercise
 # the fixed receive with exactly that result shape and both worker indexes.
 for i in [0,1]:
  obj=object.__new__(N.Pool);obj.p=p;obj.serial=1;obj.pending_requests=[dict(model=p['candidate_model']['path'],job=j) for j in p['pulse_jobs'][:2]];saved=[]
  with mock.patch.object(N.Previous.Pool,'receive',return_value=(a,dict(policy_hash=ph))),mock.patch.object(N.L,'save',side_effect=lambda f,j:saved.append(j)):
   _,r=N.Pool.receive(obj,i)
  assert 'request' not in r and len(saved)==1 and 'native_result' in saved[0] and r['signed_steering_delta']==p['pulse_jobs'][i]['steering_delta']
 # Both saturation directions and exact no-pulse drive behavior use real formula.
 for value in [-2.,-.2,0.,.9,2.]:
  loc=torch.tensor([[.3,value]],dtype=torch.float64);action=loc.clamp(-1,1).float()
  for delta in [-.25,0.,.25]:assert float(N.pulse(action,loc,delta)[0,1])==float((loc[0,1]+delta).clamp(-1,1).float())
 n=len(a['states']);Nmax=p['job']['case']['step_cap']//100+2;big={key:np.zeros((Nmax+(len(v)-n),*v.shape[1:]),dtype=v.dtype) if v.ndim and len(v) in [n,n+1] else v for key,v in a.items()};big.update(pulse_start_decision=np.asarray(k,np.int64),pulse_duration_decisions=np.asarray(30,np.int64),pulse_steering_delta=np.asarray(.25));buf=io.BytesIO();np.savez(buf,**big);assert len(buf.getvalue())*1.001+32768<p['full_bytes']
 result=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',real_Session_step_seam=checks,real_native_return_without_request_regression=True,original_result_persisted_before_annotation=True,saturation_both_signs=True,full_worst_uncompressed_bytes=len(buf.getvalue()),model_unchanged=q.phash()==ph,new_native=0,new_optimizer=0,epoch=time.time());M.save(M.DOC/'PREFLIGHT.json',result);print('T618_REAL_STEP_AND_NATIVE_RETURN_CONTRACT_PASS',result)

if __name__=='__main__':main()
