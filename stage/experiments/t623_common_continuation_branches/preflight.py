from pathlib import Path
import io,gzip,time,ast,copy,unittest.mock as mock
import t623_runtime as M
L,torch,np=M.L,M.torch,M.np
def main():
 p=M.plan();M.setup();torch.set_num_threads(2);a=M.R.R.E.N.S.arrays(p['prefix_source']);q=L.frozen(p,p['student_initial']['path']);k=p['branch_decision'];h=None
 with torch.no_grad():
  for x in a['states'][:k+1]:_,loc,h,_=q.step(torch.tensor(x[None]),h)
 action=loc[0].clamp(-1,1).float().double();state=torch.load(io.BytesIO(gzip.decompress(Path(p['critic_checkpoint']).read_bytes())),map_location='cpu',weights_only=False)['state'];critics=torch.nn.ModuleList([M.R.L.Critic(p),M.R.L.Critic(p)]);critics.load_state_dict(state['critics']);x=torch.tensor(a['states'][:k+1,None]);pred=[]
 for delta in [0.,.1,-.1]:
  u=action.clone();u[1]=(loc[0,1]+delta).clamp(-1,1).float().double();u.requires_grad_(True)
  values=[c.values(c.features(x)[-1,0],u) for c in critics];mean=torch.stack(values).mean();grad=torch.autograd.grad(mean,u)[0];pred.append(dict(delta=delta,action=u.detach().tolist(),critic_values=[float(z) for z in values],mean=float(mean),action_gradient=grad.tolist()))
 # Actual Session.step boundary delegates the full native loop, here captured
 # without advancing physics; verify recorded prefix, one action and suffix.
 checks=[]
 for delta in [0.,.1,-.1]:
  for decision in [k-1,k,k+1]:
   q=L.frozen(p,p['student_initial']['path']);q.sample=False;q.reset_noise();hh=None
   with torch.no_grad():
    for xx in a['states'][:decision]:_,_,hh,_=q.step(torch.tensor(xx[None]),hh)
   s=object.__new__(M.N.Session);s.q=q;s.rows=[None]*decision;s.hidden=hh;s.source=a;s.k=k;s.delta=delta;s.packet=lambda:a['states'][decision].copy();capt=[]
   def capture(session,forced=None,owner=0,actor_active=True):
    with torch.no_grad():act,loc,hh,_=session.q.step(torch.tensor(session.packet()[None]),session.hidden)
    expected=(loc[0].clone());expected=expected.clamp(-1,1).float()
    if decision==k:expected[1]=(loc[0,1]+delta).clamp(-1,1).float()
    assert torch.equal(act[0],expected)
    assert owner==int(decision<k or decision==k and delta!=0)
    if decision<k:assert np.array_equal(forced,a['actions'][decision])
    else:assert forced is None
    capt.append(True)
   with mock.patch.object(M.N.F.FieldSession,'step',new=capture):M.N.Session.step(s)
   assert capt;checks.append([delta,decision])
 buf=io.BytesIO();np.savez(buf,**M.R.R.E.N.S.packed(a));n=len(a['states']);upper=int(len(buf.getvalue())*1.001*536/n)+32768;assert upper<p['full_bytes']
 for f in M.HERE.glob('*.py'):ast.parse(f.read_text())
 L.save(M.DOC/'PREDICTIONS.json',dict(state='SEALED_BEFORE_NATIVE',predictions=pred,ranking_descending=[r['delta'] for r in sorted(pred,key=lambda z:-z['mean'])],critic_is_mixed_behavior_MC_not_policy_Q=True,prefix_SHA=p['prefix_source_sha256'],critic_SHA=L.sha(p['critic_checkpoint']),epoch=time.time()))
 L.save(M.DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',actual_Session_boundary_combinations=checks,three_parallel_workers=True,full_file_safe_upper=upper,full_file_cap=p['full_bytes'],prediction_before_any_native=True,new_native=0,new_optimizer=0));print('T623_BOUNDARY_PREDICTION_STORAGE_PREFLIGHT_PASS',upper)
if __name__=='__main__':main()
