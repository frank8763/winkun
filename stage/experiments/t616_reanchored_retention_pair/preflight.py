"""Real histories and analytic first displacement; zero optimizer/native calls."""
from pathlib import Path
import time,sys,ast,copy,io
import t616_learning as L
ROOT=L.ROOT;HERE=Path(__file__).resolve().parent;DOC=ROOT/'stage/docs/day_20261117/t616'

def main():
 for f in HERE.glob('*.py'):ast.parse(f.read_text())
 p=L.read(HERE/'protocol.json');cfg=L.read(p['actor_configuration']);cfg.update(p);cfg['arm']='command';p=cfg
 L.torch.set_num_threads(1);L.torch.use_deterministic_algorithms(True)
 assert not L.torch.cuda.is_initialized();data=L.dataset(p);q=L.frozen(p,p['student_initial']['path']);q.requires_grad_(False);q.branch.requires_grad_(True);before=L.digest(q.state_dict());packed=L.torch.nn.utils.rnn.pad_sequence([g['n'] for g in data]);checks=[]
 with L.torch.no_grad():
  oldmax=L.old_peak(q,data,packed);assert oldmax<1e-12
  target=next(g for g in data if g['kind']=='new_task');teacher=L.frozen(p,p['base_model']['path']);err=float((teacher.sequence(target['x'])-target['target']).abs().max());assert err<1e-12
  row=next(r for r in p['keep_rows'] if r['job_id']==p['primary_new_task']);assert row['success'] and not row['collision'];a=L.G.E.N.S.arrays(row['path']);assert L.np.array_equal(target['target'].clamp(-1,1).float().numpy(),a['actions'])
 loss,parts=L.objective(q,data,packed,p);loss.backward();L.torch.nn.utils.clip_grad_norm_(L.variables(q),1.,error_if_nonfinite=True)
 # Exact fresh first-Adam bias correction in closed form; no optimizer is created or stepped.
 delta=L.Previous.flat([-p['fit_lr']*v.grad/(v.grad.abs()+1e-8) for v in L.variables(q)]).detach();vs=L.variables(q);weights=[v.detach().clone() for v in vs]
 for method in ['reanchor','pointwise']:
  pc=dict(p,mechanism=method);start=time.monotonic();d,info=L.feasible_displacement(q,data,delta,pc,lambda:None);trials=[]
  for factor in p['step_factors']:
   with L.torch.no_grad():
    for v,w,part in zip(vs,weights,L.Previous.unflat(d,vs)):v.copy_(w+factor*part)
    nxt,detail=L.objective(q,data,packed,pc);raw,cmd=L.protection(q,data,packed);old=L.old_peak(q,data,packed)
   okay=info['converged'] and info['projected_L2']>1e-14 and float(nxt)<=float(loss.detach())+1e-12 and cmd<=p['protected_max_change'] and (method=='reanchor' or old<=p['old_command_bound'])
   trials.append(dict(factor=factor,objective_delta=float(nxt)-float(loss.detach()),recovery_command_max=cmd,old_command_max=old,passes_all_gates=okay))
  with L.torch.no_grad():
   for v,w in zip(vs,weights):v.copy_(w)
  assert L.digest(q.state_dict())==before
  checks.append(dict(mechanism=method,seconds=time.monotonic()-start,projection=info,trials=trials,has_feasible_first_analytic_candidate=any(t['passes_all_gates'] for t in trials)));print(method,'analytic_feasible',checks[-1]['has_feasible_first_analytic_candidate'],flush=True)
 L.save(DOC/'ANALYTIC_FEASIBILITY.json',dict(checks=checks,new_native=0,new_optimizer=0,gradient_evaluations=1,models_unchanged=True))
 assert all(z['has_feasible_first_analytic_candidate'] for z in checks)
 # The actual general projection function handles both ordinary and zero-radius boundaries.
 fn=L.Previous.project_halfspaces;torch=L.torch
 for rhs in [.1,0.]:
  x,d=fn(torch.ones(2,dtype=torch.float64),torch.eye(2,dtype=torch.float64),torch.full((2,),rhs,dtype=torch.float64),128,1e-10);assert d['converged'] and float(x.max())<=rhs+1e-10
 # Bound worst full recording with the real packed schema and longest fixed evaluation job.
 a=L.G.E.N.S.arrays(p['current_success_rows'][0]['path']);n=len(a['states']);N=max(j['case']['step_cap']//100+2 for j in p['evaluation_jobs']);big={k:L.np.zeros((N+(len(v)-n),*v.shape[1:]),dtype=v.dtype) if v.ndim and len(v) in [n,n+1] else v for k,v in a.items()};buf=io.BytesIO();L.np.savez(buf,**big);assert len(buf.getvalue())<=p['full_bytes']
 report=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',new_native=0,new_optimizer=0,analytic_gradient_evaluations=1,all_45_histories_qualified=True,old_40_anchor_initial_max=oldmax,new_target_teacher_raw_sequence_error=err,initial_objective=float(loss.detach()),initial_objective_parts=parts,models_unchanged=before==L.digest(q.state_dict()),checks=checks,uncompressed_worst_full_bytes=len(buf.getvalue()),full_cap=p['full_bytes'],not_actual_learning_or_navigation_gain=True,epoch=time.time());L.save(DOC/'PREFLIGHT.json',report)
 print('T616_REAL_DATA_ZERO_UPDATE_PREFLIGHT_PASS',flush=True)

if __name__=='__main__':main()
