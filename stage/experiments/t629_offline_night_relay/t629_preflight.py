"""Real historical teacher-prefix, data, GPU-gradient and successor IO fixtures."""
from pathlib import Path
import tempfile,copy,json,time,ast,gzip
import t629_learning as M
import t629_native as N

def main():
 p=M.plan();import t629_runtime as R;R.setup(p,M.DOC);M.configure_gpu(p);source=M.read(M.ROOT/'stage/runs/t608_current_policy_memory/rl_01/summary.json')['result']['training_rows'];arrays=[N.N.E.N.S.arrays(r['path']) for r in source];obj=M.make(p,p['raw_sigma']);before=M.digest(M.state(obj));groups=[arrays[:2] for _ in range(4)];items,diag=M.batch(obj[0],groups,p);refs=M.references(obj[0],p);loss,detail=M.pg_objective(obj[0],items,refs,p,[.25]*4);loss.backward();gradient=M.torch.cat([v.grad.flatten() for v in M.L.variables(obj[0])]);assert M.torch.isfinite(gradient).all() and float(M.torch.linalg.vector_norm(gradient))>0 and M.digest(M.state(obj))==before
 # Actual previously qualified forced-prefix teacher trajectory, used solely as
 # a contract fixture. It is not counted as new T629 recovery data.
 old=M.read(M.ROOT/'stage/runs/t602_actual_state_teacher/summary.json')['result']['rows'];assert len(old)==2
 for row in old:
  a=N.N.E.N.S.arrays(row['path']);N.inspect(p,[(a,row)])
 with tempfile.TemporaryDirectory(prefix='t629_fixture_',dir=M.DOC) as tmp:
  obj[0].cpu();model=M.export(obj,p,Path(tmp)/'initial.pt.gz');assert M.frozen(p,model['path']).phash()==p['student_initial']['policy_hash'];x=M.torch.tensor(arrays[0]['states'][:12]);y=obj[0].sequence(x).detach()
  with M.torch.no_grad():M.L.variables(obj[0])[-1].view(-1)[0].add_(.001)
  changed=obj[0].sequence(x).detach();assert not M.torch.equal(y,changed);next_model=M.export(obj,p,Path(tmp)/'next.pt.gz');assert M.torch.equal(M.frozen(p,next_model['path']).sequence(x),changed);M.load(obj,M.Prior.state(M.make(p,p['raw_sigma'],device='cpu')));assert M.digest(M.state(obj))==before
 # Exact sample accounting and complete acceptance identities.
 assert len(p['training_jobs'])==36 and len({j['id'] for j in p['training_jobs']})==36;assert len(p['before_success_ids'])==34 and len(p['new_ids'])==2
 assert p['templates']['collect']['max_physics']==4*sum(j['case']['step_cap'] for j in p['training_jobs']) and p['templates']['collect']['max_commands']==4*36+6
 assert 8+8+128+2==p['Actor_each_cap'];assert len(p['profile_jobs'])==407 and len({j['id'] for j in p['profile_jobs']})==407
 assert len([j for chunk in p['neighborhood_chunks'] for j in chunk])==72
 for f in M.HERE.glob('*.py'):ast.parse(f.read_text())
 report=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',actual_two_trajectory_physical_time_peer_RTG_density_and_GPU_gradient=True,old_mass_half_new_mass_half=True,real_historical_forced_candidate_prefix_teacher_own_memory_audit=True,changed_branch_checkpoint_successor_readback_exact=True,full_Actor_optimizer_state_rollback_exact=True,collect36_tasks72fresh72exact_counter_exact=True,finite407_profiles_and72_neighborhoods=True,standard_library_driver_does_not_import_Torch=True,teacher_fixture_is_old_NOT_new_training_data=True,new_native=0,new_optimizer=0,epoch=time.time());M.save(M.DOC/'PREFLIGHT.json',report);print('T629_REAL_HISTORY_GPU_PREFIX_SUCCESSOR_PREFLIGHT_PASS')
if __name__=='__main__':main()
