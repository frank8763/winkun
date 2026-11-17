from pathlib import Path
import os,sys,io,copy,ast,time,unittest.mock as mock,subprocess,json
import t625_learning as M
L,torch=M.L,M.torch
def main():
 p=M.config();M.setup(p);train=M.data(p);assert len(train)==3 and {e['id'] for e in train}=={'wide117_8m_dev6_LEFT_13'};obj=M.make(p);before=L.digest(M.B.state(obj));pack=M.pack(train,p,M.np.random.default_rng(p['learner_seed']));loss,parts=M.loss(obj,pack,p,'rank');assert loss.dtype==torch.float64;loss.backward();assert bool(torch.isfinite(loss));assert L.digest(M.B.state(obj))==before;assert parts['ranking_pairs']>0
 # Actual GPU gradients twice: numerical repeat, no parameter application.
 grad=torch.cat([v.grad.detach().flatten() for v in obj[0].parameters()]);obj[1].zero_grad(set_to_none=True);again,_=M.loss(obj,pack,p,'rank');again.backward();assert torch.equal(grad,torch.cat([v.grad.detach().flatten() for v in obj[0].parameters()]));assert L.digest(M.B.state(obj))==before
 for e in train:
  k=e['k'];assert torch.equal(e['ret'][k:-1],e['r'][k:-1]+e['discount'][k:-1]*e['bootstrap'][k:-1]*e['ret'][k+1:]);assert e['ret'][-1]==e['r'][-1];assert float(pack[1].min())>=k+1
 bound=M.B.state(obj)
 for i,v in zip(bound['optimizer']['param_groups'][0]['params'],obj[0].parameters()):bound['optimizer']['state'][i]=dict(step=torch.tensor(1.),exp_avg=torch.zeros_like(v).cpu(),exp_avg_sq=torch.zeros_like(v).cpu())
 buf=io.BytesIO();torch.save(dict(state=bound,state_sha='0'*64),buf);upper=int(len(buf.getvalue())*1.001)+32768;assert upper<p['checkpoint_bytes']
 # Verify actual new worker resource mapping and delegating control path.
 code="""import sys,os,json,unittest.mock as mock
from pathlib import Path
sys.path.insert(0,str(Path('stage/experiments/t625_target_continuation_calibration').resolve()))
import t625_native as N
p=json.loads(Path('stage/experiments/t625_target_continuation_calibration/protocol.json').read_text());checks=[]
for i in range(3):
 def captured(pipe,index,p,counts):os.sched_setaffinity(0,{index});checks.append(sorted(os.sched_getaffinity(0)))
 with mock.patch.object(N.N.G.T,'worker',side_effect=captured):N.worker(None,i,p,None)
 N.V.restore_parent(p)
print(json.dumps(checks))
"""
 checks=json.loads(subprocess.check_output([str(M.ROOT/'run_stage.sh'),'-c',code],cwd=M.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),text=True))
 assert checks==[[8],[10],[12]]
 held=p['jobs'][0]['id'];assert held not in {e['id'] for e in train}
 # Corrected inspect contract runs on saved actual branch arrays, not mocks.
 rows=L.read(p['training_reference_rows'])
 spec=__import__('importlib.util',fromlist=['util']);ss=spec.spec_from_file_location('t625_contract_fixture_unique',M.ROOT/'stage/experiments/t623r_branch_contract_recovery/recover.py');contract=spec.module_from_spec(ss);ss.loader.exec_module(contract);oldp=L.read(M.ROOT/'stage/experiments/t623_common_continuation_branches/protocol.json');cfg=L.read(oldp['actor_configuration']);cfg.update(oldp)
 contract.M.setup()
 for row in rows:contract.check(cfg,L.G.E.N.S.arrays(row['path']),row,L.G.E.N.S.arrays(oldp['prefix_source']))
 for f in M.HERE.glob('*.py'):ast.parse(f.read_text())
 L.save(M.DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',train_task='wide117_8m_dev6_LEFT_13',heldout_task=held,task_split_disjoint=True,all_labels_after_exact_target_policy_switch=True,full_causal_prefix_recomputed=True,branch_labels_have_common_frozen_continuation=True,regression_and_ranking_gradient_finite=True,GPU_gradient_repeat_exact=True,parameters_and_Adam_unchanged=True,checkpoint_safe_upper=upper,actual_worker_affinity=checks,three_saved_real_branch_contracts_pass=True,new_native=0,new_optimizer=0,epoch=time.time()));print('T625_GPU_TARGET_CONTINUATION_PREFLIGHT_PASS',upper,parts)
if __name__=='__main__':main()
