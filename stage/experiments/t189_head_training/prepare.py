"""Declare the paired experiment and verify the new common initialization."""
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
from common import *

def main():
    assert not OUT.exists(), 'No duplicate preparation'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    prior=json.loads((REPO/'stage/experiments/t187_local_robustness/protocol.json').read_text())
    parent=REPO/'stage/runs/t187_local_robustness/fixed_source/checkpoint_latest.pt'
    assert sha(parent)=='dfeed77724ee1bc921d28d3dc9ac06c5d9642fdf6ac66da53d61c03f4ae0d875'
    state=torch.load(parent,map_location='cpu',weights_only=False)['policy']
    old=LegacyPolicy().eval();old.load_state_dict(state)
    new=Policy().eval();new.load_parent(state)
    with np.load(prior['retention_demonstrations']) as z:
        states=np.column_stack([z['states'],np.ones(len(z['states']),dtype=np.float32)])
    arrays=[states]
    for path in sorted((parent.parent).glob('batch_*.npz')):
        with np.load(path) as z:arrays.append(z['states'])
    x=torch.tensor(np.concatenate(arrays)[:4096],dtype=torch.float32)
    with torch.no_grad():error=max(float((old(v)-new(v)).abs().max()) for v in x.split(64))
    assert error==0
    for k,v in state.items():assert torch.equal(new.state_dict()[k],v),k
    OUT.mkdir();checkpoint=OUT/'initial_policy.pt'
    torch.save({'policy':new.state_dict(),'parent_sha256':sha(parent),'architecture':'DirectFeaturePolicy','optimizer_updates':0},checkpoint)
    rng=np.random.default_rng(2026103189)
    evaluation=[r for r in prior['evaluation_resets'] if r['reset_group']=='development_grid']
    for side in ['LEFT','RIGHT']:
        for i in range(12):
            dx,dy=rng.uniform(-.03,.03,2);yaw=float(rng.uniform(-2,2))
            evaluation.append({'id':f'new_holdout_{side}_{i}','snapshot_side':side,'delta_x_m':float(dx),'delta_y_m':float(dy),'delta_yaw_deg':yaw,'reset_group':'fresh_holdout'})
    sources=[{'id':f'local_{side}','snapshot_side':side,'delta_x_m':0.,'delta_y_m':0.,'delta_yaw_deg':0.,'reset_group':'near_goal'} for side in ['LEFT','RIGHT']]
    sources.extend({'id':c['id'],'case':c,'reset_group':'original_5m'} for c in prior['primary_cases'][-2:])
    plan={k:prior[k] for k in ['primary_cases','anchor_sources','anchor_step_cap','retention_demonstrations','initial_std','batches_per_arm','episodes_per_arm','episodes_per_batch','max_physics_per_arm','wall_seconds','objective','algorithm','max_evaluation_physics']}
    plan.update(experiment='T189',seed=2026103089,parent_checkpoint=str(parent),initial_policy_checkpoint=str(checkpoint),
                arms={a:{'retention_weight':10.,'noise_correlation_s':1.,'direct_head_trainable':a=='trainable_head'} for a in ['frozen_head','trainable_head']},
                training_pairs=[sources for _ in range(8)],evaluation_resets=evaluation,
                controlled_difference='Only zero-initialized 600-to-2 direct feature head is frozen or trainable; body/native head/logstd trained in both. Common T187 fixed-source parent; same seeds, resets, reward, PPO and physical episode caps. Direct/native head LR1e-4, body1e-5, logstd1e-3.',
                evaluation='Initial old6 gate, then final initial+two candidates on old6/original5m2/consumed54 local development/fresh24 perturbations fixed with seed2026103189. No mid-run model selection. Original 0.20m radius and caps unchanged.',
                scope='One paired pilot seed. Half near-goal and half original5m episodes in each batch. Parent T187 is research-only. A trained new head is not automatically better; no claim of generalization or innovation and no automatic promotion. No map/planner/waypoints/runtime teacher. Zero drive does not imply active braking.',
                references=['https://arxiv.org/abs/1707.06347','https://proceedings.mlr.press/v78/florensa17a.html'])
    files=[parent,checkpoint,Path(plan['retention_demonstrations'])]+list(HERE.glob('*.py'))+[Path(v['snapshot_path']) for v in plan['anchor_sources'].values()]
    plan['input_hashes']={str(f):sha(f) for f in files}
    save(HERE/'protocol.json',plan);save(OUT/'protocol.json',plan)
    save(OUT/'initialization.json',{'state':'PASS','parent_sha256':sha(parent),'checkpoint_sha256':sha(checkpoint),'states_checked':len(x),'max_output_error':error,'original_parent_parameters_unchanged':True,'direct_head_zero':True,'optimizer_steps':0,'physics_steps':0})
    print(json.dumps({'state':'PASS','protocol_sha256':sha(HERE/'protocol.json'),'max_output_error':error,'states':len(x)}))

if __name__=='__main__':main()
