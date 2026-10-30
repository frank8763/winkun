"""Lock a bounded, reset-only adaptive curriculum before observing its outcomes."""
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
from common import *

def main():
    assert not OUT.exists()
    prior=json.loads((REPO/'stage/experiments/t189_head_training/protocol.json').read_text())
    completed=json.loads((REPO/'stage/runs/t189_head_training/summary.json').read_text())
    assert completed['state']=='COMPLETE'
    assert completed['scores']['trainable_head']['old_3m_4m']['safe_success']==6
    parent=Path(completed['training']['trainable_head']['checkpoint'])
    assert sha(parent)==completed['training']['trainable_head']['sha256']
    plan={k:prior[k] for k in ['primary_cases','anchor_sources','anchor_step_cap','retention_demonstrations','initial_std','objective','algorithm']}
    plan.update(experiment='T190',seed=2026103090,parent_checkpoint=str(parent),rounds=4,batches_per_round=8,episodes_per_batch=8,
                max_training_physics=4*64*23334,max_probe_physics=4*74*23334,max_final_physics=94*23334,wall_seconds=1800,
                proposal_per_side=16,proposal_radius_m=[.35,1.05],proposal_yaw_deg=50.,max_pool_per_side=4,
                protocol='Training-only reset proposals: seeded isotropic x/y offsets and yaw offsets around observed successful reset poses. Never change goal/geometry/radius/time limit or pose after stepping starts. Invalid proposals are recorded, not resampled. Pool is unordered training starts, never runtime waypoints.',
                selection='After each round, retain root, two successful resets closest in Euclidean position to original start, and one farthest from those for diversity. Uses start coordinates only to choose next reset distribution; no map queries, route edges or runtime planner. New proposals are evaluated before and after training. Paired gains/losses are development evidence.',
                training='Each of 8 batches has independent paired episodes from one valid failed proposal per side (or valid proposal when no failure), one previously successful reset alternating side, and one original5m alternating side. All actors single neural policies. Fresh optimizer; same AR MC/PPO and retention10 as T189. No Critic, replay value target or intermediate goals.',
                stop='Stop after four rounds, or old6 regression, or old6+original5m2 all safe success. Do not silently rollback or pick an intermediate best model. Preserve final and all failures; T159 remains incumbent.',
                scope='Exploratory adaptive curriculum from a selected research parent, not a controlled superiority study. Reset selection and task mix differ from T189. Fixed original5m tasks are training targets, not generalization. No novelty claim. Maximum256 complete training episodes, never extend automatically.',
                references=['https://proceedings.mlr.press/v78/florensa17a.html','https://arxiv.org/abs/1707.06347'])
    rng=np.random.default_rng(2026103190)
    evaluation=[r for r in prior['evaluation_resets'] if r['reset_group']=='development_grid']
    for side in ['LEFT','RIGHT']:
        for i in range(12):
            dx,dy=rng.uniform(-.03,.03,2);yaw=float(rng.uniform(-2,2))
            evaluation.append({'id':f'fresh_T190_{side}_{i}','snapshot_side':side,'delta_x_m':float(dx),'delta_y_m':float(dy),'delta_yaw_deg':yaw,'reset_group':'fresh_holdout'})
    plan['evaluation_resets']=evaluation
    files=[parent,REPO/'stage/runs/t189_head_training/summary.json',Path(plan['retention_demonstrations'])]+list(HERE.glob('*.py'))+[Path(v['snapshot_path']) for v in plan['anchor_sources'].values()]
    plan['input_hashes']={str(f):sha(f) for f in files}
    OUT.mkdir();save(HERE/'protocol.json',plan);save(OUT/'protocol.json',plan)
    print(json.dumps({'state':'PREPARED','protocol_sha256':sha(HERE/'protocol.json'),'training_physics_cap':plan['max_training_physics']}))

if __name__=='__main__':main()
