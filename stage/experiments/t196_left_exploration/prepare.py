"""Lock LEFT-focused paired exploration, its feature basis and seven-task anchors."""
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
from common import *
from replay import build

def main():
    assert not OUT.exists(),'No duplicate preparation'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    previous=json.loads((REPO/'stage/experiments/t192_sac_her/protocol.json').read_text())
    checkpoint=REPO/'stage/runs/t194_drive_sensitivity/selected_policy.pt';assert sha(checkpoint)=='4848d7bcb38a140eda227975d7c338b5eb8ad655e5ce64bea3f84ac3c89e7656'
    teacher=Policy().eval().requires_grad_(False);teacher.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['policy'])
    with np.load(previous['retention_demonstrations'],allow_pickle=False) as z:
        states=z['states'].copy();counts=z['native_counts'].copy();offsets=z['episode_offsets'].copy()
    groups=[]
    for j in range(6):
        lo,hi=offsets[j:j+2];clock=1-np.r_[0,np.cumsum(counts[lo:hi-1])]/previous['primary_cases'][j]['step_cap'];groups.append(np.column_stack([states[lo:hi],clock]).astype(np.float32))
    right=REPO/'stage/runs/t192_sac_her/episodes/no_her_b4_w7.npz'
    with np.load(right,allow_pickle=False) as z:groups.append(z['states'].copy())
    retention=np.concatenate(groups);retention_offsets=np.r_[0,np.cumsum([len(s) for s in groups])]
    with torch.no_grad():targets=torch.cat([teacher(x) for x in torch.tensor(retention).split(128)]).numpy()
    calibration=[retention]
    left_files=[REPO/f'stage/runs/t192_sac_her/episodes/no_her_b{i}_w2.npz' for i in range(4)]
    for p in left_files:
        with np.load(p,allow_pickle=False) as z:calibration.append(z['states'][::2].copy())
    calibration=np.concatenate(calibration)
    with torch.no_grad():features=torch.cat([teacher.body.net[:4](x) for x in torch.tensor(calibration).split(128)]).double()
    center=features.mean(0);centered=features-center;cov=centered.T@centered/len(features);values,vectors=torch.linalg.eigh(cov)
    assert values[-8]>1e-6
    directions=(vectors[:,-8:]/values[-8:].sqrt()).T
    whitened=(centered@directions.T).numpy();assert np.max(np.abs(whitened.T@whitened/len(whitened)-np.eye(8)))<1e-8
    OUT.mkdir();basis_path=OUT/'parameter_basis.npz';np.savez_compressed(basis_path,center=center.numpy(),directions=directions.numpy(),eigenvalues=values[-8:].numpy())
    targets_path=OUT/'retention_targets.npz';np.savez_compressed(targets_path,states=retention,actions=targets,offsets=retention_offsets)
    rng=np.random.default_rng(2026103196);coefficients=rng.normal(size=(128,9));delta=(whitened@coefficients[:,:8].T+coefficients[:,8])*.1
    rms=np.sqrt(np.square(delta).mean(0));assert np.isfinite(rms).all() and .2<float(rms.mean())<.4
    left=previous['primary_cases'][6];jobs=[{'id':left['id'],'case':left} for _ in range(6)]+[{'id':'near_LEFT','case':left,'snapshot_side':'LEFT'} for _ in range(2)]
    evaluation=[];holdout_rng=np.random.default_rng(2026103296)
    for i in range(12):
        dx,dy=holdout_rng.uniform(-.03,.03,2);yaw=float(holdout_rng.uniform(-2,2));evaluation.append({'id':f'left_holdout_{i}','case':left,'group':'left_fresh_reset','perturbation':[float(dx),float(dy),yaw]})
    seed_episodes=previous['seed_episodes'];mc_errors=[]
    for p in seed_episodes:
        with np.load(p,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        build(a)
    plan={k:previous[k] for k in ['primary_cases','anchor_sources','initial_std','alpha','warmup_updates','updates_per_batch','seed_episodes']}
    plan.update(experiment='T196',seed=2026103096,checkpoint=str(checkpoint),retention_targets=str(targets_path),parameter_basis=str(basis_path),parameter_scale=.3,training_jobs=jobs,evaluation_resets=evaluation,batches_per_arm=6,episodes_per_arm=48,max_physics_per_arm=48*23334,wall_seconds=1800,max_bytes=80*1024**2,parent_evaluation=str(REPO/'stage/runs/t195_selected_verification/evaluation.jsonl'),
                controlled_difference='Additional whole-episode parameter perturbation in steering direct_head, fixed coefficientseed and scale. Both arms retain same independent tanh-Gaussian action noise, SAC/retention objective and task budget. Does not isolate temporal coherence from increased total exploration amplitude.',
                parameter_contract='PCA whitening top8 hidden features fitted only to declared historical training states plus constantbias; nine standardNormal coefficients scaled.3/sqrt9 folded into direct_head steering weights/bias once per episode. No map/planner/waypoint/runtime teacher. All parent behavior actor parameters fixed within each episode. Central SAC actor receives true off-policy transitions.',
                training_contract='Each batch6originalLEFT5m+2nearLEFT. Fresh twinCritic/targets/optimizers,1500critic-only warmup with observedterminalMAE<.15 before Actor updates;500critic+500actor updates/batch. Same reward/10, finite-task gamma exp(-.002/60), alpha.001,tau.005, terminalbalanced25%. Warmup gate is not full calibration or action-ranking proof. No HER or oldfailedCritic reuse.',
                retention_contract='Seven equally sampled groups32states each; frozen T194 target actions on six T169 old-task states with physical remaining clock and T192 realRIGHTsuccess states. Retention10 in both arms. Targets are training-only regularization, not new observed success or runtime teacher.',
                evaluation_contract='Before Actor updates, require old6+RIGHT5m safeGOAL. After fixedbudget, both central actors frozen on old6+originalLEFT/RIGHT5m+12freshLEFTreset perturbations predeclared. Preserve original scene/radius/caps/legacy history. No best-midpoint selection or automatic promotion.',
                budget='96complete training episodes<=2240064physics, fullwall1800s. Separate frozen diagnostic<=23534steps with120s cap. NewT196outputs<=80MiB, allnewruns1GiB/free40GiB. Retain behavioractor weights perbatch for exact reconstruction and only rollinglatest/previousfullcheckpoints; delete owned previous after final arm evaluation with hash receipt.',
                scope='Single paired pilot seed, warm-start fixed-map development tasks. Extra parameter noise includes extra marginal variation, not a clean temporal-only ablation. No novelty/generalization/algorithm superiority claim. T159 official incumbent; T194 research candidate preserved. No automatic extension.',
                references=['https://arxiv.org/abs/1706.01905','https://proceedings.mlr.press/v80/haarnoja18b.html'])
    files=[checkpoint,Path(plan['parent_evaluation']),Path(previous['retention_demonstrations']),right,basis_path,targets_path]+left_files+[Path(p) for p in seed_episodes]+list(HERE.glob('*.py'))+[Path(previous['anchor_sources']['LEFT']['snapshot_path'])]
    plan['input_hashes']={str(p):sha(p) for p in files};save(HERE/'protocol.json',plan);save(OUT/'protocol.json',plan)
    save(OUT/'preparation.json',{'state':'PASS','calibration_states':len(calibration),'retention_states':len(retention),'retention_groups':7,'whitened_covariance_error':float(np.max(np.abs(whitened.T@whitened/len(whitened)-np.eye(8)))),'parameter_delta_rms_mean_128_draws':float(rms.mean()),'parameter_delta_rms_min':float(rms.min()),'parameter_delta_rms_max':float(rms.max()),'basis_eigenvalues':values[-8:].tolist(),'physics_steps':0,'optimizer_updates':0})
    print(json.dumps({'state':'PREPARED','protocol_sha256':sha(HERE/'protocol.json'),'calibration_states':len(calibration),'mean_rms':float(rms.mean())}))

if __name__=='__main__':main()
