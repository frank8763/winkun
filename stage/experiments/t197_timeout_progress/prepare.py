"""Lock a single-factor bounded timeout-distance objective comparison."""
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
from common import *
from replay import build,GAMMA

def main():
    assert not OUT.exists()
    old=json.loads((REPO/'stage/experiments/t196_left_exploration/protocol.json').read_text());summary_path=REPO/'stage/runs/t196_left_exploration/summary.json';summary=json.loads(summary_path.read_text());assert summary['state']=='COMPLETE'
    checkpoint=Path(summary['training']['action_noise']['checkpoint']);assert sha(checkpoint)==summary['training']['action_noise']['sha256']
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    teacher=Policy().eval().requires_grad_(False);teacher.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['policy'])
    with np.load(old['retention_targets']) as z:states=z['states'].copy();offsets=z['offsets'].copy()
    with torch.no_grad():targets=torch.cat([teacher(x) for x in torch.tensor(states).split(128)]).numpy()
    rows=[json.loads(s) for s in (REPO/'stage/runs/t196_left_exploration/episodes.jsonl').read_text().splitlines()]
    seed_episodes=list(old['seed_episodes'])+[r['path'] for r in rows]
    seed_hashes={r['path']:r['sha256'] for r in rows}
    errors=[];timeout_n=0
    for path in seed_episodes:
        if path in seed_hashes:assert sha(path)==seed_hashes[path]
        with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        if 'end_flags' in a:a['end_flags']=a['end_flags'].astype(float)
        before=build(a,0.);after=build(a,2.)
        def mc(tr):
            value=0.
            for r,d in zip(tr['rewards'][::-1,0],tr['discounts'][::-1,0]):value=float(r)+float(d)*value
            return value
        n=int(a['native_counts'].sum());is_timeout=bool(a['end_flags'][-1,2]) if 'end_flags' in a else bool(a['physical'][-1,9])
        d=float(np.linalg.norm(a['next_poses'][-1,:2]-a['goal_xy']))
        expected=-.2*min(d,2.)*GAMMA**(n-1) if is_timeout else 0.
        error=abs((mc(after)-mc(before))-expected);errors.append(error);assert error<2e-5
        timeout_n+=is_timeout
    OUT.mkdir();target_path=OUT/'retention_targets.npz';np.savez_compressed(target_path,states=states,actions=targets,offsets=offsets)
    rng=np.random.default_rng(2026103297);evaluation=[];left=old['primary_cases'][6]
    for i in range(12):
        dx,dy=rng.uniform(-.03,.03,2);yaw=float(rng.uniform(-2,2));evaluation.append({'id':f'left_T197_holdout_{i}','case':left,'group':'left_fresh_reset','perturbation':[float(dx),float(dy),yaw]})
    plan={k:old[k] for k in ['primary_cases','anchor_sources','initial_std','alpha','warmup_updates','updates_per_batch','parameter_basis','parameter_scale','training_jobs','batches_per_arm','episodes_per_arm','max_physics_per_arm']}
    plan.update(experiment='T197',seed=2026103097,checkpoint=str(checkpoint),retention_targets=str(target_path),seed_episodes=seed_episodes,evaluation_resets=evaluation,wall_seconds=1800,max_bytes=64*1024**2,
                arms={'no_timeout_distance':{'timeout_penalty':0.},'timeout_distance':{'timeout_penalty':2.}},
                controlled_difference='Only extra finite-timeout terminal penalty -k*min(remaining_distance_m,2), k0 versus2 before /10 normalization. Same Actor parent, fresh Critics/optimizers, seeds, parameter/action exploration, source100episodes, seven-task retention and update/episode budgets.',
                training_contract='Both arms include same whole-episode steering parameter noise(.3scale, fixed T196basis) and tanh-Gaussian action noise. 48episodes/arm,6originalLEFT+2nearLEFT perbatch. FreshCritic1500warmup;3000Actor+4500Critic total perarm. Reward0arm exactly same T196 objective; penaltyarm changes training objective, not policy-invariant shaping.',
                retention_contract='Frozen common T196 action-noise final Actor on same seven groups as T196: old6T169states+trueRIGHTsuccessstates. Uniform32/group, MSEweight10; training-only teacher, singleActor deployed.',
                penalty_contract='At true finite timeout only, event-5 becomes -5-2*min(distance,2), in[-9,-5]. GOAL+10 and collision-10 unchanged. Original goal geometry/radius.20m/deadlines unchanged. Penalty is not navigation success or invariant reward shaping. No artificial waypoint or goal-distance shortcut.',
                evaluation_contract='Initialold6+RIGHT must pass. FinalcentralActors only on original8 plus12newLEFTperturbations fixedbeforetraining; no mid-checkpoint selection, no automatic promotion. All extra exploratory successes separately preserved.',
                budget='96completeepisodes<=2240064newtrainingphysics,48frozenfinal/initialcases,wall1800s;T197newfiles<=64MiB,allnewruns1GiB/free40GiB. Previousrawinputsreusedread-only, not recounted as new interactions. Ownedrollingpreviousdeletedafterfinalarmvalidation with hash receipt.',
                scope='Single paired pilot seed in known fixedgeometry. Relative to T196 parent/replay/teacher/noise differ; only withinT197 reward term controlled. No algorithm superiority, originality or generalization claim; formalT159 and verifiedT194 preserved. No automatic extension.',
                references=['https://proceedings.mlr.press/v80/haarnoja18b.html'])
    files=[checkpoint,summary_path,Path(old['retention_targets']),Path(old['parameter_basis']),target_path]+[Path(p) for p in seed_episodes]+list(HERE.glob('*.py'))+[Path(old['anchor_sources']['LEFT']['snapshot_path'])]
    plan['input_hashes']={str(p):sha(p) for p in files};save(HERE/'protocol.json',plan);save(OUT/'protocol.json',plan)
    save(OUT/'preparation.json',{'state':'PASS','seed_episodes':len(seed_episodes),'seed_timeouts':timeout_n,'max_mc_penalty_difference_error':max(errors),'physics_steps':0,'optimizer_updates':0,'constant_timeout_old_signal':'For equal initial distance and finite timeout steps, old exact complete return independent of remainingdistance; own derivation, not an implementation failure.'})
    print(json.dumps({'state':'PREPARED','protocol_sha256':sha(HERE/'protocol.json'),'seed_episodes':len(seed_episodes),'max_mc_error':max(errors)}))

if __name__=='__main__':main()
