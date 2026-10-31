"""Register one finite pilot; never resumes or duplicates completed experiments."""
from pathlib import Path
import json, hashlib, shutil
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'stage/runs/t210_full_parameter_progress'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not OUT.exists(), 'No duplicate T210'
    old = json.loads((ROOT/'stage/experiments/t207_parameter_pg/protocol.json').read_text())
    prior = json.loads((ROOT/'stage/runs/t207_parameter_pg/summary.json').read_text())
    assert prior['state'] == 'COMPLETE' and prior['all_workers_stopped']
    tasks = [ROOT/'stage/runs/t207_parameter_pg/tasks'/('original_'+c['id']+'.npz') for c in old['primary_cases']]
    plan = dict(experiment='T210', seed=2026103210, checkpoint=old['checkpoint'], retention_targets=old['retention_targets'],
                primary_cases=old['primary_cases'], task_snapshots=[str(p) for p in tasks],
                arms=['original', 'terminal_progress'], terminal_distance_coefficients={'original': 0., 'terminal_progress': .3},
                max_rounds=4, training_episodes_per_arm_per_round=4, workers=4,
                actor_lr=3e-5, value_lr=3e-4, epochs=3, minibatch=256, clip_ratio=.2, target_KL=.02,
                grad_clip=.5, retention_weight=1., anchor_per_task=64,
                wall_seconds=900, final_reserve_seconds=180, max_episodes=200, max_physics=4700000,
                final_bytes=36*1024**2, peak_bytes=40*1024**2, pending_bytes=2*1024**2,
                robustness_xy=[-.03,0.,.03], robustness_yaw=[-2.,0.,2.])
    plan['method_contract'] = ('Two identical full-parameter 113->800->600->2 Stage Actors warm-start T194, '
        'native affine/direct heads trainable too; unused logstd fixed, native truncated Gaussian sigma(.35,.30). '
        'No CNN/GRU/added sensor/auxiliary/curiosity/curriculum/oldQ. Only arm difference is terminal remaining-distance '
        'penalty coefficient0 versus.3 at ALL goal/contact/true finite-horizon timeout terminals. Original reward otherwise exact T203. '
        'This changes the task objective; no policy-invariance claim. LEFT original starts only; old7 offline balanced action-MSE anchors '
        'weight1 both arms. Complete fresh episodes, MC-minus-previous-state-V advantages (GAE lambda1), globally standardized; '
        'new independent 113->128->128->1 V fitted to complete MC after each batch, no Q action gradient. '
        'Adam3epochs,256minibatch,clippedPPO.2,globalgradclip.5,stop subsequent epochs/minibatches when sampledapproxKL>.02. '
        'Same initial V and random action seeds in each arm, diverging states allowed. Engineering pilot not algorithm novelty or multiseseed proof.')
    plan['accounting_contract'] = ('4rounds*4LEFT*2arms=32trainingepisodes<=746688trainingphysics; '
        'parent8, per-round deterministic old8*2*4=64, final reloadold8*2=16, atmost2training replays, '
        'one prespecified first passing arm reloadold8+54grid; <=200rollout/replay commands,4.7Mallphysics. '
        'Store every behavior fullActor and complete training observations/actions/logp; eval actions/poses/flags. '
        'All model changes and minibatch counts recorded. No optimizer snapshots: not an exactly resumable training export. '
        'First paired round with center LEFT GOAL stops training; latest centers only, no historical best selection. '
        'T159/T194 unchanged. Old8+savedreload+54grid allsafe required for stable5m. Failed/partial not silently promoted. '
        '900s, last180s evaluation reserve; allnewruns1GiB/free40GiB/ownfinal36MiB/peak40MiB; no duplicate, continuation, or next trial auto-launch.')
    plan['references'] = ['https://arxiv.org/abs/2005.12729', 'https://arxiv.org/abs/1707.06347',
        'https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py', 'https://arxiv.org/html/2210.04839']
    files = [Path(plan[k]) for k in ('checkpoint','retention_targets')] + tasks + list(HERE.glob('*.py')) + [
        ROOT/'stage/experiments/t203_native_exploration/runtime.py',ROOT/'stage/experiments/t200_residual_placement/common.py',
        ROOT/'stage/docs/research_20261031_navigation/review.md']
    plan['input_hashes'] = {str(p):sha(p) for p in files}
    used = sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file())
    assert used+plan['peak_bytes'] < 1024**3 and shutil.disk_usage(ROOT).free > 40*1024**3
    OUT.mkdir()
    for p in (HERE/'protocol.json', OUT/'protocol.json'):
        p.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'state':'REGISTERED','protocol_sha256':sha(HERE/'protocol.json'),'current_runs_MiB':used/1024**2}))
if __name__ == '__main__': main()
