"""Register a bounded frozen cause audit; T201 training remains unstarted."""
from pathlib import Path
import hashlib,json,shutil

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t202_cause_audit'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert not OUT.exists(), 'No duplicate audit'
    old=json.loads((ROOT/'stage/experiments/t200_residual_placement/protocol.json').read_text())
    summary=json.loads((ROOT/'stage/runs/t200_residual_placement/summary.json').read_text())
    checkpoints={k:v['checkpoint'] for k,v in summary['training'].items()}
    inputs=[Path(p) for p in checkpoints.values()]+[Path(old['retention_targets']),ROOT/'stage/experiments/t200_residual_placement/protocol.json',ROOT/'stage/runs/t196_left_exploration/left_baseline_trace.npz']
    for arm in checkpoints:
        inputs.append(ROOT/f'stage/runs/t200_residual_placement/{arm}/replay_receipts.json')
        rows=json.loads(inputs[-1].read_text())
        inputs.extend(Path(r['path']) for r in rows)
    inputs+=list((ROOT/'stage/experiments/t200_residual_placement').glob('*.py'))+list(HERE.glob('*.py'))
    plan={'experiment':'T202','case':old['primary_cases'][6],'checkpoints':checkpoints,'retention_targets':old['retention_targets'],
          'alpha':old['alpha'],'seed':2026103402,'anchor_elapsed_steps':[0,14000],'rollout_noise_seeds':[2026103402,2026103403,2026103404,2026103405],
          'first_actions':['baseline','q_positive','q_negative'],'action_gradient_step':.25,'hold_physics':100,
          'max_physics':900000,'wall_seconds':900,'max_bytes':12*1024**2,'peak_bytes':16*1024**2,'pending_branch_bytes':1024**2,
          'optimizer_updates':0,'input_hashes':{str(p):sha(p) for p in sorted(set(inputs))},
          'offline_contract':'Reload only completed T200 models/replay. Inspect terminal/nonterminal errors, actual action/drive support, gradients of Q/entropy/seven-group BC separately, state distribution, clock and event/reward contracts. No refit or parameter mutations; offline gradient norms are not Adam update attribution.',
          'native_contract':'Two frozen final T200 policies, two own deterministic-prefix integration snapshots at absolute steps0/14000. Three first actions (mean and plus/minus normalized minQ gradient by.25), each followed by SAME frozen stochastic policy with four paired noise seeds. Maximum48 branches, no runtime planner/teacher/waypoint. Snapshot restoration only before each diagnostic branch. Original scene/goal/radius/time cap and absolute remaining clock retained. First fixed action lasts100physics or natural termination; noise/log-density starts next decision.',
          'value_contract':'Compute full native environment MC and soft-Q MC excluding current action entropy, matching SAC Q definition. Stable latent log density, physical gamma exp(-.002/60), finite absorbing events. Compare minQ to four-sample MC mean and paired action differences. Small consumed fixed-task audit, no statistically conclusive universal ranking claim. Stochastic continuation is same final policy, unlike historical behavior-vs-current-policy comparisons.',
          'storage_contract':'Final12MiB/peak16MiB, reserve1MiB per pending branch; all newruns<1GiB/free40GiB. No new model/optimizer files. Existing evidence unchanged. No training or automatic extension; T201 remains a draft.',
          'references':['https://spinningup.openai.com/en/latest/algorithms/sac.html','https://proceedings.mlr.press/v97/fu19a.html']}
    used=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file())
    assert used+plan['peak_bytes']<1024**3 and shutil.disk_usage(ROOT).free>40*1024**3
    OUT.mkdir()
    for p in [HERE/'protocol.json',OUT/'protocol.json']:p.write_text(json.dumps(plan,indent=2)+'\n')
    (OUT/'preparation.json').write_text(json.dumps({'state':'PASS','protocol_sha256':sha(HERE/'protocol.json'),'input_files':len(plan['input_hashes']),'runs_bytes_before':used,'optimizer_updates':0},indent=2)+'\n')
    print('T202 PREPARED',sha(HERE/'protocol.json'),used)

if __name__=='__main__':main()
