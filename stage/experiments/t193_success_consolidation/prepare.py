"""Select earliest real RL success per side after the complete SAC pilot."""
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
from common import *

def main():
    assert not OUT.exists()
    source=REPO/'stage/runs/t192_sac_her';summary=json.loads((source/'summary.json').read_text());assert summary['state']=='COMPLETE'
    previous=json.loads((REPO/'stage/experiments/t192_sac_her/protocol.json').read_text())
    rows=[json.loads(s) for s in (source/'episodes.jsonl').read_text().splitlines()];chosen=[]
    for side in ['LEFT','RIGHT']:
        candidates=[r for r in rows if not r.get('snapshot_side') and r['case'].endswith(side) and r['success'] and not r['collision']]
        if candidates:chosen.append(candidates[0])
    assert chosen,'No real original5m successful data; no BC launch'
    for r in chosen:assert sha(r['path'])==r['sha256']
    parent=REPO/'stage/runs/t190_adaptive_start_curriculum/checkpoint_latest.pt'
    plan={'experiment':'T193','seed':2026103093,'checkpoint':str(parent),'primary_cases':previous['primary_cases'],'retention_demonstrations':previous['retention_demonstrations'],'successful_sources':chosen,'supervised_updates':5000,'wall_seconds':600,'max_physics':(len(chosen)+8)*23334,'max_bytes':32*1024**2,
          'selection':'Earliest recorded safe native original5m stochastic SAC training success per side in completed T192 episodes.jsonl. No fastest/best-checkpoint selection; maximum2trajectories. All T192 failures remain preserved.',
          'validation':'First replay saved native actions in a newly created original task and verify113states, every physicsxy and exactGOAL step. This replay only audits the training data, not network deployment success or independent evidence. Stop on mismatch.',
          'training':'Start stable old6-retaining T190Actor, freshAdam bodylr3e-5, native/direct heads3e-4. Fixed5000supervisedMSE updates,32uniformstates/task/step across six T169 successes and each newly verified5m trajectory. All tasks equal weight, no temporal filtering or invented action labels. Past-action input columns frozen0. No new RL samples or RL updates.',
          'evaluation':'One fixed final network on original8tasks; all actions closed-loop from current113observation. No action replay, trajectory lookup, planner, waypoint or teacher at deployment. Original scene0.20m radius/caps unchanged. Report all8 separately, no automatic promotion.',
          'scope':'Supervised policy consolidation from real end-to-end RL successful behavior and existing old6 demonstration states. Not pureRL, not freshRL optimizer updates, not independent generalization/novelty. T159 incumbent; all code/data inwinkun, noPhase, allnewruns1GiB/free40GiB.'}
    files=[parent,source/'summary.json',source/'episodes.jsonl',Path(plan['retention_demonstrations'])]+list(HERE.glob('*.py'))+[Path(r['path']) for r in chosen]
    plan['input_hashes']={str(p):sha(p) for p in files};OUT.mkdir();save(HERE/'protocol.json',plan);save(OUT/'protocol.json',plan)
    print(json.dumps({'state':'PREPARED','sides':[r['case'] for r in chosen],'protocol_sha256':sha(HERE/'protocol.json')}))

if __name__=='__main__':main()
