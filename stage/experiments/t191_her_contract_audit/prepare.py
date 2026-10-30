"""Freeze the audit inputs and its four-episode budget before collection."""
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
from common import *

def main():
    assert not OUT.exists()
    previous=json.loads((REPO/'stage/experiments/t190_adaptive_start_curriculum/protocol.json').read_text())
    summary=json.loads((REPO/'stage/runs/t190_adaptive_start_curriculum/summary.json').read_text());assert summary['state']=='COMPLETE'
    checkpoint=REPO/'stage/runs/t190_adaptive_start_curriculum/checkpoint_latest.pt';assert sha(checkpoint)==summary['checkpoint_sha256']
    cases=previous['primary_cases'][-2:]
    jobs=[{'id':'original_'+c['side'],'case':c} for c in cases]+[{'id':'near_'+c['side'],'case':c,'snapshot_side':c['side']} for c in cases]
    plan={'experiment':'T191','seed':2026103091,'jobs':jobs,'checkpoint':str(checkpoint),'anchor_sources':previous['anchor_sources'],'hold_steps':100,'max_physics':93336,'wall_seconds':300,'max_bytes':16*1024**2,'optimizer_updates':0,'goal_radius_m':.2,'relabel_samples_per_episode':8,
          'contract':'Frozen one-network policy, original goals/scenes/time limits, no goal or pose writes during actual episodes. In-memory history-frame hook only records generation pose/goal. All113 fields rebuilt, original cached3m history retained. Offline future achieved-position labels; no route, planner, waypoint or runtime teacher.',
          'terminal_contract':'Native contact has priority; stop relabeled episode at first true per-physics-step radius hit, preserve original finite clock, no fictitious post-contact data. A terminal before action hold ends needs reward/discount truncation; stored later endpoint is not the exact first-hit observation and cannot enter next-state losses. Audit does not yet emit replay.',
          'scope':'Bounded data audit, not HER training, policy improvement or novelty evidence. All old experiments preserved; Stage/Phase isolated. All new runs cumulatively1GiB and disk free40GiB.',
          'references':['https://arxiv.org/abs/1707.01495']}
    files=[checkpoint]+list(HERE.glob('*.py'))+[Path(v['snapshot_path']) for v in plan['anchor_sources'].values()]
    plan['input_hashes']={str(p):sha(p) for p in files};OUT.mkdir();save(HERE/'protocol.json',plan);save(OUT/'protocol.json',plan)
    print(json.dumps({'state':'PREPARED','protocol_sha256':sha(HERE/'protocol.json')}))

if __name__=='__main__':main()
