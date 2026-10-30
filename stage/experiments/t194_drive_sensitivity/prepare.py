"""Predeclare a small neural throttle-bias sensitivity grid after failed BC."""
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
from common import *

def main():
    assert not OUT.exists()
    previous=json.loads((REPO/'stage/experiments/t193_success_consolidation/protocol.json').read_text())
    summary_path=REPO/'stage/runs/t193_success_consolidation/summary.json';summary=json.loads(summary_path.read_text());assert summary['state']=='COMPLETE' and summary['scores']['old6']['safe_success']==6
    ck=REPO/'stage/runs/t193_success_consolidation/final_policy.pt';assert sha(ck)==summary['checkpoint_sha256']
    deltas=[.05,.1,.2,.3]
    plan={'experiment':'T194','checkpoint':str(ck),'baseline_summary':str(summary_path),'primary_cases':previous['primary_cases'],'throttle_bias_deltas':deltas,'jobs':[{'id':f'd{delta}_{c["id"]}','delta':delta,'case':c} for delta in deltas for c in previous['primary_cases']],'wall_seconds':300,'max_physics':32*23334,'max_bytes':16*1024**2,
          'controlled_difference':'Only native_head.bias[0] receives one predeclared additive delta; all other weights byte-equal to T193. Single closed-loop neural policy, no controller/planner/waypoint/teacher or action replay. Original scenes/goals/caps/radius retained.',
          'selection':'Baseline0 reuses completed T193 evidence, not extra independent trials. Among new candidates with all old6 safe success, select most original5m successes, then smallest delta; retain no candidate if no new5m success. Fixed development tasks, not generalization or RL gradient learning.',
          'scope':'Frozen neural parameter sensitivity and engineering model selection after T193 RIGHT5m timeout at.974m, not new deepRL training/algorithm novelty. No automatic promotion; official T159 retained. Stageonly, newruns1GiB/free40GiB.'}
    files=[ck,summary_path]+list(HERE.glob('*.py'));plan['input_hashes']={str(p):sha(p) for p in files}
    OUT.mkdir();save(HERE/'protocol.json',plan);save(OUT/'protocol.json',plan);print(json.dumps({'state':'PREPARED','protocol_sha256':sha(HERE/'protocol.json')}))

if __name__=='__main__':main()
