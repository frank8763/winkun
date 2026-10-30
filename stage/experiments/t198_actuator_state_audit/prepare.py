"""Declare the two frozen episodes and two short reset-only probes."""
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
from common import *

def main():
    assert not OUT.exists()
    previous=json.loads((REPO/'stage/experiments/t196_left_exploration/protocol.json').read_text());diagnostic=json.loads((REPO/'stage/runs/t196_left_exploration/diagnostic.json').read_text())
    plan={'experiment':'T198','checkpoint':previous['checkpoint'],'cases':previous['primary_cases'][-2:],'snapshot':diagnostic['snapshot_path'],'snapshot_elapsed':diagnostic['snapshot_elapsed_steps'],'max_physics':2*23334+200,'wall_seconds':180,'max_bytes':8*1024**2,'optimizer_updates':0,
          'scope':'Frozen evidence audit only. Actual frontsteering joints/targets observed for original LEFT/RIGHT rollouts, plus two reset-only ±.2rad steering interventions. No statewrites after stepping, goal/geometry/caps unchanged. Joint states are not supplied to deployed Actor yet. This is not a navigation success experiment or causal proof of existing failure. T159/T194 preserved; Stageonly; allnewruns1GiB/free40GiB.'}
    files=[Path(plan['checkpoint']),Path(plan['snapshot'])]+list(HERE.glob('*.py'));plan['input_hashes']={str(p):sha(p) for p in files}
    OUT.mkdir();save(HERE/'protocol.json',plan);save(OUT/'protocol.json',plan);print(json.dumps({'state':'PREPARED','protocol_sha256':sha(HERE/'protocol.json')}))

if __name__=='__main__':main()
