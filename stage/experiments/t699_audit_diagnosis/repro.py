"""Replay the actual audit verification seam, no policy/native/optimizer call."""
from pathlib import Path
import json,sys,time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t699_matched_value'))
import t699_common as C
p=C.read(C.OUT/'protocol.json')
bound=C.read(C.DOC/'DIAGNOSIS_BOUND.json');assert time.time()<bound['absolute_stop']
calls=[];original=C.used
def measured(folder):
    start=time.perf_counter();value=original(folder);calls.append(dict(folder=str(folder),seconds=time.perf_counter()-start));return value
C.used=measured
start=time.perf_counter()
for _ in range(2):C.immutable(p)
elapsed=time.perf_counter()-start
report=dict(state='RED' if elapsed/2*500>360 else 'GREEN_SEAM_NOT_SUFFICIENT_TO_EXPLAIN_TIMEOUT',
    seconds_two_real_immutable_calls=elapsed,projected500_same_verification_seconds=elapsed/2*500,
    filesystem_scan_calls=calls,new_native=0,new_optimizer=0,new_policy_forward=0)
C.save(C.DOC/'AUDIT_VERIFICATION_RED.json',report);print(json.dumps(report),flush=True)
assert elapsed/2*500<=360,'RED: repeated real per-row verification alone exceeds original audit360s'
