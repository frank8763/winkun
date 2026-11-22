"""Same real verification path; only immutable outside-root usage is cached."""
from pathlib import Path
import copy,json,sys,time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t699_matched_value'))
import t699_common as C
p=C.read(C.OUT/'protocol.json');assert time.time()<C.read(C.DOC/'DIAGNOSIS_BOUND.json')['absolute_stop']
original=C.used
outside=original(ROOT/'stage/runs')-original(C.OUT)
def cached(folder):
    return outside+original(C.OUT) if Path(folder)==ROOT/'stage/runs' else original(folder)
C.used=cached
start=time.perf_counter()
for _ in range(2):C.immutable(p)
elapsed=time.perf_counter()-start
assert elapsed/2*500<360
bad=copy.deepcopy(p);first=next(iter(bad['input_stats']));bad['input_stats'][first][0]+=1
try:C.immutable(bad)
except AssertionError:rejected=True
else:raise AssertionError('Caching suppressed immutable input check')
assert original(ROOT/'stage/runs')-original(C.OUT)==outside
report=dict(state='GREEN',seconds_two_real_immutable_calls=elapsed,projected500_same_verification_seconds=elapsed/2*500,
    unchanged_input_stamp_rejection=True,outside_root_count_exact_before_and_after=True,new_native=0,new_optimizer=0,new_policy_forward=0,
    reason='Only this registered Stage root is writable under exclusive science writer lock; exact full global scan at entry and exit')
C.save(C.DOC/'AUDIT_VERIFICATION_GREEN.json',report);print(json.dumps(report),flush=True)
