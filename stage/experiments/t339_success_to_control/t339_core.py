"""One bounded on-policy success-to-control test; not an extension of old jobs."""
from pathlib import Path
import sys
_ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(_ROOT/'stage/experiments/t338_remaining_recovery'))
from t338r_core import *
ROOT=_ROOT;HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t339_success_to_control';DOC=ROOT/'stage/docs/review_20261104'

def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    assert K.S.used(OUT)<=p['final_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']

def cached(p):return [dict(row=r,arrays=arrays(r['path'])) for r in p['cached_training_rows']]
