"""Fixed-workload native process scaling; same Stage stateful Actor contract."""
from pathlib import Path
import sys,fcntl,resource
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t399_rejected_warm_diagnostic'))
from t399_contract import *
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t400r_parallel_native'
PREVIOUS=ROOT/'stage/runs/t399_rejected_warm_diagnostic'

def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    N.capacity(OUT,p['final_bytes'])
    assert N.K.S.used(ROOT/'stage/runs')-N.K.S.used(OUT)+p['final_bytes']<=1024**3
    return N.science(p)

def sizes(a):
    rng=np.random.default_rng(400);values={}
    for k,v in N.S.packed(a).items():
        shape=(51,*v.shape[1:]) if k.startswith('history_') else (50,*v.shape[1:]) if v.ndim and len(v)==len(a['states']) else v.shape
        values[k]=rng.normal(size=shape).astype(v.dtype) if np.issubdtype(v.dtype,np.floating) else rng.integers(0,100,size=shape,dtype=v.dtype)
    return len(N.R.blob(values))
