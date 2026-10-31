"""Single-Actor weight calibration, never runtime ensemble or planner."""
from pathlib import Path
import sys,os,json,time,copy,traceback,io,gzip
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t234_weight_calibration'
sys.path.insert(0,str(ROOT/'stage/experiments/t229_valid_start_acceptance'))
import valid_support as V
S,C,R,np,torch=V.S,V.C,V.R,V.np,V.torch
digest,used,state_np=S.digest,S.used,S.state_np

def read(path):return torch.load(io.BytesIO(gzip.decompress(Path(path).read_bytes())),map_location='cpu',weights_only=False)['policy']
def interpolate(a,b,alpha):
    assert set(a)==set(b) and 0<=alpha<=1
    if alpha==0:return {k:v.clone() for k,v in a.items()}
    if alpha==1:return {k:v.clone() for k,v in b.items()}
    return {k:(1-alpha)*a[k]+alpha*b[k] for k in a}
def worker(pipe,index,deadline):
    V.OUT=OUT;V.worker(pipe,index,deadline)
