"""Own-exploration consolidation with perturbed old-skill retention coverage."""
from pathlib import Path
import sys,os,json,time,copy,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t228_balanced_stability'
sys.path.insert(0,str(ROOT/'stage/experiments/t221_left_success'))
import left_support as L
S,C,R,np,torch=L.S,L.C,L.R,L.np,L.torch
digest,used,state_np=S.digest,S.used,S.state_np
sys.path.insert(0,str(ROOT/'stage/experiments/t222_stability'))
import acceptance_support as A

def groups(plan):
    parent,result=L.groups(plan);p=S.policy(parent)
    for source in plan['additional_sources']:
        assert C.sha(source['path'])==source['sha256']
        with np.load(source['path']) as z:x=torch.tensor(z['states']);actual=torch.tensor(z['actions'])
        index=6 if source['side']=='RIGHT' else 7
        with torch.no_grad():y=C.ParentPolicy.forward(p,x).detach() if index==6 else actual
        result[index]=(torch.cat([result[index][0],x]),torch.cat([result[index][1],y]))
    return result

def worker(pipe,index,deadline):
    A.OUT=OUT;A.worker(pipe,index,deadline)
