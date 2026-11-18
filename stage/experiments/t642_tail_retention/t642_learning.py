"""Fixed worst-task/time-tail retention; standard risk aggregation, not novelty."""
from pathlib import Path
import math,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t637_joint_coverage_recovery'))
import t637_learning as Base
L,P,E,torch,np=Base.L,Base.P,Base.E,Base.torch,Base.np
make,state,load,location,correction,export=Base.make,Base.state,Base.load,Base.location,Base.correction,Base.export
read,save,sha,digest=Base.read,Base.save,Base.sha,Base.digest


def pack(data,device):
    v=Base.pack(data,device);v['lengths']=[len(g['f']) for g in data];return v


def retention(error,lengths):
    risks=[];offset=0
    for n in lengths:
        episode=error[offset:offset+n];offset+=n
        risks.append(.5*episode.mean()+.5*episode.topk(max(1,math.ceil(.1*n))).values.mean())
    assert offset==len(error)
    risks=torch.stack(risks)
    return risks.topk(max(1,math.ceil(.1*len(risks)))).values.mean()


def objective(q,keep,new,p):
    error=((location(q,keep)-keep['target'])/.2).square().mean(-1)
    mean=(error*keep['weights']).sum();tail=retention(error,keep['lengths'])
    protection=.5*mean+.5*tail
    loss=P.action_loss(location(q,new),new['target']).mean(-1)/.2**2
    repair=.5*loss.mean()+.5*loss.topk(max(1,(len(loss)+4)//5)).values.mean()
    return repair+p['keep_weight']*protection,repair,protection


def fit(obj,keep,new,p,guard,event):
    # Identical optimizer, step count, recovery loss, rollback and final metrics;
    # only the old-history risk aggregation changes.
    previous=Base.objective;Base.objective=objective
    try:return Base.fit(obj,keep,new,p,guard,event)
    finally:Base.objective=previous
