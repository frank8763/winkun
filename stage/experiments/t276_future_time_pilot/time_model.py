"""Recorded positive-future timing regression; no reachability/Q guarantee."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t275_future_target_contract'))
from future_data import *
import torch


class TimeModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(111,64),torch.nn.Tanh(),
            torch.nn.Linear(64,64),torch.nn.Tanh(),torch.nn.Linear(64,1))
    def forward(self,x):return self.net(x)[:,0]


def load_data(p):
    original=json.loads(Path(p['source_protocol']).read_text())
    with np.load(p['pairs_path']) as z:a={k:z[k].copy() for k in z.files}
    x,y,episode=features(original,a);train=a['train'].astype(bool)
    ids=[np.flatnonzero(episode==i) for i,r in enumerate(original['records']) if r['split']=='train']
    w=np.zeros(len(y),dtype=np.float64)
    for ix in ids:w[ix]=1/len(ids)/len(ix)
    mean=(x.astype(np.float64)*w[:,None]).sum(0).astype(np.float32)
    scale=np.maximum(np.sqrt(((x-mean).astype(np.float64)**2*w[:,None]).sum(0)),.001).astype(np.float32)
    ymean=np.float32((y*w).sum())
    normalized=np.clip((x-mean)/scale,-10,10)
    return dict(raw=x,x=torch.tensor(normalized),y=torch.tensor(y),ymean=ymean,mean=mean,scale=scale,
        train=train,episode=episode,train_ids=ids,records=original['records'],pairs=a,w=w)


def design(x):
    g=x[:,-2:].astype(np.float64)*10;distance=np.linalg.norm(g,axis=1)
    loc=lambda i:int(np.flatnonzero(KEEP==i)[0])
    return np.column_stack([np.ones(len(x)),np.log1p(distance),distance,g,
        x[:,loc(53):loc(56)+1],x[:,107:109],x[:,loc(112)]])


def fit_ridge(d,p):
    X=design(d['raw']);assert X.shape[1]==12
    penalty=np.eye(12)*p['ridge_lambda'];penalty[0,0]=0
    w=d['w'];y=d['y'].numpy()
    return np.linalg.solve(X.T@(X*w[:,None])+penalty,X.T@(w*y))


def weights(d,split):
    w=np.zeros(len(d['episode']),dtype=np.float64)
    ids=[i for i,r in enumerate(d['records']) if r['split']==split]
    for i in ids:
        ix=np.flatnonzero(d['episode']==i);w[ix]=1/len(ids)/len(ix)
    return w


def metric(pred,d,split,selected=None):
    y=d['y'].numpy();w=weights(d,split)
    if selected is not None:
        w*=selected;assert w.sum()>0;w/=w.sum()
    err=pred-y
    sec=np.expm1(np.clip(pred,0,np.log1p(56)));actual=np.expm1(y)
    return dict(log_MAE=float((w*np.abs(err)).sum()),log_RMSE=float(np.sqrt((w*err**2).sum())),
        seconds_MAE_clipped_0_56=float((w*np.abs(sec-actual)).sum()),
        probability_or_shortest_time=False,valid_pairs=int((w>0).sum()))
