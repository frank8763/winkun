"""Original-command competing-event forecast. No Actor or simulator instance."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t262_event_contract'))
from contract_recovery import native_targets


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def used(path):return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def save(path,value):
    p=Path(path);b=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    q=p.with_suffix(p.suffix+'.partial');q.write_bytes(b);q.replace(p)


def dataset(plan):
    records=plan['records'];parts=[];details=[];offsets=[0]
    for k,r in enumerate(records):
        assert sha(r['path'])==r['sha256'],r['id']
        with np.load(r['path']) as z:
            x=np.concatenate([z['states'],z['actions']],axis=1).astype(np.float32)
            counts=z['native_counts'].astype(np.int64);flags=z['end_flags']
        assert x.shape[1]==122 and np.isfinite(x).all()
        assert np.array_equal(flags[-1],np.array([r['collision'],r['success'],r['timeout']],dtype=np.uint8))
        assert counts.sum()==r['physics_steps'] and r['complete']
        y,m=native_targets(counts,flags,r['cap'],r['complete'],plan['horizons_physics'])
        # The actual requested goal/history stays unchanged. These are labels,
        # not newly commanded future goals or an on-policy value for a new Actor.
        parts.append((x,y,m));offsets.append(offsets[-1]+len(x))
        details.append(dict(id=r['id'],rows=len(x),split=r['split'],side=r['side'],distance_m=r['distance_m']))
    xx=np.concatenate([a[0] for a in parts]);yy=np.concatenate([a[1] for a in parts]);mm=np.concatenate([a[2] for a in parts])
    episode=np.concatenate([np.full(len(a[0]),k,dtype=np.int16) for k,a in enumerate(parts)])
    train=np.array([r['split']=='train' for r in records])[episode]
    assert len(xx)<=30000 and len(xx)==22456
    train_ids=[np.flatnonzero((episode==k)&mm.any(1)) for k,r in enumerate(records) if r['split']=='train']
    assert all(len(i)>0 for i in train_ids)
    # Each episode has equal weight, then each decision within it has equal weight.
    norm_weights=np.zeros(len(xx),dtype=np.float64)
    for ids in train_ids:norm_weights[ids]=1/len(train_ids)/len(ids)
    mean=np.sum(xx.astype(np.float64)*norm_weights[:,None],axis=0).astype(np.float32)
    var=np.sum((xx-mean).astype(np.float64)**2*norm_weights[:,None],axis=0)
    scale=np.maximum(np.sqrt(var),1e-3).astype(np.float32)
    zz=np.clip((xx-mean)/scale,-10,10)
    return dict(x=torch.tensor(zz),raw_x=xx,y=torch.tensor(yy,dtype=torch.float32),
        mask=torch.tensor(mm,dtype=torch.bool),episode=episode,train=train,train_ids=train_ids,
        mean=mean,scale=scale,details=details,offsets=np.array(offsets,dtype=np.int64))


class EventModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(122,64),torch.nn.Tanh(),
            torch.nn.Linear(64,64),torch.nn.Tanh(),torch.nn.Linear(64,18))

    def forward(self,x):
        h=torch.softmax(self.net(x).reshape(-1,6,3),dim=-1)
        survival=torch.cumprod(h[:,:,2],dim=1)
        previous=torch.cat([torch.ones_like(survival[:,:1]),survival[:,:-1]],dim=1)
        cumulative=torch.cumsum(previous[:,:,None]*h[:,:,:2],dim=1)
        return torch.cat([cumulative,survival[:,:,None]],dim=-1)


def episode_weights(data,split):
    e=data['episode'];m=data['mask'].numpy();w=np.zeros(m.shape,dtype=np.float64)
    ids=[k for k,r in enumerate(data['details']) if r['split']==split]
    for k in ids:
        hit=np.flatnonzero(e==k)
        for h in range(6):
            ix=hit[m[hit,h]]
            if len(ix):w[ix,h]=1/len(ix)
    w/=w.sum(axis=0).clip(min=1)[None,:]
    return w


def frequency_baseline(data):
    w=episode_weights(data,'train');y=data['y'].numpy()
    return (w[:,:,None]*y).sum(axis=0)


def metrics(pred,data,split):
    y=data['y'].numpy();w=episode_weights(data,split)
    brier=((pred-y)**2*w[:,:,None]).sum(axis=0)
    nll=-(y*np.log(np.maximum(pred,1e-8))*w[:,:,None]).sum(axis=(0,2))
    # Fixed five-bin reliability description; correlated decisions are not
    # independent calibration samples and these are not confidence bounds.
    ece=[]
    for c in range(2):
        value=0.;total=w.sum()
        for low in np.arange(0,1,.2):
            select=(pred[:,:,c]>=low)&(pred[:,:,c]<(low+.2) if low<.8 else pred[:,:,c]<=1)
            weight=w*select;mass=weight.sum()
            if mass:value+=mass*abs(float((weight*(pred[:,:,c]-y[:,:,c])).sum()/mass))
        ece.append(float(value/total))
    support=[]
    for h in range(6):
        ids={int(data['episode'][k]) for k in np.flatnonzero(w[:,h]>0)}
        support.append(dict(valid_decisions=int((w[:,h]>0).sum()),valid_episodes=len(ids),
            goal_decisions=int(((y[:,h,0]>0)&(w[:,h]>0)).sum()),contact_decisions=int(((y[:,h,1]>0)&(w[:,h]>0)).sum())))
    return dict(Brier_per_horizon_class=brier.tolist(),mean_Brier_total=float(brier.sum(-1).mean()),
        mean_Brier_goal=float(brier[:,0].mean()),mean_Brier_contact=float(brier[:,1].mean()),
        NLL_per_horizon=nll.tolist(),mean_NLL=float(nll.mean()),five_bin_goal_contact_ECE=ece,support=support)
