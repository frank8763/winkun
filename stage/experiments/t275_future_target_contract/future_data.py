"""Recorded safe decision-endpoint time relations, never relabeled Actor120."""
from pathlib import Path
import hashlib,json,math
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
GOAL_COLUMNS=np.array([48,49,50,51,52,72,73,84,85,96,97,108,109],dtype=np.int64)
KEEP=np.setdiff1d(np.arange(120),GOAL_COLUMNS)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def used(path):return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
def save(path,value):
    p=Path(path);q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');q.replace(p)


def encode(states):
    x=np.asarray(states);assert x.shape[-1]==120
    return x[...,KEEP].copy()


def relative(pose,point):
    pose=np.asarray(pose,dtype=np.float64);d=np.asarray(point,dtype=np.float64)-pose[...,:2]
    c=np.cos(pose[...,2]);s=np.sin(pose[...,2])
    return np.stack([c*d[...,0]+s*d[...,1],-s*d[...,0]+c*d[...,1]],axis=-1)


def build(plan):
    epi=[];anchor=[];endpoint=[];dt=[];goals=[];split=[];details=[];invalid=0;missing=0;goal_error=0.
    source_signatures={'train':set(),'test':set()}
    for e,r in enumerate(plan['records']):
        assert sha(r['path'])==r['sha256']
        source_signatures[r['split']].add(tuple(r['signature']))
        with np.load(r['path']) as z:
            x=z['states'];a=z['actions'];counts=z['native_counts'].astype(np.int64);pose=z['poses'];npose=z['next_poses'];flags=z['end_flags']
            assert x.shape[1]==120 and np.isfinite(x).all() and r['complete']
            assert int(counts.sum())==r['physics_steps'] and np.all(flags[:-1]==0)
            assert np.array_equal(flags[-1],np.array([r['collision'],r['success'],r['timeout']],dtype=np.uint8))
            ends=counts.cumsum();starts=np.r_[0,ends[:-1]]
            assert np.array_equal(x[1:,117:119],a[:-1]) and np.array_equal(x[1:,119],counts[:-1].astype(np.float32)/100)
            # Verify the current five explicit goal columns against recorded
            # native poses without reconstructing missing historical poses.
            body=x[:,51:53].astype(np.float64)*10;c=np.cos(pose[:,2]);s=np.sin(pose[:,2])
            world=pose[:,:2]+np.stack([c*body[:,0]-s*body[:,1],s*body[:,0]+c*body[:,1]],axis=1)
            assert np.max(np.ptp(world,axis=0))<1e-5
            d=np.linalg.norm(body,axis=1);angle=np.arctan2(body[:,1],body[:,0])
            err=max(float(np.max(np.abs(x[:,48]-d/10))),float(np.max(np.abs(x[:,49]-np.sin(angle)))),float(np.max(np.abs(x[:,50]-np.cos(angle)))))
            assert err<2e-6;goal_error=max(goal_error,err)
            first=len(dt)
            for k in range(0,len(x),plan['anchor_stride_decisions']):
                for horizon in plan['requested_physics_horizons']:
                    j=int(np.searchsorted(ends,int(starts[k])+horizon,side='left'))
                    if j>=len(x):missing+=1;continue
                    assert j>=k and int(ends[j]-starts[k])>=horizon
                    if bool(flags[j,0]):invalid+=1;continue
                    delta=int(ends[j]-starts[k]);assert 0<delta<=r['cap']-starts[k] and delta<65536
                    g=relative(pose[k],npose[j,:2])
                    assert np.allclose(np.linalg.norm(g),np.linalg.norm(npose[j,:2]-pose[k,:2]),atol=1e-12,rtol=0)
                    epi.append(e);anchor.append(k);endpoint.append(j);dt.append(delta);goals.append(g.astype(np.float32));split.append(r['split']=='train')
            details.append(dict(id=r['id'],split=r['split'],distance_m=r['distance_m'],side=r['side'],pairs=len(dt)-first,
                original_success=r['success'],original_collision=r['collision'],original_timeout=r['timeout']))
    assert not source_signatures['train']&source_signatures['test']
    arrays=dict(episode=np.array(epi,dtype=np.uint8),anchor=np.array(anchor,dtype=np.uint16),
        endpoint=np.array(endpoint,dtype=np.uint16),delta_steps=np.array(dt,dtype=np.uint16),
        goal_body_m=np.array(goals,dtype=np.float32),train=np.array(split,dtype=np.uint8))
    assert 10000<len(dt)<40000
    return arrays,dict(records=96,pairs=len(dt),train_pairs=int(arrays['train'].sum()),test_pairs=int((1-arrays['train']).sum()),
        current_five_goal_field_max_error=goal_error,excluded_collision_terminal_endpoints=invalid,
        unavailable_requested_future_pairs_masked=missing,records_detail=details)


def features(plan,pairs):
    xx=[];target=[];indices=[];episodes=[]
    for e,r in enumerate(plan['records']):
        ix=np.flatnonzero(pairs['episode']==e)
        with np.load(r['path']) as z:
            k=pairs['anchor'][ix].astype(np.int64);states=z['states'][k];actions=z['actions'][k]
            state=encode(states);g=pairs['goal_body_m'][ix]/10
            x=np.concatenate([state,actions,g],axis=1).astype(np.float32)
        assert x.shape[1]==111 and np.isfinite(x).all()
        xx.append(x);target.append(np.log1p(pairs['delta_steps'][ix].astype(np.float64)*.002).astype(np.float32));indices.extend(ix.tolist());episodes.extend([e]*len(ix))
    order=np.argsort(indices)
    return np.concatenate(xx)[order],np.concatenate(target)[order],np.array(episodes,dtype=np.int16)[order]
