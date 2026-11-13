"""One fixed causal ARX vehicle model, fitted once; no policy optimizer."""
import numpy as np

def velocity(states):
    return np.asarray(states[:,[53,54,55]],np.float64)*[1.,1.,2.]

def features(v,u):
    u=np.clip(np.asarray(u,np.float32),-1.,1.);v=np.asarray(v,np.float64)
    wheel=((u[...,0].astype(np.float64)+1.)*.5)**2*5.
    steer=u[...,1].astype(np.float64)*.4363323129985824
    return np.stack([np.ones_like(wheel),v[...,0],v[...,1],v[...,2],wheel,steer,v[...,0]*np.tan(steer)],-1)

def fit(x,y,ridge):
    mu=x.mean(0);sd=x.std(0);mu[0]=0.;sd[0]=1.;sd=np.maximum(sd,1e-8)
    z=(x-mu)/sd;pen=np.eye(z.shape[1])*ridge;pen[0,0]=0.
    coef=np.linalg.solve(z.T@z+pen,z.T@y)
    assert np.isfinite(coef).all()
    return dict(mean=mu,scale=sd,coefficient=coef)

def rollout(pose,v,actions,dt,m):
    pose=np.asarray(pose,np.float64).copy();v=np.asarray(v,np.float64).copy();rows=[]
    for u,t in zip(actions,dt):
        nv=((features(v,u)-m['mean'])/m['scale'])@m['coefficient']
        ny=pose[2]+.5*(v[2]+nv[2])*t
        c,s=np.cos(pose[2]),np.sin(pose[2]);nc,ns=np.cos(ny),np.sin(ny)
        old=np.array([c*v[0]-s*v[1],s*v[0]+c*v[1]])
        new=np.array([nc*nv[0]-ns*nv[1],ns*nv[0]+nc*nv[1]])
        pose[:2]+=.5*(old+new)*t;pose[2]=ny;rows.append(pose.copy());v=nv
    return np.asarray(rows)

def fixtures():
    rng=np.random.default_rng(533);x=np.c_[np.ones(40),rng.normal(size=(40,6))];w=rng.normal(size=(7,3));y=x@w
    m=fit(x,y,0.);assert np.max(abs(((x-m['mean'])/m['scale'])@m['coefficient']-y))<1e-12
    g=dict(mean=np.zeros(7),scale=np.ones(7),coefficient=np.zeros((7,3)));g['coefficient'][1:4]=np.eye(3)
    out=rollout([0.,0.,0.],[.1,0.,0.],np.zeros((5,2)),np.ones(5)*.2,g)
    assert np.allclose(out[-1],[.1,0.,0.],atol=1e-15)
    assert features(np.zeros(3),[1.,0.])[4]==5.
    return dict(state='PASS',synthetic_checks=3,new_native=0,new_optimizer=0)
