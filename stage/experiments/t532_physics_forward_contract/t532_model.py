"""Unfitted rear-axle bicycle candidate; NOT the deployed vehicle simulator.

This is only the forward admissibility gate for a proposed physical gradient.
Recorded controls are held fixed. No sensor/GRU/closed-loop gradient is claimed.
"""
import math
import numpy as np


def advance(pose, action, dt, geometry):
    """Exact constant-curvature arc of the rear axle, then body reference point."""
    x, y, yaw = map(float, pose)
    drive, turn = np.clip(np.asarray(action, dtype=np.float32), -1., 1.)
    wheel = ((float(drive) + 1.) / 2.) ** 2 * geometry['max_wheel_rad_s']
    v = wheel * geometry['rear_radius_m']
    steer = float(turn) * geometry['max_steer_rad']
    w = v * math.tan(steer) / geometry['wheelbase_m']
    d = w * float(dt)
    distance = v * float(dt) * np.sinc(d / (2. * math.pi))
    rx, ry = geometry['rear_center_body_xy']
    # Rotate the offset from rear axle to the tracked free-body origin.
    ox, oy = -rx, -ry
    nx = x + distance * math.cos(yaw + d / 2.)
    ny = y + distance * math.sin(yaw + d / 2.)
    nx += ox * (math.cos(yaw+d)-math.cos(yaw)) - oy * (math.sin(yaw+d)-math.sin(yaw))
    ny += ox * (math.sin(yaw+d)-math.sin(yaw)) + oy * (math.cos(yaw+d)-math.cos(yaw))
    return np.array([nx, ny, yaw+d], dtype=np.float64)


def rollout(pose, actions, dt, geometry):
    rows=[]
    for u, seconds in zip(actions, dt):
        pose=advance(pose, u, seconds, geometry)
        rows.append(pose)
    return np.asarray(rows)


def constant_twist(pose, body_velocity, yaw_rate, dt):
    """Causal no-action baseline, initialized from the first recorded input only."""
    x,y,yaw=map(float,pose);vx,vy=map(float,body_velocity);rows=[]
    for seconds in dt:
        d=float(yaw_rate)*float(seconds)
        scale=float(seconds)*np.sinc(d/(2.*math.pi));mid=yaw+d/2.
        x+=scale*(vx*math.cos(mid)-vy*math.sin(mid))
        y+=scale*(vx*math.sin(mid)+vy*math.cos(mid));yaw+=d
        rows.append([x,y,yaw])
    return np.asarray(rows)


def errors(prediction, target):
    xy=np.linalg.norm(prediction[:,:2]-target[:,:2],axis=1)
    d=prediction[:,2]-target[:,2]
    yaw=np.abs(np.arctan2(np.sin(d),np.cos(d)))*180./math.pi
    return dict(xy_m=xy.tolist(),yaw_deg=yaw.tolist(),max_xy_m=float(xy.max()),max_yaw_deg=float(yaw.max()))


def fixtures():
    g=dict(max_wheel_rad_s=5.,rear_radius_m=.03,max_steer_rad=.4,wheelbase_m=.6,rear_center_body_xy=[-.3,0.])
    assert np.allclose(advance([0.,0.,0.],[1.,0.],1.,g),[.15,0.,0.],atol=1e-15)
    assert np.array_equal(advance([1.,2.,.3],[-1.,1.],1.,g),[1.,2.,.3])
    a=advance([0.,0.,0.],[1.,.5],1.,g);b=advance([0.,0.,0.],[1.,-.5],1.,g)
    assert np.allclose(a,b*[1.,-1.,-1.],atol=1e-15)
    assert np.allclose(rollout([0.,0.,0.],[[1.,.5]]*5,[.2]*5,g)[-1],a,atol=1e-15)
    err=errors(np.array([[0.,0.,2.*math.pi]]),np.zeros((1,3)))
    assert err['max_xy_m']==0. and err['max_yaw_deg']<1e-12
    # Explicit negative fixture: a 3 cm error must fail the 2 cm gate.
    assert errors(np.array([[.03,0.,0.]]),np.zeros((1,3)))['max_xy_m']>.02
    return dict(state='PASS',synthetic_checks=6,physics=0,optimizer=0)
