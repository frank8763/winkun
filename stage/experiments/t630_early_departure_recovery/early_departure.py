"""Select one preregistered early departure, using matched physical clocks."""
import math
import numpy as np


def select(candidate, baseline):
    assert np.array_equal(candidate['states'][0],baseline['states'][0])
    assert np.array_equal(candidate['initial_native_integration'],baseline['initial_native_integration'])
    assert np.array_equal(candidate['scene_sha256'],baseline['scene_sha256'])
    t=lambda a: np.r_[0,np.cumsum(a['native_counts'].astype(np.int64))[:-1]]
    ct,bt=t(candidate),t(baseline)
    bytime={int(s):i for i,s in enumerate(bt)}
    for k,s in enumerate(ct):
        if k==0 or k>=len(candidate['states'])-1 or int(s) not in bytime:
            continue
        other=baseline['poses'][bytime[int(s)]]
        pose=candidate['poses'][k]
        xy=float(np.linalg.norm(pose[:2]-other[:2]))
        dyaw=float(pose[2]-other[2])
        yaw=abs(math.atan2(math.sin(dyaw),math.cos(dyaw)))
        if xy>=.02 or yaw>=math.pi/180:
            return dict(first=k,physics_time_steps=int(s),physical_time_s=int(s)*.002,
                        position_departure_m=xy,yaw_departure_rad=yaw,
                        rule='first aligned state at >=0.02m OR >=1degree; no grid')
    return None
