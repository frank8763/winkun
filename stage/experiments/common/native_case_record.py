"""Explicit request/result envelope for native navigation workers.

Native workers return physical results without the caller's job identifier.
Attach that trusted request identifier before any legacy derived metrics.
"""
import copy
import numpy as np

def envelope(job, native_result, arrays, field_limit):
    assert isinstance(job['id'],str) and job['id']
    row=copy.deepcopy(native_result)
    if 'job_id' in row:assert row['job_id']==job['id'],'Conflicting request/result task identity'
    row['job_id']=job['id']
    n=len(arrays['actions']);assert n>0 and arrays['poses'].shape==(n,3) and arrays['next_poses'].shape==(n,3)
    assert arrays['states'].shape==(n,120) and arrays['next_states'].shape==(n,120)
    assert np.isfinite(arrays['poses']).all() and np.isfinite(arrays['next_poses']).all()
    flags=np.asarray(arrays['end_flags']);assert flags.shape==(n,3) and not flags[:-1].any() and flags[-1].sum()==1
    assert flags[-1].tolist()==[int(row['collision']),int(row['success']),int(row['timeout'])]
    assert int(arrays['native_counts'].sum())==row['physics']
    assert row['complete'] and not row['administrative_censor']
    poses=np.vstack([arrays['poses'],arrays['next_poses'][-1]])
    crossed=bool((np.abs(poses[:,:2])>float(field_limit)).any())
    row['field_detail']=dict(id=job['id'],field_boundary_crossed=crossed,field_qualified_success=bool(row['success'] and not row['collision'] and not crossed))
    # These fields are enough to preserve success/termination and qualification
    # even if optional rich diagnostics later fail. Persist this envelope first.
    row['observation_contract']=dict(decisions=n,pose_min_xy=poses[:,:2].min(0).tolist(),pose_max_xy=poses[:,:2].max(0).tolist(),input_dim=120)
    return row
