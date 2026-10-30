"""Exact boundary-state checks and bounded, owned checkpoint rotation."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import numpy as np
import torch

def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return isinstance(b,(list,tuple)) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def verify_restoration(agent,replay,rng,episodes,parent,plan,teacher,root):
    checks={}
    for key in ['actor','actor_target','critic','critic_target','actor_optimizer','critic_optimizer']:
        checks[key]=equal(getattr(agent,key).state_dict(),parent[key])
    checks['replay']=equal(list(replay),parent['replay'])
    checks['numpy_rng']=equal(rng.bit_generator.state,parent['numpy_rng'])
    checks['torch_rng']=torch.equal(torch.get_rng_state(),parent['torch_rng'])
    checks['episodes']=equal(episodes,parent['episodes'])
    checks['total_updates']=agent.total_updates==parent['total_updates']
    checks['gamma']=agent.gamma==plan['gamma']
    teacher_parent=torch.load(plan['initial_actor'],map_location='cpu',weights_only=False)
    checks['fixed_T161_retention_teacher']=equal(teacher.state_dict(),teacher_parent['actor'])
    checks['remaining_budget']=plan['new_frames']>parent['new_physics_frames']
    assert all(checks.values()),checks
    return {'state':'PASS','checks':checks,'resume_checkpoint':plan['resume_checkpoint'],
        'resume_sha256':plan['resume_sha256'],'original_frames':parent['new_physics_frames'],
        'remaining_frames':plan['new_frames']-parent['new_physics_frames'],
        'replay_entries':len(replay),'episode_boundary_restored':True,
        'not_bitwise_uninterrupted_run':'Migration discarded unsaved later progress; this continues the exact saved boundary state. No live mid-episode state claimed.'}

def guard_storage(root):
    run=root.parent
    used=sum(p.stat().st_size for p in run.rglob('*') if p.is_file() and not p.is_symlink())
    assert used<1024**3,('RUN_DISK_CAP_1_GIB',used)
    assert shutil.disk_usage(run).free>40*1024**3,'MINIMUM_FREE_DISK_40_GIB'

def rotate_checkpoint(root,state):
    guard_storage(root)
    latest=root/'checkpoint_latest.pt';previous=root/'checkpoint_previous.pt';tmp=root/'checkpoint_partial.pt'
    torch.save(state,tmp)
    assert tmp.stat().st_size<180*1024**2,'CHECKPOINT_TOO_LARGE'
    # These names are owned by this fresh run only. Archived inputs are never mutated.
    if previous.exists():previous.unlink()
    if latest.exists():os.replace(latest,previous)
    os.replace(tmp,latest)
    guard_storage(root)
    return latest
