"""Frozen-Actor conditional-start diagnostic; never a learning routine."""
from pathlib import Path
import sys,fcntl
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t393_safe_demo_pg'))
from t393_learning import *
import t393_learning as C
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t394_takeover_diagnostic'
PRIOR=ROOT/'stage/runs/t393_safe_demo_pg'
PARENT=PRIOR/'candidate.pt.gz'

def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    N.capacity(OUT,p['final_bytes'])
    assert N.K.S.used(ROOT/'stage/runs')-N.K.S.used(OUT)+p['final_bytes']<=1024**3
    return N.science(p)

def actor(base,p):
    q=C.Actor(base,p).eval().requires_grad_(False);q.load_learned(N.S.read_model(PARENT));q.reset_noise();q.sample=False
    assert q.phash()==p['parent_policy_hash']
    return q

def freeze_optimizer():
    def forbidden(*_a,**_k):raise AssertionError('No optimizer or backward in T394')
    torch.optim.Adam.step=forbidden
    torch.Tensor.backward=forbidden
    torch.autograd.backward=forbidden

def payload(session):
    a=session.arrays()
    a.update(hold_sha256=np.asarray(session.digests,np.uint8),ownership=np.asarray(session.owner,np.uint8),
        task_goal_xy=session.goal.copy(),scene_sha256=np.frombuffer(bytes.fromhex(session.scene),np.uint8).copy())
    return a

def measures(a,first):
    suffix=a['end_flags'][first:];ret=total(a,first)
    goal=a['task_goal_xy'];distance=np.linalg.norm(a['poses'][first:,:2]-goal,axis=1)
    nd=np.linalg.norm(a['next_poses'][first:,:2]-goal,axis=1)
    executed=a['actions'][first:];mode=np.clip(a['locations'][first:],-1,1).astype(np.float32)
    return dict(return_G0=ret,takeover_remaining_m=float(distance[0]),
        best_recorded_endpoint_m=float(min(distance.min(),nd.min())),
        suffix_physics=int(a['native_counts'][first:].sum()),prefix_physics=int(a['native_counts'][:first].sum()),
        command_sd=executed.std(0).tolist(),abs_sample_minus_mode_p50=np.quantile(np.abs(executed-mode),.5,axis=0).tolist(),
        executed_boundary_fraction=(np.abs(executed)==1).mean(0).tolist())
