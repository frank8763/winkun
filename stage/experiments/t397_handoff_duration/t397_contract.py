"""Frozen model, fixed verified teacher controls, two declared handoff times."""
from pathlib import Path
import sys,fcntl
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t394_takeover_diagnostic'))
from t394_contract import *
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t397_handoff_duration'
PREVIOUS=ROOT/'stage/runs/t396_iterative_recovery'
TEACHER=PREVIOUS/'teachers/0/teacher_full.npz'

def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    N.capacity(OUT,p['final_bytes'])
    assert N.K.S.used(ROOT/'stage/runs')-N.K.S.used(OUT)+p['final_bytes']<=1024**3
    return N.science(p)

def freeze_optimizer():
    def forbidden(*_a,**_k):raise AssertionError('T397 forbids all optimizer/backward calls')
    torch.optim.Adam.step=forbidden;torch.Tensor.backward=forbidden
    torch.autograd.backward=forbidden;torch.autograd.grad=forbidden

def audit_actions(q,a,p,teacher_holds):
    """Current parameters on complete observed causal history; no native calls."""
    ref=N.S.arrays(REFERENCE);original=N.S.arrays(TEACHER);h=None;q.sample=False;q.reset_noise()
    boundary=p['teacher_start_hold']+teacher_holds
    ownership=np.zeros(len(a['states']),np.uint8);ownership[:265]=1;ownership[267:boundary]=1
    assert equal(ownership,a['ownership']) and not a['stochastic'].any()
    assert equal(a['actions'][:265],ref['actions'][:265])
    assert equal(a['actions'][:boundary],original['actions'][:boundary])
    for k,x in enumerate(a['states']):
        assert np.abs(N.K.H.rebuild(x,a['poses'][k],a['history_poses'][k],a['task_goal_xy'],a['history_goals'][k])-x).max()<=2e-6
        hb=np.zeros(16) if h is None else h[0,0].numpy().copy()
        with torch.no_grad():cmd,_,h,_=q.step(torch.tensor(x[None]),h)
        assert equal(hb,a['hidden_before'][k])
        for key,v in q.tape[-1].items():assert equal(np.asarray(v),a[key][k]),(k,key)
        if ownership[k]==0:assert equal(cmd[0].numpy(),a['actions'][k])
        d0=np.linalg.norm(a['poses'][k,:2]-a['task_goal_xy']);d1=np.linalg.norm(a['next_poses'][k,:2]-a['task_goal_xy'])
        z=N.K.R.GAMMA**int(a['native_counts'][k]);f=a['end_flags'][k]
        reward=-.000002*(1-z)/(1-N.K.R.GAMMA)+.03*d0-z*.03*d1*(not f.any())
        if f.any():reward+=N.K.R.GAMMA**(int(a['native_counts'][k])-1)*(float(f[1])-float(f[0])-.5*float(f[2]))
        assert abs(reward-a['rewards'][k])<1e-12
    assert equal(h[0,0].numpy(),a['last_hidden_after'])
    for key in [*N.R.NAMES,'hidden_before','history_poses','history_goals','history_times']:
        assert equal(a[key][:boundary],original[key][:boundary]),key
    assert np.array_equal(a['states'][1:],a['next_states'][:-1])

def handoff_metrics(a,first):
    goal=a['task_goal_xy'];pose=a['poses'][first];distance=np.linalg.norm(a['next_poses'][first:,:2]-goal,axis=1)
    idx=first+int(distance.argmin());bearing=float(np.arctan2(goal[1]-pose[1],goal[0]-pose[0])-pose[2]);bearing=float(np.arctan2(np.sin(bearing),np.cos(bearing)))
    return dict(handoff_hold=first,handoff_clock_s=float(a['native_counts'][:first].sum()/500),handoff_pose=pose.tolist(),handoff_hidden_sha256=digest(a['hidden_before'][first]),handoff_remaining_m=float(np.linalg.norm(pose[:2]-goal)),goal_bearing_error_deg=float(np.degrees(bearing)),best_recorded_endpoint_m=float(distance.min()),best_recorded_endpoint_clock_s=float(a['native_counts'][:idx+1].sum()/500),autonomous_suffix_physics=int(a['native_counts'][first:].sum()),mode_action_at_handoff=np.clip(a['locations'][first],-1,1).astype(np.float32).tolist())
