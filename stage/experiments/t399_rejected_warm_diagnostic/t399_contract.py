"""Frozen rejected warm candidate: fill missing task measurements, Stage only."""
from pathlib import Path
import sys,fcntl
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t394_takeover_diagnostic'))
from t394_contract import *
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t399_rejected_warm_diagnostic'
PRIOR=ROOT/'stage/runs/t398_warm_recovery_rl'
MODEL=PRIOR/'warm/proposed.pt.gz'

def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    N.capacity(OUT,p['final_bytes'])
    assert N.K.S.used(ROOT/'stage/runs')-N.K.S.used(OUT)+p['final_bytes']<=1024**3
    return N.science(p)

def frozen(base,p):
    q=C.Actor(base,p).eval().requires_grad_(False)
    q.load_learned(N.S.read_model(MODEL));q.sample=False;q.reset_noise()
    assert q.phash()==p['policy_hash']
    return q

def forbid_learning():
    def stop(*a,**k):raise AssertionError('T399 is zero optimizer and zero backward')
    torch.optim.Adam.step=stop;torch.Tensor.backward=stop
    torch.autograd.backward=stop;torch.autograd.grad=stop

def audit_array(q,a,first):
    h=None;q.reset_noise();q.sample=False
    ownership=np.zeros(len(a['states']),np.uint8);ownership[:first]=1
    assert equal(ownership,a['ownership']) and not a['stochastic'].any()
    ref=N.S.arrays(REFERENCE);assert equal(a['actions'][:first],ref['actions'][:first])
    with torch.no_grad():
        for k,x in enumerate(a['states']):
            assert np.abs(N.K.H.rebuild(x,a['poses'][k],a['history_poses'][k],a['task_goal_xy'],a['history_goals'][k])-x).max()<=2e-6
            hb=np.zeros(16) if h is None else h[0,0].numpy().copy()
            action,_,h,_=q.step(torch.tensor(x[None]),h)
            assert equal(hb,a['hidden_before'][k])
            for name,value in q.tape[-1].items():assert equal(np.asarray(value),a[name][k]),(k,name)
            if k>=first:assert equal(action[0].numpy(),a['actions'][k])
            n=int(a['native_counts'][k]);z=N.K.R.GAMMA**n;flags=a['end_flags'][k]
            d0=np.linalg.norm(a['poses'][k,:2]-a['task_goal_xy']);d1=np.linalg.norm(a['next_poses'][k,:2]-a['task_goal_xy'])
            reward=-.000002*(1-z)/(1-N.K.R.GAMMA)+.03*d0-z*.03*d1*(not flags.any())
            if flags.any():reward+=N.K.R.GAMMA**(n-1)*(float(flags[1])-float(flags[0])-.5*float(flags[2]))
            assert abs(reward-a['rewards'][k])<1e-12
    assert equal(h[0,0].numpy(),a['last_hidden_after'])
    assert np.array_equal(a['states'][1:],a['next_states'][:-1])
    if first:
        for key in N.R.NAMES:assert equal(a[key][:first],ref[key][:first]),key

