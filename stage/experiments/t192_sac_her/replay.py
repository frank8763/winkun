"""Finite-horizon transition labels with exact per-physics first-hit boundaries."""
import math
from common import *
GAMMA=math.exp(-.002/60.)

def features(pose,goal):
    x,y,yaw=map(float,pose[:3]);dx,dy=map(float,np.asarray(goal)-[x,y]);c=math.cos(yaw);s=math.sin(yaw);bx=c*dx+s*dy;by=-s*dx+c*dy;d=math.hypot(dx,dy);angle=math.atan2(by,bx)
    return np.array([np.clip(d/10,0,2),math.sin(angle),math.cos(angle),np.clip(bx/10,-2,2),np.clip(by/10,-2,2)],dtype=np.float32),d,angle

def relabel(obs,poses,hposes,goal):
    result=np.asarray(obs,dtype=np.float32).copy()
    for i in range(len(result)):
        result[i,48:53]=features(poses[i],goal)[0]
        for j in range(4):
            _,d,a=features(hposes[i,j],goal);result[i,72+12*j]=np.clip(d/10,0,2);result[i,73+12*j]=np.clip(a/math.pi,-1,1)
    return result

def data(path):
    with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    if 'physical' in a:
        a['physical_xy']=a['physical'][:,1:3];a['physical_flags']=a['physical'][:,7:10].astype(np.uint8)
    return a

def build(a,goal=None):
    original=goal is None;goal=a['goal_xy'] if original else np.asarray(goal,dtype=np.float64)
    xy=a['physical_xy'];flags=a['physical_flags'];steps=len(xy)
    if not original:
        if np.linalg.norm(a['poses'][0,:2]-goal)<=.2:return None
        distances=np.linalg.norm(xy-goal,axis=1);hits=np.flatnonzero((distances<=.2)&~flags[:,0].astype(bool))
        if not len(hits):return None
        end=int(hits[0]+1)
        if flags[:end,0].any():return None
        event=1.
    else:
        end=steps;event=1. if flags[-1,1] else -1. if flags[-1,0] else -.5
    starts=a['steps'].astype(int);ids=np.flatnonzero(starts<end);counts=np.minimum(a['native_counts'][ids].astype(int),end-starts[ids]);ends=starts[ids]+counts
    terminal=ends==end
    states=a['states'][ids].astype(np.float32).copy();next_states=a['next_states'][ids].astype(np.float32).copy()
    if not original:
        states=relabel(states,a['poses'][ids],a['history_poses'][ids],goal)
        next_states=relabel(next_states,a['next_poses'][ids],a['next_history_poses'][ids],goal)
    d0=np.linalg.norm(a['poses'][ids,:2]-goal,axis=1);d1=np.linalg.norm(xy[ends-1]-goal,axis=1)
    discount=GAMMA**counts
    # normalized rewards: exact per-physics sum, potential Phi=-.3distance /10.
    reward=-.000002*(1-discount)/(1-GAMMA)+.03*d0-discount*.03*d1*(~terminal)
    reward+=terminal*(GAMMA**(counts-1))*event
    discount=discount*(~terminal)
    # Interior-terminal next observation was never sampled at that instant.
    # It is explicitly unused: zero bootstrap and no next-state actor/entropy losses.
    exact=(counts==a['native_counts'][ids])
    assert np.all(exact|terminal)
    return {'states':states,'actions':a['actions'][ids].astype(np.float32),'next_states':next_states,'rewards':reward.astype(np.float32)[:,None],'discounts':discount.astype(np.float32)[:,None],'terminal':terminal.astype(np.float32)[:,None],'native_counts':counts.astype(np.int32),'next_state_exact':exact.astype(np.uint8)}

class Replay:
    def __init__(self,her,seed):
        self.her=her;self.rng=np.random.default_rng(seed);self.original=[];self.hindsight=[];self.receipts=[]
    def add(self,path):
        a=data(path);original=build(a);assert original is not None;self.original.append(original)
        future=np.unique(np.linspace(max(2,len(a['steps'])//4),len(a['steps'])-1,8).astype(int));eligible=0
        for i in future:
            end=int(a['steps'][i]+a['native_counts'][i]);goal=a['physical_xy'][end-1]
            if a['physical_flags'][end-1,0]:continue
            tr=build(a,goal)
            if tr is not None:
                eligible+=1
                if self.her:self.hindsight.append(tr)
        self.receipts.append({'path':str(path),'sha256':sha(path),'original_transitions':len(original['states']),'eligible_future_goals':eligible,'her_added':eligible if self.her else 0})
    def materialize(self):
        self.orig={k:np.concatenate([a[k] for a in self.original]) for k in self.original[0]}
        self.herdata={k:np.concatenate([a[k] for a in self.hindsight]) for k in self.hindsight[0]} if self.hindsight else None
    def sample(self,n,device):
        # Uniform75% plus terminal-balanced25%, same rule for both arms.
        parts=[]
        for group,size in [(self.orig,n//2 if self.herdata is not None else n),(self.herdata,n-n//2 if self.herdata is not None else 0)]:
            if not size:continue
            terminal=np.flatnonzero(group['terminal'][:,0]);balanced=size//4
            ids=np.r_[self.rng.integers(len(group['states']),size=size-balanced),self.rng.choice(terminal,size=balanced,replace=True)]
            parts.append({k:v[ids] for k,v in group.items() if k in ['states','actions','next_states','rewards','discounts','terminal']})
        return {k:torch.tensor(np.concatenate([p[k] for p in parts]),device=device) for k in parts[0]}
