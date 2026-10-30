"""Original-goal SAC replay only; no historical Critic or hindsight labels."""
from common import *
GAMMA=math.exp(-.002/60.)

def build(a,timeout_penalty=0.):
    n=a['native_counts'].astype(int);goal=a['goal_xy'];d0=np.linalg.norm(a['poses'][:,:2]-goal,axis=1);d1=np.linalg.norm(a['next_poses'][:,:2]-goal,axis=1)
    if 'end_flags' in a:flags=a['end_flags']
    elif 'physical_flags' in a:flags=a['physical_flags'][(a['steps']+n).astype(int)-1]
    else:flags=a['physical'][(a['steps']+n).astype(int)-1,7:10]
    terminal=np.any(flags,axis=1);assert terminal[-1] and not terminal[:-1].any()
    event=flags[:,1]-flags[:,0]-.5*flags[:,2];discount=GAMMA**n
    reward=-.000002*(1-discount)/(1-GAMMA)+.03*d0-discount*.03*d1*(~terminal)+terminal*GAMMA**(n-1)*event
    reward-=GAMMA**(n-1)*flags[:,2]*.1*timeout_penalty*np.minimum(d1,2.)
    values=0.
    for r,d in zip(reward[::-1],(discount*(~terminal))[::-1]):values=float(r)+float(d)*values
    total=int(n.sum());expected=-.000002*(1-GAMMA**total)/(1-GAMMA)+GAMMA**(total-1)*float(event[-1]-.1*timeout_penalty*flags[-1,2]*min(d1[-1],2.))+.03*d0[0]
    assert abs(values-expected)<1e-7
    return {'states':a['states'].astype(np.float32),'actions':a['actions'].astype(np.float32),'next_states':a['next_states'].astype(np.float32),'rewards':reward.astype(np.float32)[:,None],'discounts':(discount*(~terminal)).astype(np.float32)[:,None],'terminal':terminal.astype(np.float32)[:,None]}

class Replay:
    def __init__(self,seed,timeout_penalty=0.):self.timeout_penalty=timeout_penalty;self.rng=np.random.default_rng(seed);self.episodes=[];self.receipts=[]
    def add(self,path):
        with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        # Convert event masks before subtraction: unsigned uint8 would wrap0-1.
        for k in ['end_flags','physical_flags']:
            if k in a:a[k]=a[k].astype(np.float64)
        item=build(a,self.timeout_penalty);self.episodes.append(item);self.receipts.append({'path':str(path),'sha256':sha(path),'transitions':len(item['states'])})
    def materialize(self):self.orig={k:np.concatenate([e[k] for e in self.episodes]) for k in self.episodes[0]}
    def sample(self,n,device):
        terminal=np.flatnonzero(self.orig['terminal'][:,0]);ids=np.r_[self.rng.integers(len(self.orig['states']),size=n-n//4),self.rng.choice(terminal,size=n//4,replace=True)]
        return {k:torch.tensor(v[ids],device=device) for k,v in self.orig.items()}
