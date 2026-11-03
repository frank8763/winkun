"""Training-only native reverse curriculum, paired SAC with prior real data.

Single causal Actor at deployment. No planner, waypoint or teacher selection.
"""
from pathlib import Path
import sys,copy,gzip,io,json,os,time,traceback,shutil,math,contextlib,importlib.util
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t336_reverse_online';DOC=ROOT/'stage/docs/overnight_20261103'
sys.path.insert(0,str(ROOT/'stage/experiments/t333_remaining_aggregation'))
import t333_learning as L
K,np,torch=L.K,L.np,L.torch
read,save=L.read,L.save
sys.path.insert(0,str(ROOT/'stage/experiments/t280_local_cost'))
import t280_worker as N

def digest(v):return K.S.digest(v)
def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    assert K.S.used(OUT)<=p['final_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']

class Actor(torch.nn.Module):
    def __init__(self,p):
        super().__init__();self.fixed=L.model(read(p['learning_protocol']))
        patch=L.read_patch(p['initial_patch']);L.load(self.fixed,patch['residual'])
        assert digest(self.fixed.state_dict())==p['initial_policy_hash']
        self.fixed.requires_grad_(False).eval();self.memory_on=False
        self.register_buffer('mean',torch.tensor(p['input_mean']))
        self.register_buffer('scale',torch.tensor(p['input_scale']))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(p['seed'])
            self.correction=torch.nn.Sequential(torch.nn.Linear(120,64),torch.nn.Tanh(),torch.nn.Linear(64,2))
            torch.nn.init.zeros_(self.correction[-1].weight);torch.nn.init.zeros_(self.correction[-1].bias)
        self.logstd=torch.nn.Parameter(torch.full((2,),p['initial_logstd']))
    def norm(self,x):return ((x-self.mean)/self.scale).clamp(-10,10)
    def base(self,x):
        with torch.no_grad():
            mode=torch.cat([self.fixed.step(row[None])[0] for row in x])
            return torch.atanh(mode.clamp(-.999999,.999999))
    def latent(self,x,base=None):return (self.base(x) if base is None else base)+self.correction(self.norm(x))
    def step(self,x,hidden=None):
        mu=self.latent(x);return torch.tanh(mu),mu,None,mu
    def sample(self,x,z,base=None):
        mu=self.latent(x,base);ls=self.logstd.clamp(-5.,-.5);u=mu+ls.exp()*z
        action=torch.tanh(u)
        lp=(-.5*z.square()-ls-.5*math.log(2*math.pi)-(2*(math.log(2)-u-torch.nn.functional.softplus(-2*u)))).sum(-1,keepdim=True)
        return action,lp
    def trainable(self):return [*self.correction.parameters(),self.logstd]

class Critic(torch.nn.Module):
    def __init__(self,p):
        super().__init__()
        self.register_buffer('mean',torch.tensor(p['input_mean']));self.register_buffer('scale',torch.tensor(p['input_scale']))
        def net():return torch.nn.Sequential(torch.nn.Linear(122,128),torch.nn.LayerNorm(128),torch.nn.ReLU(),torch.nn.Linear(128,128),torch.nn.LayerNorm(128),torch.nn.ReLU(),torch.nn.Linear(128,1))
        with torch.random.fork_rng(devices=[]):torch.manual_seed(p['seed']+1);self.q1,self.q2=net(),net()
    def forward(self,x,a):
        z=torch.cat([((x-self.mean)/self.scale).clamp(-10,10),a],-1);return self.q1(z),self.q2(z)

class Learner:
    def __init__(self,p):
        self.actor=Actor(p);self.q=Critic(p);self.target=copy.deepcopy(self.q).requires_grad_(False)
        self.ao=torch.optim.Adam(self.actor.trainable(),lr=p['actor_lr'])
        self.qo=torch.optim.Adam(self.q.parameters(),lr=p['critic_lr'])
        self.calls=dict(actor=0,critic=0)
    def state(self):
        return dict(actor={k:v.detach().clone() for k,v in self.actor.state_dict().items() if not k.startswith('fixed.')},
            q=copy.deepcopy(self.q.state_dict()),target=copy.deepcopy(self.target.state_dict()),
            actor_optimizer=copy.deepcopy(self.ao.state_dict()),critic_optimizer=copy.deepcopy(self.qo.state_dict()))
    def load(self,v):
        self.actor.load_state_dict(dict(self.actor.state_dict(),**v['actor']),strict=True)
        self.q.load_state_dict(v['q']);self.target.load_state_dict(v['target']);self.ao.load_state_dict(v['actor_optimizer']);self.qo.load_state_dict(v['critic_optimizer'])
    def actor_state(self):
        return ({k:v.detach().clone() for k,v in self.actor.state_dict().items() if not k.startswith('fixed.')},copy.deepcopy(self.ao.state_dict()))
    def rollback(self,v):
        self.actor.load_state_dict(dict(self.actor.state_dict(),**v[0]),strict=True);self.ao.load_state_dict(v[1])
    def hashes(self):return {k:digest(v) for k,v in self.state().items()}

def packed(a):
    assert np.array_equal(a['states'][1:],a['next_states'][:-1])
    return dict({k:v for k,v in a.items() if k!='next_states'},last_next_state=a['next_states'][-1:])
def arrays(path):
    with np.load(path) as z:a={k:z[k].copy() for k in z.files}
    if 'last_next_state' in a:a['next_states']=np.concatenate([a['states'][1:],a.pop('last_next_state')])
    return a
def tensor_data(a,actor):
    x=torch.tensor(a['states'],dtype=torch.float32);nx=torch.tensor(a['next_states'],dtype=torch.float32)
    terminal=np.asarray(a['end_flags']).any(-1)
    # True collision/GOAL/task cap terminate; incomplete administrative episodes
    # never enter either replay buffer.
    assert terminal[-1] and not terminal[:-1].any()
    return dict(x=x,nx=nx,a=torch.tensor(a['actions'],dtype=torch.float32),r=torch.tensor(a['rewards'],dtype=torch.float32)[:,None],
        discount=torch.tensor((K.R.GAMMA**a['native_counts'].astype(np.int64))*(~terminal),dtype=torch.float32)[:,None],
        base=actor.base(x),nextbase=actor.base(nx))
def combine(rows):return {k:torch.cat([a[k] for a in rows]) for k in rows[0]}
def prior(p,actor):return combine([tensor_data(arrays(path),actor) for path in p['prior_trajectories']])
def anchors(p,actor):
    with np.load(read(p['base_protocol'])['retention_states']) as z:
        x=torch.tensor(z['states'].copy());offs=z['offsets'].copy();tasks=z['task'].copy()
    assert len(x)==28629 and len(tasks)==211
    task=np.concatenate([np.full(int(offs[k+1]-offs[k]),int(tasks[k]),np.int64) for k in range(211)])
    base=actor.base(x);return dict(x=x,base=base,mode=torch.tanh(base),by_task=[np.flatnonzero(task==k) for k in range(8)])

def optimize(p,learner,prior_data,online,old,seed,steps,actor_update,guard,event):
    rng=np.random.default_rng(seed);gen=torch.Generator().manual_seed(seed);rows=[]
    for k in range(steps):
        guard();ids0=rng.integers(len(prior_data['x']),size=p['batch_size']//2);ids1=rng.integers(len(online['x']),size=p['batch_size']//2)
        data={name:torch.cat([prior_data[name][ids0],online[name][ids1]]) for name in prior_data}
        with torch.no_grad():
            na,nlp=learner.actor.sample(data['nx'],torch.randn((p['batch_size'],2),generator=gen),data['nextbase'])
            tq1,tq2=learner.target(data['nx'],na)
            target=data['r']+data['discount']*(torch.minimum(tq1,tq2)-p['alpha']*nlp)
        q1,q2=learner.q(data['x'],data['a']);qloss=(q1-target).square().mean()+(q2-target).square().mean()
        assert torch.isfinite(qloss);learner.qo.zero_grad(set_to_none=True);qloss.backward()
        torch.nn.utils.clip_grad_norm_(learner.q.parameters(),1.,error_if_nonfinite=True);learner.qo.step();learner.calls['critic']+=1
        with torch.no_grad():
            for dest,src in zip(learner.target.parameters(),learner.q.parameters()):dest.lerp_(src,p['tau'])
        aloss=keep=None
        if actor_update:
            learner.q.requires_grad_(False)
            try:
                a,lp=learner.actor.sample(data['x'],torch.randn((p['batch_size'],2),generator=gen),data['base']);a1,a2=learner.q(data['x'],a)
                ids=np.concatenate([rng.choice(v,size=p['old_rows_per_task']) for v in old['by_task']])
                keep=(torch.tanh(learner.actor.latent(old['x'][ids],old['base'][ids]))-old['mode'][ids]).square().mean()
                aloss=(p['alpha']*lp-torch.minimum(a1,a2)).mean()+p['retention_weight']*keep
                assert torch.isfinite(aloss);learner.ao.zero_grad(set_to_none=True);aloss.backward()
                torch.nn.utils.clip_grad_norm_(learner.actor.trainable(),1.,error_if_nonfinite=True);learner.ao.step();learner.calls['actor']+=1
                with torch.no_grad():learner.actor.logstd.clamp_(-5.,-.5)
            finally:learner.q.requires_grad_(True)
        assert all(v.grad is None for v in learner.actor.fixed.parameters())
        if (k+1)%100==0 or k+1==steps:
            row=dict(step=k+1,critic_loss=float(qloss.detach()),actor_loss=None if aloss is None else float(aloss.detach()),
                retention=None if keep is None else float(keep.detach()),hashes=learner.hashes(),actor_Adam_calls=k+1 if actor_update else 0,critic_Adam_calls=k+1)
            rows.append(row);event(row)
    return rows

def export(p,learner,path):
    buf=io.BytesIO();torch.save(dict(state=learner.state(),hashes=learner.hashes(),fixed_input_sha256=K.C.sha(p['initial_patch'])),buf)
    blob=gzip.compress(buf.getvalue(),mtime=0);assert len(blob)<=p['model_bytes'];Path(path).write_bytes(blob)
    return dict(path=str(path),sha256=K.C.sha(path),hashes=learner.hashes(),bytes=len(blob))
