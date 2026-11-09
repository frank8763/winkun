"""Plain causal Actor, training-only independent V, offline BC then own PPO."""
from pathlib import Path
import copy,gzip,io,math,sys,time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t441_joint_actor_critic'))
import t441_learning as L
torch,np,E=L.torch,L.np,L.E
read,save,sha,digest=L.read,L.save,L.sha,L.digest

class Actor(torch.nn.Module):
    def __init__(self,p):
        super().__init__();self.p=p
        self.register_buffer('mean',torch.tensor(p['input_mean'],dtype=torch.float64))
        self.register_buffer('scale',torch.tensor(p['input_scale'],dtype=torch.float64))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(p['actor_seed'])
            self.encoder=torch.nn.Sequential(torch.nn.Linear(120,128),torch.nn.Tanh()).double()
            self.core=torch.nn.GRU(128,64).double();self.head=torch.nn.Linear(64,2).double()
            torch.nn.init.zeros_(self.head.weight);torch.nn.init.zeros_(self.head.bias)
        self.sample=False;self.rng=None;self.reset_noise()
    def normalized(self,x):return ((x.double()-self.mean)/self.scale).clamp(-10,10)
    def sequence(self,x):
        # x [time,batch,120] or [time,120], always complete history from h0.
        single=x.ndim==2
        if single:x=x[:,None]
        h,_=self.core(self.encoder(self.normalized(x)));loc=1.25*torch.tanh(self.head(h))
        return loc[:,0] if single else loc
    def reset_noise(self):self.tape=[];self.previous_raw=self.previous_location=None
    def phash(self):return digest(dict(architecture='plain120_128_64GRU_2',state=self.state_dict(),sigma=self.p['raw_sigma']))
    def step(self,x,hidden=None):
        if hidden is None:hidden=x.new_zeros((1,1,64),dtype=torch.float64)
        h,new=self.core(self.encoder(self.normalized(x))[None],hidden);loc=1.25*torch.tanh(self.head(h[0]))
        z=np.zeros((1,2)) if not self.sample else self.rng.normal(size=(1,2))
        raw=loc.detach()+loc.new_tensor(z)*loc.new_tensor(self.p['raw_sigma'])
        atom=torch.where(raw>=L.EDGE,1,torch.where(raw<=-L.EDGE,-1,0)).to(torch.int8)
        action=raw.clamp(-1,1).float();assert torch.equal(atom.abs()==1,action.abs()==1)
        lp=L.density(loc,raw,atom,self.p['raw_sigma']) if self.sample else torch.zeros(1,dtype=torch.float64)
        self.tape.append(dict(innovations=z[0].copy(),raw_actions=raw[0].cpu().numpy().copy(),atom=atom[0].cpu().numpy().copy(),
            locations=loc[0].detach().cpu().numpy().copy(),logp=lp[0].detach().cpu().numpy().copy(),stochastic=np.asarray(self.sample,np.uint8)))
        self.previous_raw=raw;self.previous_location=loc.detach();return action,loc,new,loc

def make(p,device='cuda'):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    q=Actor(p).to(device);v=L.Value(q,p['value_seed']).to(device)
    q.core.train();v.core.train()
    ao=torch.optim.Adam(q.parameters(),lr=p['bc_lr'],foreach=False,fused=False)
    vo=torch.optim.Adam(v.parameters(),lr=p['value_lr'],foreach=False,fused=False)
    return q,v,ao,vo

def state(*obj):
    q,v,ao,vo=obj;return L.cpu(dict(actor=q.state_dict(),value=v.state_dict(),actor_opt=ao.state_dict(),value_opt=vo.state_dict()))
def load(obj,s):
    q,v,ao,vo=obj;q.load_state_dict(s['actor']);v.load_state_dict(s['value']);ao.load_state_dict(copy.deepcopy(s['actor_opt']));vo.load_state_dict(copy.deepcopy(s['value_opt']))
def export(obj,path,limit):
    s=state(*obj);buf=io.BytesIO();torch.save(dict(joint_state=s,policy_hash=obj[0].phash(),joint_hash=digest(s),architecture='plain120_128_64GRU_2',single_actor=True),buf)
    b=gzip.compress(buf.getvalue(),mtime=0);assert len(b)<=limit,(len(b),limit);Path(path).write_bytes(b)
    return dict(path=str(path),sha256=sha(path),policy_hash=obj[0].phash(),joint_hash=digest(s),bytes=len(b))
def checkpoint(path):return torch.load(io.BytesIO(gzip.decompress(Path(path).read_bytes())),map_location='cpu',weights_only=False)
def frozen(p,path):
    q=Actor(p).eval().requires_grad_(False);v=checkpoint(path);q.load_state_dict(v['joint_state']['actor']);assert q.phash()==v['policy_hash'];return q

def old_data(p):
    import json
    rows={r['source_reset_id']:r for r in map(json.loads,gzip.decompress(Path(p['old_rows']).read_bytes()).decode().splitlines())};out=[]
    with np.load(p['old_states']) as z:
        for i,name in enumerate(z['case_ids']):
            name=str(name);r=rows[name];assert r['success'] and not r['collision'];assert sha(r['path'])==r['sha256']
            with np.load(r['path']) as a:
                x=z['states'][z['offsets'][i]:z['offsets'][i+1]].copy();y=a['actions'].copy();assert len(x)==len(y)
                assert digest(x)==r['array_hashes']['states'] and digest(y)==r['array_hashes']['actions']
                assert np.array_equal(x[1:,-3:-1],y[:-1])
            out.append(dict(id=name,task=int(z['task'][i]),x=x,y=y))
    assert len(out)==211;return out

def dataset(p,teacher_paths):
    old=old_data(p);wide=[]
    for row in teacher_paths:
        assert sha(row['path'])==row['sha256'];a=E.N.S.arrays(row['path']);assert a['end_flags'][-1,1] and not a['end_flags'][:,0].any()
        wide.append(dict(id=row['job_id'],task=8 if '_LEFT_' in row['job_id'] else 9,x=a['states'],y=a['actions']))
    assert len(wide)==8;return old+wide

def packs(data,device):
    return [dict(g,x=torch.tensor(g['x'],device=device),y=torch.tensor(g['y'],device=device,dtype=torch.float64)) for g in data]

def action_loss(loc,target):
    # Interior executed deterministic targets, boundary feasible-set hinge.
    # This is imitation, not a fabricated Gaussian density for expert actions.
    return torch.where(target>=1,torch.relu(1-loc).square(),torch.where(target<=-1,torch.relu(1+loc).square(),(loc-target).square()))

def schedule(data,steps):
    pools=[[i for i,g in enumerate(data) if g['task']==t] for t in range(10)];assert all(pools)
    return [[pool[k%len(pool)] for pool in pools] for k in range(steps)]

def bc_step(obj,groups,indices):
    q,v,ao,vo=obj;selected=[groups[i] for i in indices];lengths=[len(g['x']) for g in selected]
    x=torch.nn.utils.rnn.pad_sequence([g['x'] for g in selected]);y=torch.nn.utils.rnn.pad_sequence([g['y'] for g in selected]);loc=q.sequence(x)
    # old8 groups total1/2, wideLEFT/RIGHT total1/2, episode means.
    losses=[action_loss(loc[:n,i],y[:n,i]).mean() for i,n in enumerate(lengths)]
    loss=sum(losses[:8])/16+sum(losses[8:])/4
    ao.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(q.parameters(),1.,error_if_nonfinite=True));ao.step()
    return dict(loss=float(loss.detach()),gradient_norm=norm)

def assess(q,data):
    result={};device=next(q.parameters()).device
    with torch.no_grad():
        for task in range(10):
            seq=[g for g in data if g['task']==task];errors=[];first=[]
            for g in seq:
                x=torch.as_tensor(g['x'],device=device);y=torch.as_tensor(g['y'],device=device,dtype=torch.float64)
                err=(q.sequence(x).clamp(-1,1).float().double()-y).abs();errors.append(float(err.square().mean()));first.append(float(err[:10].max()))
            result[str(task)]=dict(episodes=len(seq),episode_mean_RMSE=math.sqrt(float(np.mean(errors))),maximum_first10_action_error=max(first))
    return result

def ppo(obj,arrays,p,guard,event):
    q,v,ao,vo=obj;device=next(q.parameters()).device;batch=[];errors=[]
    for i,a in enumerate(arrays):
        assert not np.asarray(a['ownership']).any() and np.asarray(a['stochastic']).all()
        assert np.array_equal(a['states'][1:],a['next_states'][:-1])
        assert not np.asarray(a['end_flags'])[:-1].any() and np.asarray(a['end_flags'])[-1].sum()==1
        x=torch.tensor(a['states'],device=device);raw=torch.tensor(a['raw_actions'],device=device);atom=torch.tensor(a['atom'],device=device);old=torch.tensor(a['logp'],device=device)
        with torch.no_grad():
            loc=q.sequence(x);er=float((loc-loc.new_tensor(a['locations'])).abs().max());lp=float((L.density(loc,raw,atom,p['raw_sigma'])-old).abs().max());assert er<1e-10 and lp<1e-9
            extended=np.concatenate([a['states'],a['next_states'][-1:]]);vv=v(torch.tensor(extended,device=device)).cpu().numpy()
        mask=np.ones((len(x),1),bool);mask[-1]=False
        adv,target=L.physical_gae(a['rewards'][:,None],vv[:-1,None],vv[1:,None],a['native_counts'].astype(np.int64)[:,None],mask,mask,
            gamma_native=p['gamma_native'],lambda_native=p['lambda_native'])
        batch.append(dict(x=x,raw=raw,atom=atom,old=old,adv=adv[:,0],target=loc.new_tensor(target[:,0]),group=i//2));errors.append([er,lp])
    for group in [0,1]:
        selected=[b for b in batch if b['group']==group];mean=np.mean([b['adv'].mean() for b in selected]);second=np.mean([np.mean(b['adv']**2) for b in selected]);sd=math.sqrt(max(0,second-mean**2))+1e-8
        for b in selected:b['adv']=b['old'].new_tensor((b['adv']-mean)/sd)
    trace=[]
    for k in range(p['epochs']):
        guard();ao.zero_grad(set_to_none=True);vo.zero_grad(set_to_none=True);pg=vl=0.
        for b in batch:
            ratio=(L.density(q.sequence(b['x']),b['raw'],b['atom'],p['raw_sigma'])-b['old']).exp()
            loss=-torch.minimum(ratio*b['adv'],ratio.clamp(.8,1.2)*b['adv']).mean()/4;loss.backward();pg+=float(loss.detach())
            loss=.5*(v(b['x'])-b['target']).square().mean()/4;loss.backward();vl+=float(loss.detach())
        an=float(torch.nn.utils.clip_grad_norm_(q.parameters(),1.,error_if_nonfinite=True));vn=float(torch.nn.utils.clip_grad_norm_(v.parameters(),1.,error_if_nonfinite=True))
        event('BEFORE',k);ao.step();vo.step();event('AFTER',k)
        with torch.no_grad():
            kl=sum(float(((z:=(L.density(q.sequence(b['x']),b['raw'],b['atom'],p['raw_sigma'])-b['old'])).exp()-1-z).mean()) for b in batch)/4
        trace.append(dict(epoch=k,policy_loss=pg,value_loss=vl,actor_grad_norm=an,value_grad_norm=vn,KL=kl))
        if kl>p['target_kl']:break
    return dict(trace=trace,density_errors=errors,Actor_Adam=len(trace),V_Adam=len(trace))
