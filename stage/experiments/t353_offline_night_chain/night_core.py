"""Precommitted, network-independent recurrent navigation experiments."""
from pathlib import Path
import sys,math,signal,multiprocessing as mp,subprocess,hashlib,fcntl
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t351_recurrent_recovery'))
from t351_core import *
import t351_core as P
HERE=Path(__file__).resolve().parent;DOC=ROOT/'stage/docs/overnight_20261104'


def read_model(path):
    return torch.load(io.BytesIO(gzip.decompress(Path(path).read_bytes())),weights_only=False,map_location='cpu')['state']['learned']


def conditional_locations(loc,raw,rho):
    if not rho:return loc
    return torch.cat([loc[:1],loc[1:]+rho*(raw[:-1]-loc[:-1])])


def raw_logp(loc,raw,rho,sigma):
    mean=conditional_locations(loc,raw,rho)
    sd=torch.tensor(sigma,dtype=torch.float64).expand_as(loc).clone()
    if rho:sd[1:]*=math.sqrt(1-rho*rho)
    return (-.5*((raw-mean)/sd).square()-sd.log()-.5*math.log(2*math.pi)).sum(-1)


class NightActor(P.C.Actor):
    def __init__(self,p,rho):
        super().__init__(p,'memory');self.p=p;self.rho=rho;self.sample=False;self.rng=None;self.reset_noise()
        self.load_learned(read_model(p['initial_model']))
    def reset_noise(self):self.previous_raw=None;self.previous_location=None;self.tape=[]
    def phash(self):return digest(dict(actor=super().phash(),rho=self.rho,raw_sigma=self.p['raw_sigma'],density='full_latent_rawGaussian_AR'))
    def step(self,x,hidden=None):
        action,loc,h,_=super().step(x,hidden)
        z=np.zeros((1,2)) if not self.sample else self.rng.normal(size=(1,2))
        sd=torch.tensor(self.p['raw_sigma'],dtype=torch.float64)
        mean=loc.detach().clone()
        if self.sample and self.previous_raw is not None:
            mean=mean+self.rho*(self.previous_raw-self.previous_location)
            sd=sd*math.sqrt(1-self.rho**2)
        raw=mean+sd*torch.tensor(z,dtype=torch.float64) if self.sample else loc.detach()
        action=raw.clamp(-1,1).float()
        lp=(-.5*((raw-mean)/sd).square()-sd.log()-.5*math.log(2*math.pi)).sum(-1) if self.sample else torch.zeros(1,dtype=torch.float64)
        self.tape.append(dict(innovations=z[0],raw_actions=raw[0].numpy().copy(),locations=loc[0].detach().numpy().copy(),logp=lp[0].numpy().copy()))
        self.previous_raw=raw.detach();self.previous_location=loc.detach()
        return action,loc,h,loc


class NightLearner:
    def __init__(self,p,rho):
        self.actor=NightActor(p,rho);self.opt=torch.optim.Adam(self.actor.trainable(),lr=p['actor_lr']);self.calls=0
    def state(self):return dict(learned=self.actor.learned(),optimizer=copy.deepcopy(self.opt.state_dict()))
    def load(self,s):s=copy.deepcopy(s);self.actor.load_learned(s['learned']);self.opt.load_state_dict(s['optimizer'])
    def hashes(self):return {k:digest(v) for k,v in self.state().items()}


def night_worker(pipe,index,p,counts):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=Path(p['out']);P.C.C.OUT=out
    folder=out/'workers'/str(index);folder.mkdir(parents=True);log=N.LimitedLog(folder/'worker.log');sys.stdout=sys.stderr=log;total=0;w=None
    torch.optim.Adam.step=lambda *_a,**_k:(_ for _ in ()).throw(AssertionError('No native worker optimizer'))
    try:
        module,native,create,w,_=P.C.B.setup(folder/'native',p)
        original=copy.deepcopy(module.ACTIVE)
        q=NightActor(p,0.).requires_grad_(False)
        pipe.send(dict(ready=True,pid=os.getpid(),affinity=sorted(os.sched_getaffinity(0))))
        while True:
            j=pipe.recv()
            if j is None:break
            q.rho=j['rho'];q.load_learned(j['learned']);q.sample=j.get('stochastic',False);q.rng=np.random.default_rng(j.get('seed',0));q.reset_noise()
            if j.get('readback_model'):q.load_learned(read_model(j['readback_model']))
            assert q.phash()==j['policy_hash']
            module.ACTIVE=copy.deepcopy(original)
            module.ACTIVE['circle_shift_x_m']=j.get('circle_shift_x_m',original['circle_shift_x_m'])
            item=P.C.rollout(p,q,j,native,create,w,j['deadline_epoch'],counts,index,total)
            a=item['arrays'];assert len(q.tape)==len(a['states'])
            a.update({name:np.asarray([v[name] for v in q.tape]) for name in q.tape[0]})
            item['row'].update(rho=q.rho,stochastic=q.sample,seed=j.get('seed',0),
                circle_shift_x_m=j.get('circle_shift_x_m',.6) if j.get('wide_course') else 0.,array_hashes={k:digest(v) for k,v in a.items()})
            total+=item['row']['physics_steps'];pipe.send(item)
    except EOFError:pass
    except BaseException:
        failure=dict(error=traceback.format_exc(),actual_physics=counts[index]);save(folder/'failure.json',failure)
        try:pipe.send(failure)
        except (OSError,EOFError):pass
    finally:
        if w is not None:w.close()
        pipe.close();log.close()


def references(p,q):
    old,_=P.C.dataset(dict(p,sequence_sources=[]),q,lambda:None)
    rows=[json.loads(s) for s in Path(p['reference_cases']).read_text().splitlines()]
    induced=[];supports=[]
    for r in rows:
        if r['role']=='initial_old275' and r.get('old_reference_full'):
            a=arrays(r['path']);x=torch.tensor(a['states']);b=q.base_actions(x);induced.append(dict(x=x,b=b,y=b,failed=not r['success']))
        elif r['role']=='initial_wide5' and r.get('full_arrays_saved'):
            a=arrays(r['path']);assert r['success'];x=torch.tensor(a['states']);supports.append(dict(x=x,b=q.base_actions(x),y=torch.tensor(a['locations'],dtype=torch.float64)))
    induced.sort(key=lambda g:not g['failed']);assert len(induced)==20 and len(supports)==2
    return old,induced,supports


def reward_to_go(a):
    g=0.;r=np.empty(len(a['rewards']))
    for k in range(len(r)-1,-1,-1):g=float(a['rewards'][k])+K.R.GAMMA**int(a['native_counts'][k])*g;r[k]=g
    return r


def optimize(p,obj,items,refs,rnd,guard,event):
    assert len(items)==4
    g=[reward_to_go(i['arrays']) for i in items];start_g0=[float(v[0]) for v in g];table=np.zeros((4,max(map(len,g))))
    for j,v in enumerate(g):table[j,:len(v)]=v
    blocks=[]
    for j,item in enumerate(items):
        a=item['arrays'];x=torch.tensor(a['states']);b=obj.actor.base_actions(x);raw=torch.tensor(a['raw_actions']);before=torch.tensor(a['logp'])
        with torch.no_grad():
            loc=obj.actor.sequence(x,b);lp=raw_logp(loc,raw,obj.actor.rho,p['raw_sigma'])
            assert float((loc-torch.tensor(a['locations'])).abs().max())<1e-10
            assert float((lp-before).abs().max())<1e-9
        if p['credit']=='full_G0':c=np.full(len(x),g[j][0]-(sum(v[0] for v in g)-g[j][0])/3)
        else:
            discount=K.R.GAMMA**np.r_[0,np.cumsum(a['native_counts'].astype(np.int64))[:-1]]
            c=discount*(g[j]-(table.sum(0)-table[j])[:len(x)]/3)
        blocks.append((x,b,raw,before,torch.tensor(c/(4*400))))
    old,induced,supports=refs;trace=[]
    for k in range(p['epochs']):
        guard();obj.opt.zero_grad(set_to_none=True);pg=aux=0.;maxlr=0.
        for x,b,raw,oldlp,c in blocks:
            lr=raw_logp(obj.actor.sequence(x,b),raw,obj.actor.rho,p['raw_sigma'])-oldlp
            ratio=lr.exp();assert torch.isfinite(ratio).all()
            loss=-torch.minimum(ratio*c,ratio.clamp(.9,1.1)*c).sum();loss.backward();pg+=float(loss.detach());maxlr=max(maxlr,float(lr.detach().abs().max()))
        step=rnd*p['epochs']+k
        oldg=next(g for g in old if g['task']==step%8)
        for g,weight in [(oldg,1.),(induced[step%20],1.),(supports[step%2],1.)]:
            loss=weight*(obj.actor.sequence(g['x'],g['b'])-g['y']).square().mean();loss.backward();aux+=float(loss.detach())
        norm=float(torch.nn.utils.clip_grad_norm_(obj.actor.trainable(),1.,error_if_nonfinite=True))
        event(dict(phase='BEFORE',step=k));obj.opt.step();obj.calls+=1;event(dict(phase='AFTER',step=k))
        assert all(v.grad is None for v in obj.actor.base.parameters())
        trace.append(dict(step=k,pg=pg,auxiliary=aux,gradient_norm=norm,max_abs_logratio=maxlr))
    return dict(trace=trace,start_returns=start_g0)


def export(p,obj,path):
    buff=io.BytesIO();torch.save(dict(state=obj.state(),memory_on=True,rho=obj.actor.rho,hashes=obj.hashes()),buff)
    b=gzip.compress(buff.getvalue(),mtime=0);assert len(b)<=p['model_bytes'];path.write_bytes(b)
    return dict(path=str(path),sha256=K.C.sha(path),policy_hash=obj.actor.phash(),hashes=obj.hashes(),rho=obj.actor.rho)
