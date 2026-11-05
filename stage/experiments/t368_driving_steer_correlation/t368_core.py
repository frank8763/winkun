"""Steer-only temporal covariance with exact vector-AR latent policy density."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t367_factual_rehearsal_rl'))
from t367_core import *
import t367_core as BASE
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t368_driving_steer_correlation';DOC=ROOT/'stage/docs/narrow_20261105'
STEER_RHO=math.exp(-.2)


def raw_logp(loc,raw,rho,sigma):
    r=torch.tensor(rho,dtype=torch.float64);assert r.shape==(2,) and r[0]==0 and float(r[1])==STEER_RHO
    mean=torch.cat([loc[:1],loc[1:]+r*(raw[:-1]-loc[:-1])])
    sd=torch.tensor(sigma,dtype=torch.float64).expand_as(loc).clone();sd[1:]*=torch.sqrt(1-r*r)
    return (-.5*((raw-mean)/sd).square()-sd.log()-.5*math.log(2*math.pi)).sum(-1)


class NightActor(BASE.NightActor):
    def __init__(self,p,rho=0.):
        super().__init__(p,0.)
        if isinstance(rho,(list,tuple)):assert list(rho)==p['rho_vector']
        else:assert rho==0.
        assert p['rho_vector']==[0.,STEER_RHO];self.rho=tuple(p['rho_vector'])
    def phash(self):return digest(dict(actor=P.C.Actor.phash(self),rho_vector=self.rho,raw_sigma=self.p['raw_sigma'],density='full_latent_vector_AR_steering_only'))
    def step(self,x,hidden=None):
        _,loc,h,_=P.C.Actor.step(self,x,hidden)
        z=np.zeros((1,2)) if not self.sample else self.rng.normal(size=(1,2))
        sd=torch.tensor(self.p['raw_sigma'],dtype=torch.float64);r=torch.tensor(self.rho,dtype=torch.float64);mean=loc.detach().clone()
        if self.sample and self.previous_raw is not None:
            mean=mean+r*(self.previous_raw-self.previous_location);sd=sd*torch.sqrt(1-r*r)
        raw=mean+sd*torch.tensor(z,dtype=torch.float64) if self.sample else loc.detach();action=raw.clamp(-1,1).float()
        lp=(-.5*((raw-mean)/sd).square()-sd.log()-.5*math.log(2*math.pi)).sum(-1) if self.sample else torch.zeros(1,dtype=torch.float64)
        self.tape.append(dict(innovations=z[0],raw_actions=raw[0].numpy().copy(),locations=loc[0].detach().numpy().copy(),logp=lp[0].numpy().copy()))
        self.previous_raw=raw.detach();self.previous_location=loc.detach();return action,loc,h,loc


class NightLearner(BASE.NightLearner):
    def __init__(self,p,rho=0.):
        self.actor=NightActor(p,rho);self.opt=torch.optim.Adam(self.actor.trainable(),lr=p['actor_lr']);self.calls=0


def export(p,obj,path):
    buff=io.BytesIO();torch.save(dict(state=obj.state(),memory_on=True,rho_vector=list(obj.actor.rho),hashes=obj.hashes()),buff)
    b=gzip.compress(buff.getvalue(),mtime=0);assert len(b)<=p['model_bytes'];path.write_bytes(b)
    return dict(path=str(path),sha256=K.C.sha(path),policy_hash=obj.actor.phash(),hashes=obj.hashes(),rho_vector=list(obj.actor.rho))


def night_worker(pipe,index,p,counts):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=Path(p['out']);P.C.C.OUT=out
    folder=out/'workers'/str(index);folder.mkdir(parents=True);log=N.LimitedLog(folder/'worker.log');sys.stdout=sys.stderr=log;total=0;w=None
    torch.optim.Adam.step=lambda *_a,**_k:(_ for _ in ()).throw(AssertionError('No native worker optimizer'))
    try:
        module,native,create,w,_=P.C.B.setup(folder/'native',p)
        original=copy.deepcopy(module.ACTIVE)
        q=NightActor(p,p['rho_vector']).requires_grad_(False)
        pipe.send(dict(ready=True,pid=os.getpid(),affinity=sorted(os.sched_getaffinity(0))))
        while True:
            j=pipe.recv()
            if j is None:break
            assert list(j['rho'])==p['rho_vector'];q.rho=tuple(j['rho']);q.load_learned(j['learned']);q.sample=j.get('stochastic',False);q.rng=np.random.default_rng(j.get('seed',0));q.reset_noise()
            if j.get('readback_model'):q.load_learned(read_model(j['readback_model']))
            assert q.phash()==j['policy_hash']
            module.ACTIVE=copy.deepcopy(original)
            module.ACTIVE['circle_shift_x_m']=j.get('circle_shift_x_m',original['circle_shift_x_m'])
            item=P.C.rollout(p,q,j,native,create,w,j['deadline_epoch'],counts,index,total)
            a=item['arrays'];assert len(q.tape)==len(a['states'])
            a.update({name:np.asarray([v[name] for v in q.tape]) for name in q.tape[0]})
            item['row'].update(rho=list(q.rho),stochastic=q.sample,seed=j.get('seed',0),
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



def optimize(p,obj,items,refs,rnd,guard,event):
    assert len(items)==4
    returns=[float(reward_to_go(it['arrays'])[0]) for it in items];blocks=[]
    for j,it in enumerate(items):
        a=it['arrays'];x=torch.tensor(a['states']);b=obj.actor.base_actions(x);raw=torch.tensor(a['raw_actions']);before=torch.tensor(a['logp'])
        with torch.no_grad():
            loc=obj.actor.sequence(x,b);lp=raw_logp(loc,raw,p['rho_vector'],p['raw_sigma'])
            assert float((loc-torch.tensor(a['locations'])).abs().max())<1e-10
            assert float((lp-before).abs().max())<1e-9
        advantage=returns[j]-(sum(returns)-returns[j])/3
        blocks.append((x,b,raw,before,advantage/(4*400)))
    old,humans,narrow,wide=refs;trace=[]
    for k in range(p['epochs']):
        guard();obj.opt.zero_grad(set_to_none=True);pg=0.;maxlog=0.;losses=dict(old=0.,human=0.,wide=0.,narrow=0.)
        for x,b,raw,before,c in blocks:
            delta=raw_logp(obj.actor.sequence(x,b),raw,p['rho_vector'],p['raw_sigma'])-before;ratio=delta.exp();assert torch.isfinite(ratio).all()
            loss=-torch.minimum(ratio*c,ratio.clamp(.9,1.1)*c).sum();assert torch.isfinite(loss);loss.backward();pg+=float(loss.detach());maxlog=max(maxlog,float(delta.detach().abs().max()))
        task=(rnd*p['epochs']+k)%8;og=next(g for g in old if g['task']==task)
        choices=[('old',og,1.)]+[('wide',g,.2) for g in wide]+[('narrow',g,.5) for g in narrow]
        for name,g,weight in choices:
            loss=weight*(obj.actor.sequence(g['x'],g['b'])-g['y']).square().mean();assert torch.isfinite(loss);loss.backward();losses[name]+=float(loss.detach())
        for g in humans:
            loss=.5*labeled_loss(obj.actor.sequence(g['x'],g['b']),g);assert torch.isfinite(loss);loss.backward();losses['human']+=float(loss.detach())
        norm=float(torch.nn.utils.clip_grad_norm_(obj.actor.trainable(),1.,error_if_nonfinite=True));event(dict(phase='BEFORE',step=k))
        obj.opt.step();obj.calls+=1;event(dict(phase='AFTER',step=k));assert all(v.grad is None for v in obj.actor.base.parameters())
        trace.append(dict(step=k,pg=pg,supervised_components=losses,gradient_norm=norm,max_abs_logratio=maxlog,old_task=task))
    return dict(trace=trace,start_returns=returns)
