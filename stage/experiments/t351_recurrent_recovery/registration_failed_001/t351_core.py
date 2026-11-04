"""Current-history recurrent CAPG/PPO with fixed training-only reference losses."""
from pathlib import Path
import sys, math
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t349_consistent_demo'))
import t349_core as C
from t349_core import *
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t351_recurrent_recovery';DOC=ROOT/'stage/docs/review_20261104'
C.C.OUT=OUT
EDGE=(1.+float(np.nextafter(np.float32(1),np.float32(0))))/2


def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    assert K.S.used(OUT)<=p['final_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']


def logp(loc,raw,atom,p):
    sd=torch.tensor(p['raw_sigma'],dtype=torch.float64)
    normal=-.5*((raw-loc)/sd).square()-sd.log()-.5*math.log(2*math.pi)
    return torch.where(atom<0,torch.special.log_ndtr((-EDGE-loc)/sd),
        torch.where(atom>0,torch.special.log_ndtr((loc-EDGE)/sd),normal)).sum(-1)


class Actor(C.Actor):
    def __init__(self,p):
        super().__init__(p,'memory');self.p=p;self.sample=False;self.rng=None;self.tape=[]
        v=torch.load(io.BytesIO(gzip.decompress(Path(p['initial_model']).read_bytes())),weights_only=False,map_location='cpu')
        assert v['memory_on'];self.load_learned(v['state']['learned'])
    def phash(self):return digest(dict(distribution='iid_clippedGaussian_float32_edge',sigma=self.p['raw_sigma'],actor=super().phash()))
    def step(self,x,hidden=None):
        mode,loc,new,_=super().step(x,hidden)
        z=np.zeros((1,2)) if not self.sample else self.rng.normal(size=(1,2))
        raw=loc.detach()+torch.tensor(z,dtype=torch.float64)*torch.tensor(self.p['raw_sigma'],dtype=torch.float64)
        atom=torch.where(raw>=EDGE,1,torch.where(raw<=-EDGE,-1,0)).to(torch.int8)
        executed=raw.clamp(-1,1).float();assert torch.equal(atom.abs()==1,executed.abs()==1)
        lp=logp(loc,raw,atom,self.p) if self.sample else torch.zeros(1,dtype=torch.float64)
        self.tape.append(dict(innovations=z[0].copy(),raw_actions=raw[0].numpy().copy(),atom=atom[0].numpy().copy(),
                             locations=loc[0].detach().numpy().copy(),logp=lp[0].detach().numpy().copy()))
        return executed,loc,new,loc


class Learner(C.Learner):
    def __init__(self,p):
        self.actor=Actor(p);self.opt=torch.optim.Adam(self.actor.trainable(),lr=p['actor_lr']);self.calls=0


def worker(pipe,index,p,counts,preflight=False):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    q=Actor(p).eval().requires_grad_(False)
    torch.optim.Adam.step=lambda *_a,**_kw:(_ for _ in ()).throw(AssertionError('No worker optimizer'))
    if preflight:
        cmd=pipe.recv();q.load_learned(cmd['learned']);q.sample=True;q.rng=np.random.default_rng(cmd['seed']);q.tape=[];h=None;actions=[]
        with torch.no_grad():
            for x in cmd['states']:
                a,_,h,_=q.step(torch.tensor(x[None]),h);actions.append(a[0].numpy())
        pipe.send(dict(actions=np.asarray(actions),tape=q.tape,policy_hash=q.phash()));pipe.close();return
    folder=OUT/'workers'/str(index);folder.mkdir(parents=True);log=N.LimitedLog(folder/'worker.log');sys.stdout=sys.stderr=log;total=0;w=None
    try:
        _,native,create,w,_=C.B.setup(folder/'native',p)
        pipe.send(dict(ready=True,pid=os.getpid(),affinity=sorted(os.sched_getaffinity(0))))
        while True:
            j=pipe.recv()
            if j is None:break
            q.load_learned(j['learned']);q.sample=j.get('stochastic',False);q.rng=np.random.default_rng(j.get('seed',0));q.tape=[]
            if j.get('readback_model'):
                v=torch.load(io.BytesIO(gzip.decompress(Path(j['readback_model']).read_bytes())),weights_only=False,map_location='cpu');q.load_learned(v['state']['learned'])
            assert q.phash()==j['policy_hash']
            item=C.rollout(p,q,j,native,create,w,j['deadline_epoch'],counts,index,total)
            a=item['arrays'];assert len(a['states'])==len(q.tape)
            a.update({name:np.asarray([v[name] for v in q.tape]) for name in q.tape[0]})
            item['row'].update(stochastic=q.sample,seed=j.get('seed',0),array_hashes={name:digest(v) for name,v in a.items()})
            total+=item['row']['physics_steps'];pipe.send(item)
    except EOFError:pass
    except BaseException:
        f=dict(error=traceback.format_exc(),actual_physics=counts[index]);save(folder/'failure.json',f)
        try:pipe.send(f)
        except (OSError,EOFError):pass
    finally:
        if w is not None:w.close()
        pipe.close();log.close()


def reference_data(p,q,old_items,supports,guard):
    old,_=C.dataset(dict(p,sequence_sources=[]),q,guard)
    induced=[]
    for item in sorted(old_items,key=lambda i:i['row']['success']):
        x=torch.tensor(item['arrays']['states']);b=q.base_actions(x)
        induced.append(dict(x=x,b=b,y=b.clone()))
    good=[]
    for item in supports:
        assert item['row']['success'] and not item['row']['collision']
        x=torch.tensor(item['arrays']['states']);good.append(dict(x=x,b=q.base_actions(x),y=torch.tensor(item['arrays']['actions'],dtype=torch.float64)))
    assert len(old)==211 and len(induced)==20 and len(good)==2
    return old,induced,good


def total_return(a):
    assert a['end_flags'][-1].any() and not a['end_flags'][:-1].any()
    g=0.
    for r,n in zip(a['rewards'][::-1],a['native_counts'][::-1]):g=float(r)+K.R.GAMMA**int(n)*g
    return g


def update(p,obj,items,refs,rnd,guard,event):
    assert len(items)==4;returns=np.asarray([total_return(i['arrays']) for i in items]);blocks=[]
    for i,item in enumerate(items):
        a=item['arrays'];assert item['row']['stochastic'];x=torch.tensor(a['states']);b=obj.actor.base_actions(x)
        raw=torch.tensor(a['raw_actions']);atom=torch.tensor(a['atom']);old=torch.tensor(a['logp'])
        with torch.no_grad():
            loc=obj.actor.sequence(x,b);error=float((loc-torch.tensor(a['locations'])).abs().max())
            density_error=float((logp(loc,raw,atom,p)-old).abs().max())
            assert error<1e-10 and density_error<1e-9,(error,density_error)
        advantage=float(returns[i]-(returns.sum()-returns[i])/3)
        blocks.append((x,b,raw,atom,old,advantage/(4*400)))
    trace=[];old,induced,good=refs
    for k in range(p['epochs']):
        guard();obj.opt.zero_grad(set_to_none=True);pg=0.;aux=0.;maxlog=0.
        for x,b,raw,atom,before,c in blocks:
            ratio_log=logp(obj.actor.sequence(x,b),raw,atom,p)-before
            ratio=ratio_log.exp();assert torch.isfinite(ratio).all()
            loss=-torch.minimum(ratio*c,ratio.clamp(1-p['ppo_clip'],1+p['ppo_clip'])*c).sum()
            loss.backward();pg+=float(loss.detach());maxlog=max(maxlog,float(ratio_log.detach().abs().max()))
        step=rnd*p['epochs']+k
        # Training-only old reference at real student-visited states; no claim
        # that its action certifies recovery, and never used at deployment.
        old_group=next(g for g in old if g['task']==step%8)
        choices=[(old_group,p['old_weight']),
                 (induced[step%20],p['induced_weight']),
                 (good[step%2],p['support_weight'])]
        for g,weight in choices:
            loss=weight*(obj.actor.sequence(g['x'],g['b'])-g['y']).square().mean()
            loss.backward();aux+=float(loss.detach())
        norm=float(torch.nn.utils.clip_grad_norm_(obj.actor.trainable(),1.,error_if_nonfinite=True))
        event(dict(phase='BEFORE',step=k));obj.opt.step();obj.calls+=1;event(dict(phase='AFTER',step=k))
        assert all(v.grad is None for v in obj.actor.base.parameters())
        trace.append(dict(step=k,pg=pg,auxiliary=aux,gradient_norm=norm,max_abs_logratio=maxlog))
    return dict(trace=trace,start_returns=returns.tolist())


def export(p,obj,path):
    b=io.BytesIO();torch.save(dict(state=obj.state(),memory_on=True,hashes=obj.hashes()),b)
    blob=gzip.compress(b.getvalue(),mtime=0);assert len(blob)<=p['model_bytes'];path.write_bytes(blob)
    return dict(path=str(path),sha256=K.C.sha(path),hashes=obj.hashes(),policy_hash=obj.actor.phash())
