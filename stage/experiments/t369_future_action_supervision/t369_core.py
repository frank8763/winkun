"""Causal multi-horizon supervision; deployment uses the unchanged current head."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t365_correction_aggregation'))
from t365_core import *
import t365_core as PRIOR
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t369_future_action_supervision';DOC=ROOT/'stage/docs/narrow_20261105'


class NightActor(PRIOR.NightActor):
    def __init__(self,p,rho=0.):
        super().__init__(p,rho)
        self.future_head=torch.nn.Linear(16,8).double()
        with torch.no_grad():
            self.future_head.weight.copy_(self.head.weight.repeat(4,1));self.future_head.bias.copy_(self.head.bias.repeat(4))
    def trainable(self):return [*super().trainable(),*self.future_head.parameters()]
    def future_location(self,b,h):
        v=self.future_head(h).reshape(-1,4,2)
        return 1.25*torch.tanh(torch.atanh(b[:,None]/1.25)+v)
    def chunks(self,x,b):
        z=self.encoder(((x.double()-self.mean)/self.scale).clamp(-10,10));h,_=self.core(z[:,None]);h=h[:,0]
        current=self.location(b,h);future=self.future_location(b,h)
        return torch.cat([current[:,None],future],1)
    def step(self,x,hidden=None):
        action,loc,h,other=super().step(x,hidden)
        future=self.future_location(self.base_actions(x),h[0])
        self.tape[-1]['future_locations']=future[0].detach().numpy().copy()
        return action,loc,h,other


class NightLearner(PRIOR.NightLearner):
    def __init__(self,p,rho=0.):
        self.actor=NightActor(p,rho);self.opt=torch.optim.Adam(self.actor.trainable(),lr=p['actor_lr']);self.calls=0


def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    assert K.S.used(OUT)<=p['final_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']


def dataset(p,q,guard):
    old,_=P.C.dataset(dict(p,sequence_sources=[]),q,guard);demos=[];wide=[];narrow=[]
    for name,path,first,n in [('T361',p['demonstration'],0,295),('T364',p['correction'],20,297)]:
        a=arrays(path);assert len(a['states'])==n and a['end_flags'][-1,1] and not a['end_flags'][:-1].any() and not a['end_flags'][:,0].any()
        x=torch.tensor(a['states']);demos.append(dict(id=name,x=x,b=q.base_actions(x),y=torch.tensor(a['actions'],dtype=torch.float64),first_label=first));guard()
    for r in p['factual_rehearsal']:
        assert K.C.sha(r['path'])==r['sha256'];a=arrays(r['path']);x=torch.tensor(a['states']);g=dict(id=r['job_id'],x=x,b=q.base_actions(x),y=torch.tensor(a['locations']),first_label=0)
        (wide if r['job_id'].startswith('wide') else narrow).append(g);guard()
    for g in old:g['first_label']=0
    assert len(old)==211 and len(wide)==5 and len(narrow)==2
    return old,demos,wide,narrow


def targets(g):
    n=len(g['y']);idx=torch.arange(n)[:,None]+torch.arange(5)[None,:]
    mask=(idx<n)&(idx>=g['first_label']);return g['y'][idx.clamp(max=n-1)],mask


def sequence_loss(pred,g,future_weight):
    y,mask=targets(g);current=(pred[:,0]-y[:,0])[mask[:,0]].square().mean()
    future=(pred[:,1:]-y[:,1:])[mask[:,1:]].square().mean()
    return current+future_weight*future,current,future


def fit(p,obj,data,arm,guard,event,steps=None):
    old,demos,wide,narrow=data;seq=schedule(p,old);trace=[];fw=p['arms'][arm]
    for k,oi in enumerate(seq[:p['fit_steps'] if steps is None else steps]):
        guard();obj.opt.zero_grad(set_to_none=True);parts=[]
        choices=[('human_prior',demos[0],.5),('human_correction',demos[1],.5),('old',old[oi],1.),('wide',wide[k%5],1.),('narrow',narrow[k%2],1.)]
        for name,g,weight in choices:
            loss,cur,fut=sequence_loss(obj.actor.chunks(g['x'],g['b']),g,fw);loss=weight*loss;assert torch.isfinite(loss)
            loss.backward();parts.append(dict(name=name,current=float(cur.detach()),future=float(fut.detach()),weighted=float(loss.detach())))
        if fw==0.:
            # Multiplying by zero may leave an exact-zero gradient, never an update.
            assert all(v.grad is None or not torch.count_nonzero(v.grad) for v in obj.actor.future_head.parameters())
        norm=float(torch.nn.utils.clip_grad_norm_(obj.actor.trainable(),1.,error_if_nonfinite=True));event(dict(phase='BEFORE',step=k))
        obj.opt.step();obj.calls+=1;event(dict(phase='AFTER',step=k));assert all(v.grad is None for v in obj.actor.base.parameters())
        trace.append(dict(step=k,old_episode=oi,wide=k%5,narrow=k%2,future_weight=fw,components=parts,gradient_norm=norm))
    return trace


def contracts(p,obj,data):
    q=obj.actor;g=data[1][1];pred=q.chunks(g['x'],g['b']);loss,_,_=sequence_loss(pred,g,.5)
    grad=torch.autograd.grad(loss,q.trainable(),allow_unused=True)
    alt=dict(g,y=g['y'].clone());alt['y'][:20]=12345.;other,_,_=sequence_loss(q.chunks(g['x'],g['b']),alt,.5)
    grad2=torch.autograd.grad(other,q.trainable(),allow_unused=True)
    assert torch.equal(loss,other) and all((a is None and b is None) or (a is not None and b is not None and torch.equal(a,b)) for a,b in zip(grad,grad2))
    assert all(v is not None and torch.count_nonzero(v)>0 for v in grad[-2:])
    counts=[]
    for d in data[1]:counts.append(targets(d)[1].sum(0).tolist())
    assert counts==[[295,294,293,292,291],[277]*5]
    x=g['x'].clone().requires_grad_(True);out=q.chunks(x,g['b']);t=20;dx=torch.autograd.grad(out[t].sum(),x)[0]
    assert not torch.count_nonzero(dx[t+1:]) and torch.count_nonzero(dx[:t])>0
    changed=g['x'].clone();changed[t+1:]=changed[t+1:]+.123
    with torch.no_grad():
        out2=q.chunks(changed,q.base_actions(changed));assert torch.equal(out[:t+1].detach(),out2[:t+1])
        base=PRIOR.NightActor(p,0.).requires_grad_(False);h=hh=None;actions=[];future=[];q.reset_noise()
        for row in g['x']:
            a,_,h,_=q.step(row[None],h);b,_,hh,_=base.step(row[None],hh);assert torch.equal(a,b);actions.append(a[0]);future.append(q.tape[-1]['future_locations'])
        seq=q.sequence(g['x'],g['b']);chunk=q.chunks(g['x'],g['b']);assert torch.equal(seq,chunk[:,0])
        err=float((torch.stack(actions).double()-chunk[:,0].clamp(-1,1).float().double()).abs().max());assert err==0.
        assert np.max(np.abs(np.asarray(future)-chunk[:,1:].numpy()))<1e-10
    assert sum(v.numel() for v in q.trainable())==6442
    return dict(state='PASS',parameters=6442,shared6306_plus_future136=True,first_head_initial_actions_exact_parent=True,
        prefix20targets_never_labels_in_any_head=True,human_valid_counts_by_horizon=counts,causal_no_future_observation_gradient=True,
        prefix_history_gradient_nonzero=True,current_sequence_chunk_equal=True,sequence_step_executed_max_error=err,future_head_gradient_nonzero=True,native=0,Adam=0)


def descriptive_error(q,data):
    out={}
    with torch.no_grad():
        for g in data[1]:
            pred=q.chunks(g['x'],g['b']);y,m=targets(g);out[g['id']]=dict(current_rmse=(pred[:,0]-y[:,0])[m[:,0]].square().mean(0).sqrt().tolist(),future_rmse=(pred[:,1:]-y[:,1:])[m[:,1:]].square().mean(0).sqrt().tolist())
    return out


def export(p,obj,path):
    b=io.BytesIO();torch.save(dict(state=obj.state(),memory_on=True,rho=0.,horizon=5,executes_first_only=True,hashes=obj.hashes()),b)
    z=gzip.compress(b.getvalue(),mtime=0);assert len(z)<=p['model_bytes'];path.write_bytes(z)
    return dict(path=str(path),sha256=K.C.sha(path),hashes=obj.hashes(),policy_hash=obj.actor.phash(),horizon=5)


def night_worker(pipe,index,p,counts):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.optim.Adam.step=lambda *_a,**_k:(_ for _ in ()).throw(AssertionError('No worker optimizer'))
    out=Path(p['out']);P.C.C.OUT=out;folder=out/'workers'/str(index);folder.mkdir(parents=True);log=N.LimitedLog(folder/'worker.log');sys.stdout=sys.stderr=log;total=0;w=None
    try:
        module,native,create,w,_=P.C.B.setup(folder/'native',p);original=copy.deepcopy(module.ACTIVE);q=NightActor(p,0.).requires_grad_(False)
        pipe.send(dict(ready=True,pid=os.getpid(),affinity=sorted(os.sched_getaffinity(0))))
        while True:
            j=pipe.recv()
            if j is None:break
            q.load_learned(j['learned']);q.sample=False;q.reset_noise()
            if j.get('readback_model'):q.load_learned(read_model(j['readback_model']))
            assert q.phash()==j['policy_hash'];module.ACTIVE=copy.deepcopy(original);module.ACTIVE['circle_shift_x_m']=j.get('circle_shift_x_m',original['circle_shift_x_m'])
            it=P.C.rollout(p,q,j,native,create,w,j['deadline_epoch'],counts,index,total);a=it['arrays'];assert len(q.tape)==len(a['states'])
            a.update({name:np.asarray([v[name] for v in q.tape]) for name in q.tape[0]});it['row'].update(stochastic=False,only_current_head_executed=True,array_hashes={k:digest(v) for k,v in a.items()})
            total+=it['row']['physics_steps'];pipe.send(it)
    except EOFError:pass
    except BaseException:
        failure=dict(error=traceback.format_exc(),actual_physics=counts[index]);save(folder/'failure.json',failure)
        try:pipe.send(failure)
        except (EOFError,OSError):pass
    finally:
        if w is not None:w.close()
        pipe.close();log.close()
