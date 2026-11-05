"""Full genuine human drive+steer transfer with a558parameter observation residual."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t375_observation_residual'))
from t375_core import *
import t375_core as PREV
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t377_full_human_observation_transfer';DOC=ROOT/'stage/docs/narrow_20261105'


class NightActor(PREV.NightActor):
    def __init__(self,p,rho=0.):
        super().__init__(p,rho)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(p['adapter_seed']);self.adapter=torch.nn.Sequential(torch.nn.Linear(136,4),torch.nn.ReLU(),torch.nn.Linear(4,2)).double()
            with torch.no_grad():self.adapter[0].bias.copy_(torch.tensor(p['adapter_initial_bias'],dtype=torch.float64))
            torch.nn.init.zeros_(self.adapter[2].weight);torch.nn.init.zeros_(self.adapter[2].bias)
    def location_with_observation(self,b,h,s):
        return 1.25*torch.tanh(torch.atanh(b/1.25)+self.head(h)+self.adapter(torch.cat([h,s],1)))


class NightLearner(PREV.NightLearner):
    def __init__(self,p,rho=0.):
        self.actor=NightActor(p,rho);self.actor.requires_grad_(False);self.actor.adapter.requires_grad_(True);self.opt=torch.optim.Adam(self.actor.adapter.parameters(),lr=p['actor_lr']);self.calls=0


def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes'] and K.S.used(OUT)<=p['final_bytes']
    assert shutil.disk_usage(ROOT).free>=p['min_free_bytes']


def dataset(p,q,guard):
    old,_=P.C.dataset(dict(p,sequence_sources=[]),q,guard);demos=[];factual=[]
    for ref in old:
        with torch.no_grad():ref['y']=q.sequence(ref['x'],ref['b']).detach().clone()
    for name,path,first,n in [('T361',p['demonstration'],0,295),('T364',p['correction'],20,297)]:
        a=arrays(path);assert len(a['states'])==n and a['end_flags'][-1,1] and not a['end_flags'][:,0].any() and not a['end_flags'][:-1].any()
        x=torch.tensor(a['states']);demos.append(dict(id=name,x=x,b=q.base_actions(x),y=torch.tensor(a['actions'],dtype=torch.float64),first=first));guard()
    for r in p['factual_parent_rows']:
        assert K.C.sha(r['path'])==r['sha256'];a=arrays(r['path']);assert {k:digest(v) for k,v in a.items()}==r['array_hashes'];x=torch.tensor(a['states']);b=q.base_actions(x)
        with torch.no_grad():y=q.sequence(x,b).detach().clone()
        assert float((y-torch.tensor(a['locations'])).abs().max())<1e-10 and np.array_equal(y.clamp(-1,1).float().numpy(),a['actions'])
        factual.append(dict(id=r['job_id'],x=x,b=b,y=y));guard()
    assert len(old)==211 and len(factual)==14 and sum(len(d['x'])-d['first'] for d in demos)==572
    return old,demos,factual


def fit(p,obj,data,guard,event,steps=None):
    old,demos,factual=data;seq=schedule(p,old);trace=[];fixed=fixed_hash(obj.actor)
    for k,oi in enumerate(seq[:p['fit_steps'] if steps is None else steps]):
        guard();obj.opt.zero_grad(set_to_none=True);losses=[]
        for d in demos:
            pred=obj.actor.sequence(d['x'],d['b']);first=d['first'];loss=.5*(pred[first:]-d['y'][first:]).square().mean();loss.backward();losses.append(float(loss.detach()))
        for ref,reduction in [(old[oi],'mean'),(factual[k%14],'max')]:
            error=(obj.actor.sequence(ref['x'],ref['b'])-ref['y']).square();loss=error.mean() if reduction=='mean' else error.max();loss.backward();losses.append(float(loss.detach()))
        assert all(np.isfinite(losses));norm=float(torch.nn.utils.clip_grad_norm_(obj.actor.adapter.parameters(),1.,error_if_nonfinite=True));event(dict(phase='BEFORE',step=k));obj.opt.step();obj.calls+=1;event(dict(phase='AFTER',step=k))
        assert fixed_hash(obj.actor)==fixed and all(v.grad is None for name,v in obj.actor.named_parameters() if not name.startswith('adapter.'))
        trace.append(dict(step=k,old_episode=oi,factual_history=k%14,human_prior_loss=losses[0],human_correction_loss=losses[1],old_loss=losses[2],factual_max_squared_error=losses[3],gradient_norm=norm))
    return trace


def metrics(q,data):
    out=dict(demonstrations=[],factual_routes=[],not_closed_loop_success=True)
    with torch.no_grad():
        for d in data[1]:
            pred=q.sequence(d['x'],d['b']);error=pred[d['first']:]-d['y'][d['first']:];out['demonstrations'].append(dict(id=d['id'],human_labels=len(error),masked_prefix=d['first'],drive_steer_RMSE=error.square().mean(0).sqrt().tolist()))
        for ref in data[2]:
            error=(q.sequence(ref['x'],ref['b']).clamp(-1,1).float()-ref['y'].clamp(-1,1).float()).abs();out['factual_routes'].append(dict(id=ref['id'],drive_steer_max=error.max(0).values.tolist(),executed_actions_exact=not bool(torch.count_nonzero(error))))
    return out


def contract(p,obj,data):
    q=obj.actor;old,demos,factual=data;parent=BASE.NightActor(p,0.).eval().requires_grad_(False);parent.load_learned(read_model(p['repair_parent']['path']));assert sum(v.numel() for v in q.trainable())==558 and fixed_hash(q)==p['repair_parent']['hashes']['learned']
    with torch.no_grad():
        for ref in [*demos,*old,*factual]:assert torch.equal(q.sequence(ref['x'],ref['b']),parent.sequence(ref['x'],ref['b']))
    d=demos[1];first=d['first'];pred=q.sequence(d['x'],d['b']);loss=(pred[first:]-d['y'][first:]).square().mean();grads=torch.autograd.grad(loss,q.trainable())
    y=d['y'].clone();y[:20]=12345.;other=(q.sequence(d['x'],d['b'])[20:]-y[20:]).square().mean();grads2=torch.autograd.grad(other,q.trainable())
    assert torch.equal(loss,other) and all(torch.equal(a,b) for a,b in zip(grads,grads2))
    assert all(not torch.count_nonzero(v) for v in grads[:2]) and all(torch.count_nonzero(grads[2][i]) for i in [0,1])
    x=d['x'].clone().requires_grad_(True);predx=q.sequence(x,d['b']);dx=torch.autograd.grad((predx[20:]-d['y'][20:]).square().mean(),x)[0];assert torch.count_nonzero(dx[:20])>0
    q.reset_noise();parent.reset_noise();h=ph=None;locations=[]
    with torch.no_grad():
        for x in d['x']:
            action,value,h,_=q.step(x[None],h);pa,pvalue,ph,_=parent.step(x[None],ph);assert torch.equal(action,pa) and torch.equal(value,pvalue) and torch.equal(h,ph)
            assert all(np.array_equal(v,parent.tape[-1][name]) for name,v in q.tape[-1].items());locations.append(value[0])
    error=float((torch.stack(locations)-pred.detach()).abs().max());assert error<1e-10
    return dict(state='PASS',parameters=558,full_human_labels=572,T364_prefix20_targets_masked_both_commands=True,prefix20_current_history_gradient_nonzero=True,drive_and_steer_output_gradients_nonzero=True,initial_input_layer_gradient_zero=True,zero_initial_parent_output_hidden_tape_exact=True,sequence_step_error=error,native=0,Adam=0)


def export(p,obj,path):
    b=io.BytesIO();torch.save(dict(state=obj.state(),memory_on=True,rho=0.,horizon=5,observation_residual=True,adapter_input_dimension=136,adapter_output_dimension=2,adapter_parameters=558,deterministic_execution_only=True,hashes=obj.hashes()),b);z=gzip.compress(b.getvalue(),mtime=0);assert len(z)<=p['model_bytes'];path.write_bytes(z)
    return dict(path=str(path),sha256=K.C.sha(path),hashes=obj.hashes(),policy_hash=obj.actor.phash(),architecture='T377_human_drive_steer_136x4x2')


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
