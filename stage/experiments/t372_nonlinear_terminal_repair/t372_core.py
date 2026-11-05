"""Single-Actor73parameter nonlinear steering-logit repair; frozen parent."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t371_terminal_head_repair'))
from t371_core import *
import t371_core as PRIOR
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t372_nonlinear_terminal_repair';DOC=ROOT/'stage/docs/narrow_20261105'


class NightActor(PRIOR.OLD.NightActor):
    def __init__(self,p,rho=0.):
        super().__init__(p,rho);self.load_learned(read_model(p['repair_parent']['path']))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(p['adapter_seed']);self.adapter=torch.nn.Sequential(torch.nn.Linear(16,4),torch.nn.ReLU(),torch.nn.Linear(4,1)).double()
            with torch.no_grad():self.adapter[0].bias.copy_(torch.tensor(p['adapter_initial_bias'],dtype=torch.float64))
            torch.nn.init.zeros_(self.adapter[2].weight);torch.nn.init.zeros_(self.adapter[2].bias)
    def trainable(self):return list(self.adapter.parameters())
    def location(self,b,h):
        original=self.head(h);delta=self.adapter(h)
        logits=torch.cat([original[:,:1],original[:,1:]+delta],1)
        return 1.25*torch.tanh(torch.atanh(b/1.25)+logits)


class NightLearner(PRIOR.OLD.NightLearner):
    def __init__(self,p,rho=0.):
        self.actor=NightActor(p,rho);self.actor.requires_grad_(False);self.actor.adapter.requires_grad_(True)
        self.opt=torch.optim.Adam(self.actor.adapter.parameters(),lr=p['actor_lr']);self.calls=0


def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes'] and K.S.used(OUT)<=p['final_bytes']
    assert shutil.disk_usage(ROOT).free>=p['min_free_bytes']


def fixed_hash(q):return digest({k:v for k,v in q.learned().items() if not k.startswith('adapter.')})


dataset=PRIOR.dataset
descriptive_error=PRIOR.descriptive_error


def fit(p,obj,data,guard,event,steps=None):
    old,g,factual=data;seq=schedule(p,old);trace=[];fixed=fixed_hash(obj.actor)
    for k,oi in enumerate(seq[:p['fit_steps'] if steps is None else steps]):
        guard();obj.opt.zero_grad(set_to_none=True);pred=obj.actor.sequence(g['x'],g['b'])[:,1]
        patch=(pred[g['mask']]-g['target'][g['mask']]).square().mean();hold=(pred[~g['mask']]-g['y'][~g['mask'],1]).square().mean()
        patch.backward(retain_graph=True);hold.backward();losses=[float(patch.detach()),float(hold.detach())]
        for ref in [old[oi],factual[k%7]]:
            loss=(obj.actor.sequence(ref['x'],ref['b'])[:,1]-ref['y'][:,1]).square().mean();loss.backward();losses.append(float(loss.detach()))
        assert all(np.isfinite(losses))
        feature_grad=any(bool(torch.count_nonzero(v.grad)) for v in obj.actor.adapter[0].parameters())
        norm=float(torch.nn.utils.clip_grad_norm_(obj.actor.adapter.parameters(),1.,error_if_nonfinite=True));event(dict(phase='BEFORE',step=k));obj.opt.step();obj.calls+=1;event(dict(phase='AFTER',step=k))
        assert fixed_hash(obj.actor)==fixed and all(v.grad is None for name,v in obj.actor.named_parameters() if not name.startswith('adapter.'))
        trace.append(dict(step=k,old_episode=oi,factual_history=k%7,patch_loss=losses[0],route_hold_loss=losses[1],old_hold_loss=losses[2],factual_hold_loss=losses[3],gradient_norm=norm,adapter_feature_gradient_nonzero=feature_grad))
    return trace


def repair_contract(obj,data):
    q=obj.actor;old,g,factual=data;parent=PRIOR.OLD.NightActor(q.p,0.);parent.load_learned(read_model(q.p['repair_parent']['path']));parent.requires_grad_(False)
    assert sum(v.numel() for v in q.trainable())==73 and fixed_hash(q)==q.p['repair_parent']['hashes']['learned']
    maxerr=0.;preserve=0.
    with torch.no_grad():
        for ref in [g,*old,*factual]:
            pred=q.sequence(ref['x'],ref['b']);expected=parent.sequence(ref['x'],ref['b']);err=float((pred-expected).abs().max());maxerr=max(maxerr,err);assert torch.equal(pred,expected)
            preserve=max(preserve,float((pred-ref['y']).abs().max()))
    assert maxerr==0. and preserve<1e-10
    pred=q.sequence(g['x'],g['b']);mask=g['mask'];patch=(pred[mask,1]-g['target'][mask]).square().mean()
    grads=torch.autograd.grad(patch,list(q.adapter.parameters()))
    assert all(not torch.count_nonzero(v) for v in grads[:2]) and all(torch.count_nonzero(v)>0 for v in grads[2:]) and float(patch.detach())>0.
    assert all(not v.requires_grad for name,v in q.named_parameters() if not name.startswith('adapter.'))
    q.reset_noise();h=None;loc=[]
    with torch.no_grad():
        for x in g['x']:
            _,value,h,_=q.step(x[None],h);loc.append(value[0])
        step_error=float((torch.stack(loc)-pred.detach()).abs().max())
    assert step_error<1e-10
    return dict(state='PASS',trainable_parameters=73,zero_output_initial_parent_all_fixed_histories_exact=True,frozen_parent_SHA_exact=True,initial_feature_gradient_zero_as_expected=True,initial_output_gradient_nonzero=True,sequence_step_max_error=step_error,initial_patch_loss=float(patch.detach()),initial_preservation_max_error=preserve,explicit_nonhuman_intervention_labels=5,native=0,Adam=0)


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
