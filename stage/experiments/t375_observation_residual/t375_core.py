"""Single-Actor current-observation/history steering residual,553parameters."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t374_factual_worst_state_repair'))
from t374_core import *
import t374_core as LAST
BASE=LAST.PREVIOUS.PRIOR.OLD
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t375_observation_residual';DOC=ROOT/'stage/docs/narrow_20261105'


class NightActor(BASE.NightActor):
    def __init__(self,p,rho=0.):
        super().__init__(p,rho);self.load_learned(read_model(p['repair_parent']['path']))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(p['adapter_seed']);self.adapter=torch.nn.Sequential(torch.nn.Linear(136,4),torch.nn.ReLU(),torch.nn.Linear(4,1)).double()
            with torch.no_grad():self.adapter[0].bias.copy_(torch.tensor(p['adapter_initial_bias'],dtype=torch.float64))
            torch.nn.init.zeros_(self.adapter[2].weight);torch.nn.init.zeros_(self.adapter[2].bias)
    def trainable(self):return list(self.adapter.parameters())
    def normalized(self,x):return ((x.double()-self.mean)/self.scale).clamp(-10,10)
    def location(self,b,h):raise AssertionError('Explicit current observation required for T375location')
    def location_with_observation(self,b,h,s):
        original=self.head(h);delta=self.adapter(torch.cat([h,s],1));logits=torch.cat([original[:,:1],original[:,1:]+delta],1)
        return 1.25*torch.tanh(torch.atanh(b/1.25)+logits)
    def sequence(self,x,b):
        s=self.normalized(x);z=self.encoder(s);h=self.core(z[:,None])[0][:,0]
        return self.location_with_observation(b,h,s)
    def step(self,x,hidden=None):
        assert x.shape==(1,120) and not self.sample and self.memory_on
        if hidden is None:hidden=x.new_zeros((1,1,16),dtype=torch.float64)
        s=self.normalized(x);z=self.encoder(s);h,new=self.core(z[None],hidden);b=self.base_actions(x);loc=self.location_with_observation(b,h[0],s);raw=loc.detach();action=raw.clamp(-1,1).float()
        future=self.future_location(b,new[0]);self.tape.append(dict(innovations=np.zeros(2),raw_actions=raw[0].numpy().copy(),locations=loc[0].detach().numpy().copy(),logp=np.asarray(0.,dtype=np.float64),future_locations=future[0].detach().numpy().copy()))
        self.previous_raw=raw;self.previous_location=loc.detach()
        return action,loc,new,loc


class NightLearner(BASE.PRIOR.NightLearner):
    def __init__(self,p,rho=0.):
        self.actor=NightActor(p,rho);self.actor.requires_grad_(False);self.actor.adapter.requires_grad_(True)
        self.opt=torch.optim.Adam(self.actor.adapter.parameters(),lr=p['actor_lr']);self.calls=0


def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes'] and K.S.used(OUT)<=p['final_bytes']
    assert shutil.disk_usage(ROOT).free>=p['min_free_bytes']


dataset=LAST.dataset
fit=LAST.fit
locality=LAST.locality
gate=LAST.gate


def contract(p,obj,data):
    q=obj.actor;old,g,factual=data;parent=BASE.NightActor(p,0.).eval().requires_grad_(False);parent.load_learned(read_model(p['repair_parent']['path']))
    assert sum(v.numel() for v in q.trainable())==553 and fixed_hash(q)==p['repair_parent']['hashes']['learned']
    with torch.no_grad():
        for ref in [g,*old,*factual]:assert torch.equal(q.sequence(ref['x'],ref['b']),parent.sequence(ref['x'],ref['b']))
    pred=q.sequence(g['x'],g['b']);mask=g['mask'];loss=(pred[mask,1]-g['target'][mask]).square().mean();grad=torch.autograd.grad(loss,list(q.adapter.parameters()))
    assert all(not torch.count_nonzero(v) for v in grad[:2]) and all(torch.count_nonzero(v)>0 for v in grad[2:])
    q.reset_noise();parent.reset_noise();q.sample=False;parent.sample=False;h=ph=None;locations=[]
    with torch.no_grad():
        for x in g['x']:
            action,value,h,_=q.step(x[None],h);pa,pvalue,ph,_=parent.step(x[None],ph)
            assert torch.equal(action,pa) and torch.equal(value,pvalue) and torch.equal(h,ph)
            assert q.tape[-1].keys()==parent.tape[-1].keys() and all(np.array_equal(v,parent.tape[-1][name]) for name,v in q.tape[-1].items());locations.append(value[0])
    step_error=float((torch.stack(locations)-pred.detach()).abs().max());assert step_error<1e-10
    m=locality(q,data);assert m['prefix248_executed_steer_max']==m['first20_executed_steer_max']==m['factual_executed_steer_max']==0. and all(v['executed_actions_exact'] for v in m['factual_routes'])
    assert all(not v.requires_grad for name,v in q.named_parameters() if not name.startswith('adapter.'))
    assert sum(len(ref['x']) for ref in factual)==2639
    return dict(state='PASS',trainable_parameters=553,raw_observation_columns=120,history_features=16,zero_initial_parent_all_histories_actions_hidden_tape_exact=True,sequence_step_max_error=step_error,initial_feature_gradient_zero_as_expected=True,initial_output_gradient_nonzero=True,frozen_parent_SHA_exact=True,factual_parent_routes_exact=14,factual_decisions=2639,initial_patch_loss=float(loss.detach()),new_native=0,new_Adam=0)


def current_path_contract(obj,data):
    g=data[1];q=obj.actor
    with torch.no_grad():h=q.core(q.encoder(q.normalized(g['x']))[:,None])[0][:,0]
    x=g['x'].clone().requires_grad_(True);out=q.adapter(torch.cat([h.detach(),q.normalized(x)],1));k=248;dx=torch.autograd.grad(out[k].sum(),x)[0]
    assert torch.count_nonzero(dx[k])>0 and not torch.count_nonzero(dx[:k]) and not torch.count_nonzero(dx[k+1:])
    assert torch.count_nonzero(q.adapter[0].weight.grad[:,16:])>0
    return dict(state='PASS',current_observation_path_gradient_nonzero=True,no_future_or_past_observation_dependence_in_direct_branch=True,frozen_history_features_conditioned_separately=True,optimizer=0,native=0)


def export(p,obj,path):
    buf=io.BytesIO();torch.save(dict(state=obj.state(),memory_on=True,rho=0.,horizon=5,executes_first_only=True,observation_residual=True,adapter_input_dimension=136,adapter_trainable_parameters=553,deterministic_execution_only=True,hashes=obj.hashes()),buf)
    blob=gzip.compress(buf.getvalue(),mtime=0);assert len(blob)<=p['model_bytes'];path.write_bytes(blob)
    return dict(path=str(path),sha256=K.C.sha(path),hashes=obj.hashes(),policy_hash=obj.actor.phash(),horizon=5,architecture='T375_current120_plus_history16_steering_logit_residual')


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
