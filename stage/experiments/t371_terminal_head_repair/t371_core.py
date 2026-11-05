"""One 17-effective-parameter steering repair from explicit non-human intervention data."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t369_future_action_supervision'))
from t369_core import *
import t369_core as OLD
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t371_terminal_head_repair';DOC=ROOT/'stage/docs/narrow_20261105'


class NightActor(OLD.NightActor):
    def __init__(self,p,rho=0.):
        super().__init__(p,rho);self.load_learned(read_model(p['repair_parent']['path']))


class NightLearner(OLD.NightLearner):
    def __init__(self,p,rho=0.):
        self.actor=NightActor(p,rho);self.actor.requires_grad_(False);self.actor.head.requires_grad_(True)
        self.actor.head.weight.register_hook(lambda g:g*torch.tensor([[0.],[1.]],dtype=g.dtype))
        self.actor.head.bias.register_hook(lambda g:g*torch.tensor([0.,1.],dtype=g.dtype))
        self.opt=torch.optim.Adam(self.actor.head.parameters(),lr=p['actor_lr']);self.calls=0


def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes'] and K.S.used(OUT)<=p['final_bytes']
    assert shutil.disk_usage(ROOT).free>=p['min_free_bytes']


def fixed_hash(q):
    a=q.learned();a['head.weight']=a['head.weight'][:1].clone();a['head.bias']=a['head.bias'][:1].clone();return digest(a)


def dataset(p,q,guard):
    old,_=P.C.dataset(dict(p,sequence_sources=[]),q,guard);factual=[]
    for g in old:
        with torch.no_grad():g['y']=q.sequence(g['x'],g['b']).detach().clone()
    for row in p['factual_rehearsal']:
        a=arrays(row['path']);x=torch.tensor(a['states']);b=q.base_actions(x)
        with torch.no_grad():y=q.sequence(x,b).detach().clone()
        factual.append(dict(id=row['job_id'],x=x,b=b,y=y));guard()
    r=p['repair_source'];assert K.C.sha(r['path'])==r['sha256'];a=arrays(r['path']);x=torch.tensor(a['states']);b=q.base_actions(x)
    with torch.no_grad():y=q.sequence(x,b).detach().clone()
    assert float((y-torch.tensor(a['locations'])).abs().max())<1e-10
    mask=torch.tensor(a['override_mask'].astype(bool));assert mask.sum()==5 and np.array_equal(np.flatnonzero(mask.numpy()),np.arange(248,253)) and np.all(a['actions'][mask.numpy(),1]==0.)
    g=dict(id='T370_neutral_success',x=x,b=b,y=y,mask=mask,target=torch.tensor(a['actions'][:,1],dtype=torch.float64))
    assert len(old)==211 and len(factual)==7
    return old,g,factual


def fit(p,obj,data,guard,event,steps=None):
    old,g,factual=data;seq=schedule(p,old);trace=[];fixed=fixed_hash(obj.actor)
    for k,oi in enumerate(seq[:p['fit_steps'] if steps is None else steps]):
        guard();obj.opt.zero_grad(set_to_none=True);pred=obj.actor.sequence(g['x'],g['b'])[:,1]
        patch=(pred[g['mask']]-g['target'][g['mask']]).square().mean();hold=(pred[~g['mask']]-g['y'][~g['mask'],1]).square().mean()
        patch.backward(retain_graph=True);hold.backward();losses=[float(patch.detach()),float(hold.detach())]
        for ref in [old[oi],factual[k%7]]:
            loss=(obj.actor.sequence(ref['x'],ref['b'])[:,1]-ref['y'][:,1]).square().mean();loss.backward();losses.append(float(loss.detach()))
        assert all(np.isfinite(losses))
        assert torch.count_nonzero(obj.actor.head.weight.grad[0])==0 and obj.actor.head.bias.grad[0]==0
        norm=float(torch.nn.utils.clip_grad_norm_(obj.actor.head.parameters(),1.,error_if_nonfinite=True));event(dict(phase='BEFORE',step=k));obj.opt.step();obj.calls+=1;event(dict(phase='AFTER',step=k))
        assert fixed_hash(obj.actor)==fixed and all(v.grad is None for name,v in obj.actor.named_parameters() if not name.startswith('head.'))
        trace.append(dict(step=k,old_episode=oi,factual_history=k%7,patch_loss=losses[0],route_hold_loss=losses[1],old_hold_loss=losses[2],factual_hold_loss=losses[3],gradient_norm=norm))
    return trace


def repair_contract(obj,data):
    old,g,factual=data;pred=obj.actor.sequence(g['x'],g['b']);mask=g['mask'];patch=(pred[mask,1]-g['target'][mask]).square().mean();dw,db=torch.autograd.grad(patch,list(obj.actor.head.parameters()))
    assert torch.count_nonzero(dw[0])==0 and db[0]==0 and torch.count_nonzero(dw[1])>0 and db[1]!=0
    refs=[(pred[~mask,1]-g['y'][~mask,1]).abs().max()]
    for ref in [old[0],*factual]:
        refs.append((obj.actor.sequence(ref['x'],ref['b'])[:,1]-ref['y'][:,1]).abs().max())
    assert max(float(v) for v in refs)<1e-10
    assert float(patch)>0 and not obj.actor.encoder[0].weight.requires_grad and not obj.actor.core.weight_ih_l0.requires_grad and not obj.actor.future_head.weight.requires_grad
    return dict(state='PASS',effective_steering_parameters=17,allocated_head_parameters=34,drive_grad_exact_zero=True,body_future_head_frozen=True,initial_preservation_losses_below_1e_minus10=True,initial_patch_loss=float(patch.detach()),five_explicit_nonhuman_intervention_labels=True,not_onpolicy_or_human=True,native=0,Adam=0)


def descriptive_error(obj,data):
    _,g,_=data
    with torch.no_grad():pred=obj.actor.sequence(g['x'],g['b'])[:,1]
    return dict(patch_steer_rmse=float((pred[g['mask']]-g['target'][g['mask']]).square().mean().sqrt()),route_hold_steer_rmse=float((pred[~g['mask']]-g['y'][~g['mask'],1]).square().mean().sqrt()),not_closed_loop_navigation=True)
