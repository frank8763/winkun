"""Fixed-representation, 66-parameter output repair for Stage T427.

The deployed architecture is unchanged. Only the existing final Linear32->2
weight and bias may change. All representation, recurrent, parent, adapter,
normalization and earlier extra-layer tensors stay fixed. Static preservation
and local gradient projection are not closed-loop safety guarantees.

Import is inert: no model construction, native simulation, optimizer update,
network request or filesystem write. This is an engineering pilot, not a claim
of an original algorithm or a proven solution to catastrophic interference.
"""
from pathlib import Path
import gzip,importlib.util,io,sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t418_evening_repair_chain'))
import repair_core as E

_spec=importlib.util.spec_from_file_location(
    'stage427_actual_execution_only_loss',
    ROOT/'stage/experiments/t422_feedback_recovery_night/learn.py')
EXECUTION=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EXECUTION)
torch,np=E.torch,E.np
TRAINABLE_NAMES=('extra.2.weight','extra.2.bias')


def params(q):
    """Never call historical E.params: it would unfreeze 10722 parameters."""
    q.requires_grad_(False)
    named=dict(q.named_parameters())
    assert all(name in named for name in TRAINABLE_NAMES)
    assert named['extra.2.weight'].shape==(2,32)
    assert named['extra.2.bias'].shape==(2,)
    for name in TRAINABLE_NAMES:named[name].requires_grad_(True)
    result=[value for name,value in q.named_parameters() if name in TRAINABLE_NAMES]
    assert sum(value.numel() for value in result)==66
    assert all(value.requires_grad==(name in TRAINABLE_NAMES)
               for name,value in q.named_parameters())
    return result


def fixedhash(q):
    """Hash every state tensor except the two explicitly selected tensors."""
    return E.digest({name:value for name,value in q.state_dict().items()
                     if name not in TRAINABLE_NAMES})


def _pair(entry):
    if isinstance(entry,dict):return entry['arrays'],entry['row']
    assert isinstance(entry,(tuple,list)) and len(entry)==2
    return entry[0],entry[1]


def _safe_arrays(arrays):
    states=np.asarray(arrays['states']);flags=np.asarray(arrays['end_flags'])
    assert states.ndim==2 and states.shape[1]==120 and len(states)>0
    assert flags.shape==(len(states),3)
    assert np.array_equal(flags[-1],[0,1,0]) and not flags[:-1].any()
    assert np.asarray(arrays['actions']).shape==(len(states),2)
    assert np.isfinite(states).all() and np.isfinite(arrays['actions']).all()
    assert np.abs(arrays['actions']).max()<=1


def _label(q,states,locations,actions):
    value=E.labelled(q,states,locations,actions)
    return {key:item.detach().clone() if isinstance(item,torch.Tensor) else item
            for key,item in value.items()}


def groups(q,p,anchors,targets):
    """Keep genuine current successes and supervise verified actual controls.

    Production contains49 current-six and8 current-old-centre successful
    histories plus7 verified recovery episodes. A marked preflight may use
    fewer authenticated historical fixtures; those are never called new
    current-policy successes. Broad211 states retain this starting student's
    own locations, rather than pulling them back toward T255 locations.
    """
    params(q)
    probe=p.get('preflight_clone_probe') is True
    old_ids={job['id'] for job in p['old8_jobs']}
    six_ids={job['id'] for job in p['main_jobs']}
    assert len(old_ids)==8 and len(six_ids)==54 and not old_ids&six_ids
    assert anchors and targets
    old=[];six=[];seen=set()
    for entry in anchors:
        arrays,row=_pair(entry);_safe_arrays(arrays);identity=row['job_id']
        assert identity not in seen and identity in old_ids|six_ids
        seen.add(identity)
        assert row['success'] and not row['collision'] and not row['timeout']
        assert not np.asarray(arrays['ownership']).any()
        assert not np.asarray(arrays['stochastic']).any()
        if not probe:assert row['policy_hash']==q.phash()
        item=_label(q,arrays['states'],arrays['locations'],arrays['actions'])
        item.update(id=identity,actual_complete_history=True,
                    recorded_starting_student_location=True,
                    historical_preflight_fixture=probe)
        (old if identity in old_ids else six).append(item)
    assert old and six
    if probe:assert len(old)<=8 and len(six)<=49 and len(targets)<=7
    else:assert len(old)==8 and len(six)==49 and len(targets)==7

    history=E.N.old_data(q,E.N.science(p),lambda:None)
    assert len(history)==211 and {item['task'] for item in history}==set(range(8))
    broad=[]
    for item in history:
        # This target is frozen before any fit, evaluated with the complete
        # recorded historical observation sequence and the starting student.
        with torch.no_grad():location=q.sequence(item['x'],item['b']).detach().clone()
        target=_label(q,item['x'],location,location.clamp(-1,1).float())
        target.update(id=item['id'],task=item['task'],self_distillation=True,
                      self_distillation_policy_hash=q.phash(),
                      historical_states_not_current_onpolicy=True)
        broad.append(target)

    corrections=[];ids=set()
    for target in targets:
        identity=target['id'];assert identity not in ids;ids.add(identity)
        _safe_arrays(target['arrays']);first=int(target['first'])
        assert first in [1,0] or (probe and first>=0)
        if not probe:assert first in p.get('allowed_teacher_first',[1,0])
        corrections.append(EXECUTION.recovery_group(
            q,target['arrays'],first,identity))
    return dict(old=old,six=six,broad=broad,corrections=corrections,
                old_weight=1.,six_weight=1.,broad_weight=1.,correction_weight=1.,
                fixed_representation=True,trainable_parameters=66,
                original_211_targets_are_starting_student_locations=True,
                historical_preflight_fixture_only=probe)


make_groups=groups


def support_losses(q,g,p,k):
    assert g['fixed_representation'] and g['trainable_parameters']==66
    assert g['original_211_targets_are_starting_student_locations']
    assert g['old_weight']==g['six_weight']==g['broad_weight']==g['correction_weight']==1.
    old=sum(E.old_kl(q,item,p) for item in g['old'])/len(g['old'])
    six=sum(E.old_kl(q,item,p) for item in g['six'])/len(g['six'])
    choices=[item for item in g['broad'] if item['task']==k%8]
    assert choices
    broad=E.old_kl(q,choices[(k//8)%len(choices)],p)
    correction=sum(EXECUTION.controlled_loss(q,item,p)
                   for item in g['corrections'])/len(g['corrections'])
    losses=[old,six,broad,correction]
    assert all(bool(torch.isfinite(loss)) for loss in losses)
    return losses


def make_optimizer(q,p):
    return torch.optim.Adam(params(q),lr=p['lr'],foreach=False,fused=False)


def fit(q,opt,g,p,guard,event,steps=120):
    """One fixed120-step proposal; callers separately execute exact shadow."""
    assert steps==120 or (steps==1 and p.get('preflight_clone_probe') is True)
    assert opt.defaults.get('foreach') is False and opt.defaults.get('fused') is False
    variables=params(q)
    assert len(opt.param_groups)==1
    assert [id(value) for value in opt.param_groups[0]['params']]==[id(v) for v in variables]
    fixed=fixedhash(q);trace=[]
    for k in range(steps):
        guard();opt.zero_grad(set_to_none=True)
        losses=support_losses(q,g,p,k)
        values=[float(loss.detach()) for loss in losses]
        reference=E.vector_gradient(losses[0]+losses[1]+losses[2],variables)
        correction=E.protect_vector(E.vector_gradient(losses[3],variables),reference)
        E.install_vector(variables,reference+correction)
        norm=float(torch.nn.utils.clip_grad_norm_(variables,1.,error_if_nonfinite=True))
        event('BEFORE',k);opt.step();event('AFTER',k)
        trace.append([*values,norm])
        assert all(value.grad is None for name,value in q.named_parameters()
                   if name not in TRAINABLE_NAMES)
        if k%16==0:assert fixedhash(q)==fixed
    assert fixedhash(q)==fixed
    return dict(trace=trace,steps=steps,new_RL=0,trainable_parameters=66,
                trainable_names=list(TRAINABLE_NAMES),fixed_state_hash=fixed,
                frozen_encoder_core_and_all_other_tensors=True,
                unchanged_deployed_architecture=True,
                current_old_anchor_count=len(g['old']),six_anchor_count=len(g['six']),
                recovery_count=len(g['corrections']),
                old_211_starting_student_self_distillation=True,
                full_actual_history_before_teacher_suffix_mask=True,
                actual_executed_controls_only_no_teacher_latent=True,
                no_teacher_data_in_onpolicy_ratios=True,
                projection_is_local_soft_rule_not_closed_loop_guarantee=True)


def export_model(q,opt,path):
    """Own metadata: historical E.export would claim10722 trainable tensors."""
    assert sum(value.numel() for value in q.parameters() if value.requires_grad)==66
    state=E.L.state(q,opt);hashes=E.L.statehash(q,opt)
    buffer=io.BytesIO()
    torch.save(dict(state=state,hashes=hashes,policy_hash=q.phash(),memory_on=True,
        architecture='unchanged_T426_Actor_fixed_representation_existing_extra_final_linear_only',
        deployed_actor_count=1,trainable_parameters=66,
        trainable_names=list(TRAINABLE_NAMES),frozen_state_hash=fixedhash(q),
        teacher_at_deployment=False),buffer)
    data=gzip.compress(buffer.getvalue(),mtime=0)
    assert len(data)<=512*1024
    Path(path).write_bytes(data)
    return dict(path=str(path),sha256=E.sha(path),hashes=hashes,policy_hash=q.phash())


def load_complete_state(path):
    return torch.load(io.BytesIO(gzip.decompress(Path(path).read_bytes())),
                      map_location='cpu',weights_only=False)
