"""Fixed family-balanced supervised retention pilot for Stage T426.

The old-centre and six-metre anchor families each have unit weight.  Their
targets are the starting student's recorded real full-history locations.
Recovery targets are only actual controls from complete safe-GOAL native
episodes; no teacher latent or on-policy density is invented.  Gradient
projection is a local soft rule, never a closed-loop safety guarantee.

Import is inert: no model construction, native simulation, optimizer step,
network call or file write.  This is an engineering pilot using known methods.
"""
from pathlib import Path
import importlib.util
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t418_evening_repair_chain'))
import repair_core as E

_spec=importlib.util.spec_from_file_location(
    'stage426_execution_only_recovery_loss',
    ROOT/'stage/experiments/t422_feedback_recovery_night/learn.py')
EXECUTION=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EXECUTION)
torch,np=E.torch,E.np


def _anchor_pair(entry):
    if isinstance(entry,dict):return entry['arrays'],entry['row']
    assert isinstance(entry,(list,tuple)) and len(entry)==2
    return entry[0],entry[1]


def _labelled(q,states,locations,actions):
    """Detach immutable recorded labels; the fixed base is only a feature."""
    item=E.labelled(q,states,locations,actions)
    return {key:value.detach().clone() if isinstance(value,torch.Tensor) else value
            for key,value in item.items()}


def groups(q,p,anchors,targets):
    """Make the two equally weighted real-success families and recoveries.

    Source hashes, exact native replay and the starting policy identity are
    verified by the caller/auditor.  Production has exactly54+7 anchors and11
    recoveries.  A separately marked zero-native one-step preflight may use
    smaller authentic historical fixtures; it cannot export a retained Actor.
    """
    assert p['phase']=='protect'
    probe=p.get('preflight_clone_probe') is True
    assert anchors and targets
    six_ids={job['id'] for job in p['main_jobs']}
    old_ids={job['id'] for job in p['old8_jobs']}
    assert len(six_ids)==54 and len(old_ids)==8 and not six_ids&old_ids
    old=[];six=[];seen=set()
    for entry in anchors:
        arrays,row=_anchor_pair(entry);identity=row['job_id']
        assert identity not in seen;seen.add(identity)
        assert row['success'] and not row['collision'] and not row['timeout']
        flags=np.asarray(arrays['end_flags'])
        assert flags.shape==(len(arrays['states']),3)
        assert np.array_equal(flags[-1],[0,1,0]) and not flags[:-1].any()
        assert arrays['states'].shape[1:]==(120,)
        assert not np.asarray(arrays['ownership']).any()
        assert not np.asarray(arrays['stochastic']).any()
        assert identity in six_ids|old_ids
        item=_labelled(q,arrays['states'],arrays['locations'],arrays['actions'])
        item.update(id=identity,own_starting_student_success=True,
                    actual_full_causal_history=True,
                    recorded_location_target_not_recomputed=True)
        (six if identity in six_ids else old).append(item)
    assert six and old
    if probe:assert len(six)<=54 and len(old)<=7 and len(targets)<=11
    else:assert len(six)==54 and len(old)==7 and len(targets)==11

    historical=E.N.old_data(q,E.N.science(p),lambda:None)
    assert len(historical)==211 and {item['task'] for item in historical}==set(range(8))
    broad=[]
    for item in historical:
        labelled=_labelled(q,item['x'],item['b'],item['b'].clamp(-1,1).float())
        labelled.update(task=item['task'],id=item['id']);broad.append(labelled)
    corrections=[];seen_targets=set()
    for target in targets:
        assert target['id'] not in seen_targets;seen_targets.add(target['id'])
        # Production first tries one real student hold. The registered t0
        # fallback, if used after actual recovery failure, remains explicit.
        assert target['first'] in p.get('allowed_teacher_first',[1,0]) or (probe and target['first']>=0)
        item=EXECUTION.recovery_group(q,target['arrays'],target['first'],target['id'])
        corrections.append(item)
    return dict(old_centres=old,six=six,broad=broad,corrections=corrections,
                old_family_weight=1.,six_family_weight=1.,broad_weight=1.,
                correction_weight=1.,no_legacy_left_right_query_auxiliary=True,
                preflight_fixture_only=probe)


make_groups=groups


def support_losses(q,g,p,k):
    """Return old-centre, six, balanced legacy support and control losses."""
    assert g['old_family_weight']==g['six_family_weight']==1.
    assert g['broad_weight']==g['correction_weight']==1.
    assert g['no_legacy_left_right_query_auxiliary'] is True
    old=sum(E.old_kl(q,item,p) for item in g['old_centres'])/len(g['old_centres'])
    six=sum(E.old_kl(q,item,p) for item in g['six'])/len(g['six'])
    options=[item for item in g['broad'] if item['task']==k%8]
    assert options
    broad=E.old_kl(q,options[(k//8)%len(options)],p)
    correction=sum(EXECUTION.controlled_loss(q,item,p)
                   for item in g['corrections'])/len(g['corrections'])
    losses=[old,six,broad,correction]
    assert all(bool(torch.isfinite(loss)) for loss in losses)
    return losses


def make_optimizer(q,p):
    assert p['phase']=='protect'
    variables=E.params(q)
    assert sum(value.numel() for value in variables)==10722
    return torch.optim.Adam(variables,lr=p['lr'],foreach=False,fused=False)


def stepfit(q,opt,g,p,start,count,guard,event):
    """Execute a fixed slice of one120-update research trajectory.

    Caller separately snapshots/evaluates at60 and120 and selects only a
    candidate satisfying its registered complete closed-loop acceptance gate.
    This routine performs no acceptance, rollback, sampling or model export.
    """
    probe=p.get('preflight_clone_probe') is True
    assert (start,count) in [(0,60),(60,60)] or (probe and (start,count)==(0,1))
    assert p['phase']=='protect' and opt.defaults.get('foreach') is False
    assert opt.defaults.get('fused') is False
    variables=E.params(q)
    assert sum(value.numel() for value in variables)==10722
    assert all(value.requires_grad==name.startswith(E.trained_prefixes(q))
               for name,value in q.named_parameters())
    fixed=E.original_hash(q);trace=[]
    for k in range(start,start+count):
        guard();opt.zero_grad(set_to_none=True)
        losses=support_losses(q,g,p,k)
        values=[float(loss.detach()) for loss in losses]
        reference=E.vector_gradient(losses[0]+losses[1],variables)
        broad=E.protect_vector(E.vector_gradient(losses[2],variables),reference)
        correction=E.protect_vector(E.vector_gradient(losses[3],variables),reference)
        E.install_vector(variables,reference+broad+correction)
        norm=float(torch.nn.utils.clip_grad_norm_(variables,1.,error_if_nonfinite=True))
        event('BEFORE',k);opt.step();event('AFTER',k)
        trace.append([*values,norm])
        assert all(value.grad is None for name,value in q.named_parameters()
                   if not name.startswith(E.trained_prefixes(q)))
        if k%32==0:assert E.original_hash(q)==fixed
    assert E.original_hash(q)==fixed
    return dict(trace=trace,steps=count,absolute_step_start=start,
                absolute_step_end_exclusive=start+count,new_RL=0,
                trainable_parameters=10722,original_frozen_hash=fixed,
                old_centre_anchor_count=len(g['old_centres']),
                six_anchor_count=len(g['six']),recovery_count=len(g['corrections']),
                equal_unit_weight_old_and_six_families=True,
                full_actual_history_before_suffix_mask=True,
                actual_executed_controls_only_no_teacher_latent=True,
                no_onpolicy_density_or_teacher_at_deployment=True,
                anchor_gradient_priority_is_local_soft_rule_not_safety_guarantee=True,
                no_legacy_left_right_query_auxiliary=True)


def export_model(q,opt,path):return E.export(q,opt,path)


def load_complete_state(path):
    return torch.load(E.io.BytesIO(E.gzip.decompress(Path(path).read_bytes())),
                      weights_only=False,map_location='cpu')
