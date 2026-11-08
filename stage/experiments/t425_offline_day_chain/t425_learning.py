"""Fixed supervised-recovery and fresh-PPO learners for T425.

Only actual successful execution controls become recovery labels.  Current
student successes retain their recorded full-history latent locations as
zero-initial-loss anchors.  Teacher trajectories never enter PPO ratios.
This module performs no optimizer, native simulation or file writes at import.
"""
from pathlib import Path
import importlib.util
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t418_evening_repair_chain'))
import repair_core as E

_spec=importlib.util.spec_from_file_location(
    'stage425_registered_execution_target_learning',
    ROOT/'stage/experiments/t422_feedback_recovery_night/learn.py')
LEARN=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LEARN)
torch,np=E.torch,E.np


def _anchor_pair(entry):
    """Accept a native (arrays,row) pair or the equivalent explicit dict."""
    if isinstance(entry,dict):
        assert 'arrays' in entry and 'row' in entry
        return entry['arrays'],entry['row']
    assert isinstance(entry,(list,tuple)) and len(entry)==2
    return entry[0],entry[1]


def make_groups(q,p,anchors,targets):
    """Build immutable recorded targets without inventing teacher latents.

    ``anchors`` contains successful current-student native full episodes.
    ``targets`` contains only fully verified safe-GOAL recovery arrays with
    ``first`` and ``id``. Source SHA/exact replay/native-budget evidence is
    checked by the registered pipeline and its independent auditor.
    """
    assert p['phase'] in ['protect','rl']
    assert 0<len(anchors)<=80 and 0<len(targets)<=19
    actual=[];seen=set()
    for entry in anchors:
        arrays,row=_anchor_pair(entry)
        assert row['success'] and not row['collision'] and not row['timeout']
        flags=np.asarray(arrays['end_flags'])
        assert flags[-1,1] and not flags[:,0].any() and not flags[-1,2]
        assert not flags[:-1].any()
        assert not np.asarray(arrays['ownership']).any()
        assert not np.asarray(arrays['stochastic']).any()
        assert arrays['states'].shape[1:]==(120,)
        identity=row['job_id'];assert identity not in seen;seen.add(identity)
        # These recorded locations belong to the actual student history.
        # Recomputing them from a later candidate would erase the constraint.
        item=E.labelled(q,arrays['states'],arrays['locations'],arrays['actions'])
        item.update(id=identity,own_initial_success=True,
                    recorded_current_student_latent_target=True,
                    actual_full_causal_history=True)
        actual.append(item)

    broad=E.N.old_data(q,E.N.science(p),lambda:None)
    assert len(broad)==211 and {item['task'] for item in broad}==set(range(8))
    old_broad=[]
    for item in broad:
        entry=E.labelled(q,item['x'],item['b'],item['b'].clamp(-1,1).float())
        entry.update(task=item['task'],id=item['id'])
        old_broad.append(entry)

    left=[]
    assert len(p['left_history_files'])==5
    for path in p['left_history_files']:
        arrays=E.N.S.arrays(path)
        assert arrays['end_flags'][-1,1] and not arrays['end_flags'][:,0].any()
        left.append(E.labelled(q,arrays['states'],arrays['locations'],arrays['actions']))
    arrays=E.N.S.arrays(p['right_history_file'])
    assert arrays['end_flags'][-1,1] and not arrays['end_flags'][:,0].any()
    right=E.labelled(q,arrays['states'],arrays['locations'],arrays['actions'])

    corrections=[]
    for entry in p.get('query_sources',[]):
        item=E.checked_query(q,entry,p)
        item.update(executed_teacher=False,original_checked_query=True)
        corrections.append(item)
    identities=set()
    for entry in targets:
        assert isinstance(entry,dict) and set(['arrays','first','id'])<=set(entry)
        assert entry['id'] not in identities;identities.add(entry['id'])
        item=LEARN.recovery_group(q,entry['arrays'],entry['first'],entry['id'])
        corrections.append(item)
    assert len(identities)==len(targets)
    return dict(old=actual,anchors=actual,broad=old_broad,left=left,
                right=right,corrections=corrections)


def make_optimizer(q,p):
    """Explicit CPU Adam path for new T425; does not repair old T422 history."""
    assert p['phase'] in ['protect','rl']
    variables=E.params(q)
    assert sum(v.numel() for v in variables)==10722
    return torch.optim.Adam(variables,lr=p['lr'],foreach=False,fused=False)


def fit_a(q,opt,groups,p,guard,event,steps=120):
    """Bounded execution-target supervision with old-gradient priority."""
    assert steps==120 or (steps==1 and p.get('preflight_clone_probe') is True)
    assert opt.defaults.get('foreach') is False and opt.defaults.get('fused') is False
    return LEARN.fit_bc(q,opt,groups,p,guard,event,steps=steps)


def fit_b(q,opt,items,groups,p,rnd,guard,event):
    """Fresh on-policy PPO only; verified recovery targets remain auxiliary."""
    assert p.get('epochs',8)==8 and len(items)==4
    assert opt.defaults.get('foreach') is False and opt.defaults.get('fused') is False
    return LEARN.fit_ppo(q,opt,items,groups,p,rnd,guard,event)


def export_model(q,opt,path):
    return E.export(q,opt,path)


def load_complete_state(path):
    """Load Actor+Adam snapshot; historical read_model returns learned only."""
    return torch.load(E.io.BytesIO(E.gzip.decompress(Path(path).read_bytes())),
                      weights_only=False,map_location='cpu')
