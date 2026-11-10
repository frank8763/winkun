"""Two fixed joint-repair objectives; raw control labels remain off-policy."""
from pathlib import Path
import importlib.util
ROOT=Path(__file__).resolve().parents[3]
sp=importlib.util.spec_from_file_location('stage474_pinned472',ROOT/'stage/experiments/t472_joint_recovery_preservation/learning.py')
B=importlib.util.module_from_spec(sp);sp.loader.exec_module(B)
C,N=B.C,B.N
J,H,M,P,E,R,torch,np=B.J,B.H,B.M,B.P,B.E,B.R,B.torch,B.np
read,save,sha,digest=B.read,B.save,B.sha,B.digest
make,state,load,export,frozen,base_digest=B.make,B.state,B.load,B.export,B.frozen,B.base_digest
MODES=['joint','priority']

def data(p,device='cuda'):
    initial,rejected=C._input_model(p)
    originals=p['original_corrections'];assert len(originals)==2
    seeds=p['seed_occupancy_rows']+p['baseline_success_rows'];assert len(seeds)==10
    # Only original327 pure-student histories enter this safe-own path.
    batch=N.data(p,seeds,originals,'occupancy',device=device)
    assert {r['job_id']:r['teacher_first'] for r in originals}=={'dev_task6_00':26,'dev_task6_21':20}
    for row in originals:
        assert row['success'] and not row['collision']
        if row['job_id']=='dev_task6_21':assert row['recurrent_teacher_model_sha256']==p['recovery_teacher_model']['sha256']
        else:assert row['teacher_kind']=='safeT255reset' and row['teacher_source_SHA256']==p['expected_reset_teacher_digest']
    for g in batch['corrections']:g['raw_control_only']=False
    reset=p['new_reset03'];a=B._history(reset,rejected,safe=True)
    assert reset['job_id']=='dev_task6_03' and reset['teacher_first']==20
    assert reset['teacher_kind']=='safeT255reset' and reset['teacher_source_SHA256']==p['expected_reset_teacher_digest']
    g=B._group(reset,a,device);g['raw_control_only']=False;batch['corrections'].append(g)
    row=p['verified_raw15'];a=B._history(row,rejected,safe=True)
    assert row['job_id']=='dev_task6_15' and row['teacher_first']==20 and row['raw_teacher_control']
    assert row['teacher_kind']=='native_full_horizon_phasewarp_CEM'
    assert row['array_digest']==p['raw15_reference']['array_digest']
    assert np.array_equal(a['teacher_actions'],a['actions'])
    g=B._group(row,a,device);g['raw_control_only']=True;batch['corrections'].append(g)
    assert len(batch['corrections'])==4
    # Successful current327 trajectories and verified wide recovery histories
    # are finite function support, never failed-A action targets.
    wide_students=p['wide_counterexample_student_rows'];wide_recoveries=p['wide_counterexample_recovery_rows']
    assert len(wide_students)==len(wide_recoveries)==2
    assert {r['job_id'] for r in wide_students}==B.WIDE_IDS
    source={r['job_id']:C._history(r,rejected) for r in wide_students}
    anchor_cpu=frozen(p,initial['path']).eval().requires_grad_(False)
    anchor=frozen(p,initial['path']).to(device).eval().requires_grad_(False)
    groups=[];provenance=[]
    for kind,rows in [('failed_A_history',wide_students),('verified_wide_recovery',wide_recoveries)]:
        for row in rows:
            a=source[row['job_id']] if kind=='failed_A_history' else C._history(row,rejected,True)
            if kind=='verified_wide_recovery':
                assert row['teacher_source_SHA256']==initial['sha256'];B._matching_prefix(a,source[row['job_id']],row['teacher_first'])
            x,target,check=C._target(anchor_cpu,anchor,a['states'],device)
            groups.append(dict(x=x,target=target,id=row['job_id'],kind=kind))
            provenance.append(dict(path=row['path'],sha256=row['sha256'],id=row['job_id'],kind=kind,target_policy=initial['policy_hash'],source_actions_NOT_targets=True,**check))
    batch.update(counterexample_groups=groups,counterexample_provenance=provenance,
        correction_count=4,raw15_ownership_suffix_only=True,initial_counterexample_function=0.,mode=p['mode'])
    return batch

def _preservation(q,batch):
    ref,own=N._retention(q,batch);extra=0.
    for g in batch['counterexample_groups']:
        loss=(q.sequence(g['x'])-g['target']).square().mean()/4
        loss.backward();extra+=float(loss.detach())
    return ref,own,extra

def _flat(parameters):
    return torch.cat([(v.grad if v.grad is not None else torch.zeros_like(v)).reshape(-1) for v in parameters])

def _set(parameters,flat):
    offset=0
    for v in parameters:
        v.grad=flat[offset:offset+v.numel()].reshape_as(v).clone();offset+=v.numel()
    assert offset==flat.numel()

def bc_step(obj,batch,mode=None):
    q,opt=obj;mode=batch['mode'] if mode is None else mode;assert mode==batch['mode'] and mode in MODES
    C._parameters(q);opt.zero_grad(set_to_none=True)
    parameters=C.L._trainable(q)
    # Keep both task gradients before the one true mature-Adam step.
    result=dict(mode=mode,student_prefix=0.,teacher_suffix=0.,raw15_suffix=0.,correction_objective=0.)
    for g in batch['corrections']:
        loc=q.sequence(g['x']);axis=P.action_loss(loc,g['y']);first=g['first'];n=first+1
        if g['raw_control_only']:
            loss=axis[first:].mean();result['raw15_suffix']+=float(loss.detach())
        else:
            early=axis[:n].mean();tail=axis[n:].mean() if n<len(loc) else loc.sum()*0
            loss=early+.1*tail;result['student_prefix']+=float(early.detach())/4;result['teacher_suffix']+=float(tail.detach())/4
        (loss/4).backward();result['correction_objective']+=float(loss.detach())/4
    correction=_flat(parameters).clone();opt.zero_grad(set_to_none=True)
    ref,own,extra=_preservation(q,batch);preserve=_flat(parameters).clone()
    dot=torch.dot(correction,preserve);den=torch.dot(preserve,preserve)
    projected=mode=='priority' and float(dot)<0 and float(den)>0
    if projected:correction=correction-dot/den*preserve
    combined=correction+preserve;_set(parameters,combined)
    result.update(reference218=ref,own10=own,counterexample_function=extra,
        correction_preservation_dot=float(dot),preservation_norm_sq=float(den),priority_projection=projected)
    result['gradient_norm']=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
    C._parameters(q);opt.step();C._parameters(q)
    return result
