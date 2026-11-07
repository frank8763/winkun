"""Frozen current-history LEFT-policy recovery, training-label diagnostic only.

The student and teacher both ingest the actual causal observation history from
episode zero. Only the actor that owns a decision supplies its executed action.
There are no optimizer steps, physical resets at handoff, or action suffixes.
"""
from pathlib import Path
import sys,os,json,time,copy,signal,traceback,fcntl,gzip,io
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t410_day_offline_chain'))
from day_core import *
import day_core as STUDENT_CORE
import t406_core as TEACHER_CORE
import t404_core as STUDENT_AUDIT
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t411_closed_teacher_recovery'
DOC=ROOT/'stage/docs/day_20261107'

PHYSICAL_KEYS=['states','actions','native_counts','next_states','poses','next_poses',
               'end_flags','rewards','history_poses','history_goals','history_times',
               'environment_rewards','initial_native_integration','final_native_integration']
TEACHER_KEYS=['teacher_actions','teacher_locations','teacher_hidden_before','teacher_hidden_after']

def plan():return read(OUT/'protocol.json')

def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    N.capacity(OUT,p['final_bytes'])
    outside=sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file() and OUT not in f.parents)
    assert outside+p['final_bytes']<=2**30
    return N.science(p)

def actors(base,p):
    student=STUDENT_CORE.Actor(base,p).eval().requires_grad_(False)
    teacher=TEACHER_CORE.embedded(base,p).eval().requires_grad_(False)
    teacher.load_learned(N.S.read_model(p['teacher_model']))
    # The verified legacy embedding has a zero contribution from its later
    # observation adapter, not the unrelated inherited T393 adapter.
    assert not torch.count_nonzero(teacher.adapter[-1].weight)
    assert not torch.count_nonzero(teacher.adapter[-1].bias)
    assert teacher.phash()==p['teacher_policy_hash']
    assert student.phash()==p['student_policy_hash']
    for actor in [student,teacher]:actor.sample=False;actor.reset_noise();assert not any(v.requires_grad for v in actor.parameters())
    return student,teacher

def prohibited(*args,**kwargs):raise AssertionError('T411 cannot optimize or backpropagate')

def install_no_learning():
    # Install after Torch and actors are initialized, avoiding historical lazy
    # Torch/Dynamo initialization failures from early method replacement.
    torch.optim.Adam.step=prohibited;torch.Tensor.backward=prohibited

def finish_payload(session,tape):
    result=STUDENT_AUDIT.payload(session)
    result.update({key:np.asarray([row[key] for row in tape]) for key in TEACHER_KEYS})
    assert len(result['states'])==len(tape)
    return result

def rollout(case,student,teacher,native,create,witness,guard,charge,partial_path):
    session=None;teacher_hidden=None;teacher_tape=[]
    student.reset_noise();teacher.reset_noise();student.sample=teacher.sample=False
    try:
        session=L.Session(case['job'],native,create,witness,guard,charge,student)
        L.install_stop_handlers()
        while not session.done():
            guard();hold=len(session.rows);x=session.packet().copy()
            before=np.zeros(16) if teacher_hidden is None else teacher_hidden[0,0].numpy().copy()
            with torch.no_grad():command,location,teacher_hidden,_=teacher.step(torch.tensor(x[None]),teacher_hidden)
            teacher_tape.append(dict(teacher_actions=command[0].numpy().copy(),
                teacher_locations=location[0].numpy().copy(),teacher_hidden_before=before,
                teacher_hidden_after=teacher_hidden[0,0].numpy().copy()))
            if hold>=case['first']:
                session.step(command[0].numpy(),owner=1,actor_active=True)
            else:session.step(owner=0)
            assert equal(session.rows[-1][0],x)
        payload=finish_payload(session,teacher_tape);row=session.result()
        row.update(student_policy_hash=student.phash(),teacher_policy_hash=teacher.phash(),
                   array_digest=digest(payload),integration_sha256=digest(payload['final_native_integration']),
                   source_control=bool(case.get('source_control',False)),first=case['first'],case_id=case['id'],
                   no_native_reset_at_handoff=True,teacher_hidden_reconstructed_from_actual_episode_history=True)
        return payload,row
    except BaseException:
        if session is not None and session.rows:
            # An interrupted decision may leave one teacher proposal ahead of
            # the completed student rows; store only completed row evidence.
            payload=finish_payload(session,teacher_tape[:len(session.rows)])
            N.write_npz(partial_path,N.S.packed(payload),plan()['full_bytes'])
        raise
    finally:
        if session is not None:session.close()

def source_control(a,p):
    original=N.S.arrays(p['source_control_file'])
    keys=p.get('source_control_shared_keys',PHYSICAL_KEYS)
    assert set(N.R.NAMES)<=set(keys)
    assert {'initial_native_integration','final_native_integration','history_poses','history_goals','history_times'}<=set(keys)
    for key in keys:assert equal(a[key],original[key]),('Source control mismatch',key)
    assert a['end_flags'][-1,1] and not a['end_flags'][:,0].any()
    return dict(state='EXACT_HISTORICAL_LEFT_PHYSICS_AND_CAUSAL_INPUTS',
        file=p['source_control_file'],sha256=sha(p['source_control_file']),keys=keys)

def audit_arrays(student,teacher,a,case,p):
    STUDENT_AUDIT.audit_array(student,a,p,0,None)
    assert not a['stochastic'].any()
    first=case['first'];n=len(a['states'])
    assert not a['ownership'][:min(first,n)].any()
    assert np.all(a['ownership'][first:]==1)
    teacher.reset_noise();teacher.sample=False;hidden=None
    with torch.no_grad():
        for index,x in enumerate(a['states']):
            before=np.zeros(16) if hidden is None else hidden[0,0].numpy().copy()
            action,location,hidden,_=teacher.step(torch.tensor(x[None]),hidden)
            assert equal(before,a['teacher_hidden_before'][index])
            assert equal(hidden[0,0].numpy(),a['teacher_hidden_after'][index])
            assert equal(action[0].numpy(),a['teacher_actions'][index])
            assert equal(location[0].numpy(),a['teacher_locations'][index])
            if index>=first:assert equal(action[0].numpy(),a['actions'][index])
    assert not any(v.requires_grad for v in student.parameters())
    assert not any(v.requires_grad for v in teacher.parameters())
    if case.get('source_control'):source_control(a,p)
