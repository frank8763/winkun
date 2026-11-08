"""Independent zero-native, zero-optimizer audit of T428 early recoveries.

Imports are inert.  Recurrent teachers are evaluated on each actual episode's
observations from hold zero; the trigger is independently computed from the
student and teacher trajectories on their common original physics clock.
"""
from pathlib import Path
import fcntl,importlib.util,json,sys,time

ROOT=Path(__file__).resolve().parents[3]
_spec=importlib.util.spec_from_file_location('stage428_immutable427_audit',
    ROOT/'stage/experiments/t427_fixed_representation/audit.py')
T=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(T)
H=T.T
sha,jsonl=H.sha,H.jsonl


def full_policy(C,arrays,row,p):
    """Reuse the pinned T427 action, density, observation and teacher proof."""
    assert row['full'] and row['seed'] is None
    request=row['request'];mode=request.get('mode','student')
    assert mode in ['student','safe_reset','actor_teacher']
    assert request['model']==p['student_start_model']
    assert row['loaded_model_sha256']==p['student_start_sha256']
    assert row['policy_hash']==p['expected_policy_hash']
    if mode=='actor_teacher':
        assert request['teacher_model']==p['teacher_A_model']
        assert request['teacher_sha256']==p['teacher_A_sha256']
    first=request.get('first',0)
    assert type(first) is int and 0<=first<len(arrays['actions'])
    assert row['first']==first
    T.full_policy(C,arrays,row,dict(p,allowed_teacher_first=[first]))
    if mode=='student':
        assert not row['teacher'] and not arrays['ownership'].any()
    else:
        assert row['teacher']
        assert not arrays['ownership'][:first].any()
        assert arrays['ownership'][first:].all()


def independent_trigger(C,student,baseline,p):
    """Return raw common-clock measurements, without producer trigger code."""
    np=C.np
    sc=np.r_[0,np.cumsum(student['native_counts'].astype(np.int64))][:-1]
    tc=np.r_[0,np.cumsum(baseline['native_counts'].astype(np.int64))][:-1]
    assert np.all(student['native_counts'][:-1]==100)
    assert np.all(baseline['native_counts'][:-1]==100)
    clocks,si,ti=np.intersect1d(sc,tc,return_indices=True)
    assert len(clocks)>0 and clocks[0]==0
    # A complete nonterminal hold is 100 physical steps in the registered
    # Session; intersected observations must be consecutive holds in both runs.
    assert np.array_equal(si,np.arange(len(si)))
    assert np.array_equal(ti,np.arange(len(ti)))
    s=student['poses'][si];t=baseline['poses'][ti]
    distance=np.linalg.norm(s[:,:2]-t[:,:2],axis=1)
    yaw=np.abs((s[:,2]-t[:,2]+np.pi)%(2*np.pi)-np.pi)
    eligible=(distance>.10)|(yaw>np.deg2rad(5.))
    first=None
    for i in range(max(0,len(eligible)-2)):
        if bool(eligible[i:i+3].all()):
            first=i;break
    handoff=None if first is None else max(0,int(si[first])-1)
    return dict(clocks=clocks,student_indices=si,baseline_indices=ti,
        distance_m=distance,yaw_rad=yaw,eligible=eligible,
        first_run_index=first,handoff_hold=handoff)


def prefix(C,student,mixed,first):
    """Check the whole recorded causal student prefix and initial state."""
    E,np=C.E,C.np
    assert 0<=first<len(student['states']) and first<len(mixed['states'])
    for key in ['initial_native_integration','task_goal_xy','scene_sha256']:
        assert E.equal(student[key],mixed[key]),key
    n=len(student['actions']);m=len(mixed['actions'])
    for key,value in student.items():
        if key not in mixed or not hasattr(value,'shape') or not value.ndim:continue
        other=mixed[key]
        if value.shape[0]==n and other.shape[0]==m:
            assert E.equal(value[:first],other[:first]),('student prefix',key)
        elif value.shape[0]==n+1 and other.shape[0]==m+1:
            assert E.equal(value[:first+1],other[:first+1]),('student history',key)
    # The handoff receives the student's actual next observation/history;
    # no native reset or restarted remaining-time counter is permitted.
    for key in ['states','poses','hidden_before']:
        assert E.equal(student[key][first],mixed[key][first]),('handoff',key)
    assert np.array_equal(student['native_counts'][:first],mixed['native_counts'][:first])


def historical_equal(old,new):
    for key in ['job_id','array_digest','array_hashes','integration_sha256',
                'physics_steps','success','collision','timeout','policy_hash',
                'loaded_model_sha256','remaining_m']:
        assert old[key]==new[key],('historical reconstruction',old['job_id'],key)
    assert old['request']==new['request']


def replay_equal(row,reference,p):
    assert row['request']==reference['request']
    assert row['reference_path']==reference['path']
    assert row['complete'] and not row['administrative_censor']
    assert row['exact'] and row['all_arrays_exact'] and not row['independent_sample']
    assert row['physics']==row['physics_steps']
    for key in ['array_digest','array_hashes','integration_sha256','physics_steps',
                'success','collision','timeout','policy_hash','loaded_model_sha256',
                'mode','first','teacher','seed','remaining_m']:
        assert row[key]==reference[key],('exact replay',row['job_id'],key)
    assert row['loaded_model_sha256']==p['student_start_sha256']


def check_trigger(C,identity,actual,expected):
    """Compare the complete producer trigger tape against original arrays."""
    first=expected['first_run_index'];handoff=expected['handoff_hold']
    independently=dict(id=identity,status='TRIGGER' if first is not None else 'NO_TRIGGER',
        first=handoff,first_run_index=first,matched_holds=len(expected['clocks']),
        aligned_native_start_counts=expected['clocks'].astype(int).tolist(),
        distance_m=expected['distance_m'].tolist(),
        yaw_abs_deg=C.np.rad2deg(expected['yaw_rad']).tolist(),
        threshold_exceeded=expected['eligible'].astype(bool).tolist(),
        distance_threshold_m=.1,yaw_threshold_deg=5.,consecutive_holds=3,
        trigger_native_start_count=None if first is None else int(expected['clocks'][first]),
        handoff_native_start_count=None if handoff is None else int(expected['clocks'][handoff]),
        training_privileged_offline_diagnostic=True,runtime_future_access=False)
    assert actual==independently,('independent trigger',identity)


def diagnostic_audit(C,p,out,summary,rows,replays,arrays,guard):
    result=summary['result'];source=result['student_source']
    assert source==C.initial_record(p)
    assert source['path']==p['student_start_model']
    assert source['sha256']==p['student_start_sha256']==sha(source['path'])
    assert source['policy_hash']==p['expected_policy_hash']
    H.snapshot(C.E,source)
    previous=p['failed_rows'];assert len(previous)==7
    ids=[row['job_id'] for row in previous];assert len(set(ids))==7
    compact=[row for row in previous if not row['full']]
    full=[row for row in previous if row['full']]
    assert len(compact)==4 and len(full)==3
    assert all(not row['success'] or row['collision'] for row in previous)
    rebuilt=[row for row in rows if row['role']=='STUDENT_REBUILD']
    baselines=[row for row in rows if row['role']=='TEACHER_BASELINE']
    recoveries=[row for row in rows if row['role']=='RECOVERY']
    base_repeats=[row for row in replays if row['role']=='TEACHER_BASELINE_EXACT']
    recovery_repeats=[row for row in replays if row['role']=='RECOVERY_EXACT']
    assert len(rebuilt)==4 and [row['job_id'] for row in rebuilt]==[row['job_id'] for row in compact]
    assert len(baselines)==len(base_repeats)==7
    assert [row['job_id'] for row in baselines]==[row['job_id'] for row in base_repeats]==ids
    assert rows==rebuilt+baselines+recoveries
    assert replays==base_repeats+recovery_repeats
    students={}
    for old in previous:
        guard();assert str(old['path']) in p['inputs']
        assert sha(old['path'])==old['sha256']==p['inputs'][str(old['path'])]
        old_arrays=H.checked_array(C,old)
        assert old['request']['model']==source['path']
        assert old['loaded_model_sha256']==source['sha256']
        assert old['policy_hash']==source['policy_hash']
        assert not old['teacher'] and old['mode']=='student' and old['seed'] is None
        if old['full']:
            full_policy(C,old_arrays,old,p)
            students[old['job_id']]=(old_arrays,old)
    for old,row in zip(compact,rebuilt):
        guard();historical_equal(old,row)
        assert set(arrays[row['path']])==set(old['array_hashes'])
        students[old['job_id']]=(arrays[row['path']],row)
    assert result['baseline_rows']==baselines
    diagnostics=result['diagnostics'];assert len(diagnostics)==7
    assert [item['id'] for item in diagnostics]==ids
    assert C.read(out/'diagnostics.json')==diagnostics
    assert C.read(out/'triggers.json')==[item['trigger'] for item in diagnostics]
    by_recovery={row['job_id']:row for row in recoveries}
    assert len(by_recovery)==len(recoveries)
    expected_recovery_ids=[]
    for old,baseline,repeat,item in zip(previous,baselines,base_repeats,diagnostics):
        guard();identity=old['job_id'];student,srow=students[identity]
        assert item['student_row']==srow and item['student_row_path']==srow['path']
        assert item['baseline_row']==baseline
        mode='actor_teacher' if old['request']['job'].get('wide_course') else 'safe_reset'
        request=dict(model=source['path'],job=old['request']['job'],mode=mode,first=0,seed=None)
        if mode=='actor_teacher':request.update(teacher_model=p['teacher_A_model'],teacher_sha256=p['teacher_A_sha256'])
        assert baseline['request']==request and baseline['first']==0
        teacher=arrays[baseline['path']]
        for key in ['initial_native_integration','task_goal_xy','scene_sha256']:
            assert C.E.equal(student[key],teacher[key]),('same original task',identity,key)
        replay_equal(repeat,baseline,p)
        independent=independent_trigger(C,student,teacher,p)
        check_trigger(C,identity,item['trigger'],independent)
        safe=baseline['success'] and not baseline['collision']
        first=independent['handoff_hold']
        if not safe or first is None:
            assert item['recovery_row'] is None and identity not in by_recovery
            expected='TEACHER_BASELINE_NOT_ACTUAL_SAFE_GOAL' if not safe else 'NO_TRIGGER'
            assert item['recovery_skipped_reason']==expected
            continue
        assert 'recovery_skipped_reason' not in item
        expected_recovery_ids.append(identity)
        row=by_recovery[identity]
        assert item['recovery_row']==row and row['request']==dict(request,first=first)
        assert row['request']['job']['case']['step_cap']==old['request']['job']['case']['step_cap']
        prefix(C,student,arrays[row['path']],first)
    assert [row['job_id'] for row in recoveries]==expected_recovery_ids
    assert len(recovery_repeats)==len(recoveries)
    for row,repeat in zip(recoveries,recovery_repeats):
        assert row['job_id']==repeat['job_id'];replay_equal(repeat,row,p)
    assert result['recovery_rows']==recoveries
    assert result['student_rebuilds']==4 and result['reused_full_student_rows']==3
    assert result['baselines_replayed_exact']==7
    assert result['recoveries_replayed_exact']==len(recoveries)
    safe_count=sum(bool(row['success'] and not row['collision']) for row in recoveries)
    assert result['actual_safe_recoveries']==safe_count
    assert result['actual_failed_recoveries']==len(recoveries)-safe_count
    assert result['not_navigation_acceptance'] is True
    assert result['Actor_updates']==result['optimizer_calls']==0
    assert result['six_gate'] is False and result['eight_started'] is False
    return dict(reused_historical_full_students=3,rebuilt_historical_compact_students=4,
        historical_student_array_hashes_and_integration_exact=True,
        independently_recomputed_clock_aligned_triggers=7,
        teacher_baselines=7,exact_baseline_repeats=7,
        mixed_recoveries=len(recoveries),exact_mixed_repeats=len(recovery_repeats),
        actual_safe_recoveries=safe_count,actual_failed_recoveries=len(recoveries)-safe_count,
        all_completed_mixed_outcomes_repeated_including_failures=True,
        actual_teacher_observations_actions_locations_and_hidden_carry_verified=True,
        no_reset_at_handoff_and_original_task_clock_verified=True,
        first_handoff_determined_once_without_grid=True,
        teacher_recovery_independence_not_asserted=True,
        every_native_execution_charged_even_if_historical_behavior_repeats=True,
        new_navigation_acceptance=False)


def main():
    import t428_core as C
    p=C.read(C.OUT/'protocol.json');C.verify(p);out=C.OUT/'native'
    assert not (out/'audit.json').exists()
    for name in ['main_started.json','workers.json','summary.json','local_manifest.json']:
        assert (out/name).is_file(),name
    summary=C.read(out/'summary.json')
    assert summary['state']=='COMPLETE' and summary['all_workers_stopped'] and summary['all_sessions_closed']
    assert summary['Actor_updates']==summary['optimizer_calls']==0 and summary['error'] is None
    assert summary['no_restart'] is True
    lock=(C.ROOT/'stage/runs/.stage_writer.lock').open('a+b')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        absent=H.stopped(out);members=H.manifest(C.ROOT,out)
        assert not list(out.rglob('partial*.npz'))
        assert not (out/'optimizer_calls.jsonl').exists()
        assert not list(out.rglob('*.pt')) and not list(out.rglob('*.pt.gz'))
        start=time.time();deadline=min(start+300,p['deadline_all'])
        C.E.N.offline();C.torch.set_num_threads(1);C.torch.use_deterministic_algorithms(True)
        def guard():assert time.time()<deadline-5
        def prohibited(*args,**kwargs):raise AssertionError('T428 audit forbids native, optimizer, backward and model export')
        C.E.L.Session=prohibited;C.E.Pool=prohibited;C.Pool=prohibited
        C.PREVIOUS.Pool=prohibited
        C.E.N.S.P.C.B.setup=prohibited
        for optimizer in [C.torch.optim.Adam,C.torch.optim.AdamW,C.torch.optim.SGD]:optimizer.step=prohibited
        C.torch.Tensor.backward=prohibited
        C.E.export=prohibited;C.LEARNING.export_model=prohibited
        rows=jsonl(out/'cases.jsonl');replays=jsonl(out/'replays.jsonl');commands=jsonl(out/'commands.jsonl')
        assert rows and commands
        arrays={}
        for row in rows:
            guard();assert row['full'] and row['pid'] in absent
            assert Path(row['path']).resolve().is_relative_to(out.resolve())
            value=H.checked_array(C,row);full_policy(C,value,row,p)
            assert set(value)==set(row['array_hashes'])
            assert row['path'] not in arrays;arrays[row['path']]=value
        for row in replays:
            guard();assert row['pid'] in absent
            assert row['complete'] and not row['administrative_censor']
            assert row['physics']==row['physics_steps']
            assert row['exact'] and row['all_arrays_exact'] and not row['independent_sample']
        details=diagnostic_audit(C,p,out,summary,rows,replays,arrays,guard)
        assert [row['id'] for row in commands]==list(range(1,len(commands)+1))
        assert len(commands)==summary['commands']==4+len(rows)+len(replays)<=p['max_commands']
        assert [(row['role'],row['physics']) for row in commands[:4]]==[('SETUP',0)]*4
        expected=[]
        for role in ['STUDENT_REBUILD','TEACHER_BASELINE','TEACHER_BASELINE_EXACT','RECOVERY','RECOVERY_EXACT']:
            expected.extend((row['role'],row['physics'],row['pid']) for row in rows+replays if row['role']==role)
        assert [(row['role'],row['physics'],row['pid']) for row in commands[4:]]==expected
        assert all(row['pid'] in absent for row in commands)
        assert all(type(row['physics']) is int and row['physics']>=0 for row in commands)
        assert sum(row['physics'] for row in commands)==summary['physics']<=p['max_physics']
        assert p['max_physics']<=1100000 and p['max_commands']<=60
        assert summary['elapsed_s']<=p['main_seconds']
        guard();C.verify(p)
        C.save(out/'audit.json',dict(state='PASS',stage='native',native=0,
            new_navigation_training=0,new_RL=0,Adam=0,AdamW=0,SGD=0,backward=0,
            elapsed_s=time.time()-start,native_processes_absent=absent,
            manifest_members_before_audit=members,physics=summary['physics'],commands=summary['commands'],
            raw_controls_counts_terminal_and_command_ledger_exact=True,
            actual_full_payload_history_hidden_actions_rewards_and_densities_checked=True,
            every_persisted_payload_array_hash_verified=True,
            original_T422_history_audit_not_repaired=True,**details))
        C.E.N.manifest(out)
        print('T428_NATIVE_AUDIT_PASS',flush=True)
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':
    assert len(sys.argv)==1
    main()

