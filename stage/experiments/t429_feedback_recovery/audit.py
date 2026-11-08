"""Zero-native, zero-optimizer audit of five fixed T429 feedback recoveries.

Original T428 evidence is immutable.  Feedback is reconstructed at every hold
from the actual current pose, actual speed and original clock; no teacher
latent or native state is transplanted from the successful reference.
"""
from pathlib import Path
import fcntl,importlib.util,sys,time

ROOT=Path(__file__).resolve().parents[3]
_spec=importlib.util.spec_from_file_location('stage429_immutable428_audit',
    ROOT/'stage/experiments/t428_early_recovery/audit.py')
O=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(O)
T,H=O.T,O.H
sha,jsonl=H.sha,H.jsonl


def source_array(C,p,row,guard):
    guard();path=row['path']
    assert path in p['inputs'] and p['inputs'][path]==row['sha256']==sha(path)
    assert row['full'] and row['seed'] is None
    assert row['request']['model']==p['student_start_model']
    assert row['loaded_model_sha256']==p['student_start_sha256']
    assert row['policy_hash']==p['expected_policy_hash']
    a=H.checked_array(C,row)
    assert set(a)==set(row['array_hashes'])
    T.full_policy(C,a,row,dict(p,allowed_teacher_first=[row['request'].get('first',0)]))
    return a


def check_prefix(C,student,actual,first):
    E=C.E;n=len(student['states']);m=len(actual['states'])
    assert 0<=first<n and first<m
    constants=['initial_native_integration','task_goal_xy','scene_sha256']
    history_keys=['history_poses','history_goals','history_times']
    terminal=['final_native_integration','last_hidden_after']
    assert set(actual)==set(student)|{'feedback_progress','feedback_speed'}
    for key in constants:assert E.equal(student[key],actual[key]),key
    aligned=[];histories=[]
    for key,value in student.items():
        if key in constants+terminal:continue
        other=actual[key]
        if key in history_keys:
            assert value.shape[0]==n+1 and other.shape[0]==m+1
            assert E.equal(value[:first+1],other[:first+1]),('history',key)
            histories.append(key)
        else:
            assert value.shape[0]==n and other.shape[0]==m
            assert E.equal(value[:first],other[:first]),('prefix',key)
            aligned.append(key)
    for key in ['states','poses','hidden_before']:
        assert E.equal(student[key][:first+1],actual[key][:first+1]),('handoff',key)
    return sorted(aligned),sorted(histories)


def diagnostic_audit(C,p,out,summary,rows,replays,arrays,guard):
    previous=C.ROOT/'stage/runs/t428_early_recovery/native'
    for name in ['summary.json','audit.json']:
        file=str(previous/name)
        assert file in p['inputs'] and p['inputs'][file]==sha(file)
    prior=C.read(previous/'summary.json');audit=C.read(previous/'audit.json')
    assert prior['state']=='COMPLETE' and audit['state']=='PASS'
    assert prior['all_workers_stopped'] and prior['all_sessions_closed']
    assert audit['native']==audit['new_navigation_training']==0
    prior_diagnostics=prior['result']['diagnostics']
    failed=[item for item in prior_diagnostics if item['recovery_row'] is not None and
            (not item['recovery_row']['success'] or item['recovery_row']['collision'])]
    assert len(failed)==5
    assert sum(bool(item['recovery_row'] is not None and item['recovery_row']['success'] and
                    not item['recovery_row']['collision']) for item in prior_diagnostics)==2
    targets=p['recoveries'];assert len(targets)==5
    identities=[target['job_id'] for target in targets]
    assert identities==[item['id'] for item in failed] and len(set(identities))==5
    gains=p['gains'];assert len(gains)==6
    assert C.np.isfinite(C.np.asarray(gains,dtype=float)).all()
    gain_source=str(C.ROOT/'stage/runs/t422_feedback_recovery_night/search/summary.json')
    assert gain_source in p['inputs'] and p['inputs'][gain_source]==sha(gain_source)
    assert gains==C.read(gain_source)['result']['best']['gains']
    assert p['jobs']==[target['job'] for target in targets]
    assert len(rows)==len(replays)==5
    assert all(row['role']=='FEEDBACK_RECOVERY' and row['full'] for row in rows)
    assert all(row['role']=='FEEDBACK_EXACT' for row in replays)
    assert [row['job_id'] for row in rows]==[row['job_id'] for row in replays]==identities
    result=summary['result'];source=result['student_source']
    assert source==C.initial_record(p)
    assert source==prior['result']['student_source']
    H.snapshot(C.E,source)
    assert result['recoveries']==rows
    checks=result['prefix_checks'];assert len(checks)==5
    assert C.read(out/'prefix_checks.json')==checks
    for target,item,row,repeat,proof in zip(targets,failed,rows,replays,checks):
        guard();identity=target['job_id'];first=target['first']
        assert type(first) is int and first>=0
        assert first==item['trigger']['first']==item['recovery_row']['first']
        assert target['student_row']==item['student_row']
        assert target['failed_mixed_row']==item['recovery_row']
        assert target['job']==item['student_row']['request']['job']
        assert target['reference_path']==item['baseline_row']['path']
        assert target['reference_sha256']==item['baseline_row']['sha256']
        assert item['baseline_row']['success'] and not item['baseline_row']['collision']
        assert not target['failed_mixed_row']['success'] or target['failed_mixed_row']['collision']
        student=source_array(C,p,target['student_row'],guard)
        failed_mixed=source_array(C,p,target['failed_mixed_row'],guard)
        reference=source_array(C,p,item['baseline_row'],guard)
        O.prefix(C,student,failed_mixed,first)
        assert not student['ownership'].any()
        assert reference['end_flags'][-1,1] and not reference['end_flags'][:,0].any()
        assert item['baseline_row']['first']==0
        path=target['reference_path']
        assert path in p['inputs'] and p['inputs'][path]==target['reference_sha256']==sha(path)
        ref=C.R.load_reference(path,target['reference_sha256'])
        assert C.E.equal(ref['goal'],student['task_goal_xy'])
        a=arrays[row['path']]
        for key in ['initial_native_integration','task_goal_xy','scene_sha256']:
            assert C.E.equal(student[key],reference[key]),('reference original task',identity,key)
            assert C.E.equal(student[key],a[key]),('recovery original task',identity,key)
        request=dict(model=source['path'],job=target['job'],mode='feedback',first=first,
            seed=None,reference_path=path,reference_sha256=target['reference_sha256'],gains=gains)
        assert row['request']==request and row['first']==first and row['mode']=='feedback'
        assert row['teacher'] and row['seed'] is None
        assert row['policy_hash']==source['policy_hash'] and row['loaded_model_sha256']==source['sha256']
        assert not a['ownership'][:first].any() and a['ownership'][first:].all()
        assert not a['stochastic'].any()
        assert not any(key.startswith('teacher_') for key in a)
        aligned,histories=check_prefix(C,student,a,first)
        # This pinned independent helper separately implements the feedback
        # algebra and verifies every action, progress index and speed field.
        T.full_policy(C,a,row,p)
        assert proof['id']==identity and proof['first']==first
        assert proof['source_path']==target['student_row']['path']
        assert proof['source_sha256']==target['student_row']['sha256']
        assert proof['source_array_digest']==target['student_row']['array_digest']
        assert proof['reference_path']==path and proof['reference_sha256']==target['reference_sha256']
        for key in ['prefix_exact','handoff_state_exact','handoff_hidden_exact','feedback_recomputed_exact']:
            assert proof[key] is True
        assert sorted(proof['prefix_keys'])==aligned
        assert sorted(proof['history_keys'])==histories
        assert proof['constant_keys']==['initial_native_integration','task_goal_xy','scene_sha256']
        assert proof['native_prefix_steps']==int(a['native_counts'][:first].sum())
        assert proof['speed_packet_column']==56
        assert proof['no_native_state_or_clock_or_history_reset'] is True
        O.replay_equal(repeat,row,p)
    safe=sum(bool(row['success'] and not row['collision']) for row in rows)
    assert result['total_safe']==safe and result['total_failed']==5-safe
    assert result['exact_replays']==5
    assert result['Actor_updates']==result['optimizer_calls']==0
    assert result['gains']==gains and result['training_only_controller_diagnostic'] is True
    assert result['not_navigation_acceptance'] is True
    assert result['six_gate'] is False and result['eight_started'] is False
    return dict(fixed_failed_T428_recoveries=5,previously_recovered_cases_excluded=2,
        actual_student_prefix_and_handoff_hidden_history_exact=True,
        same_original_task_clock_and_native_initial_state_checked=True,
        fixed_single_gain_no_search=True,
        every_actual_feedback_action_progress_and_speed_recomputed=True,
        successful_reference_payloads_authenticated=True,
        no_transplanted_teacher_latent_or_hidden=True,
        all_successes_and_failures_repeated_exact=True,
        exact_replays=5,actual_safe_recoveries=safe,actual_failed_recoveries=5-safe,
        new_navigation_acceptance=False)


def main():
    import t429_core as C
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
        start=time.time();budget=p['stage_budgets']['native']
        deadline=min(start+min(300,budget['audit_seconds']),p['deadline_all'])
        C.E.N.offline();C.torch.set_num_threads(1);C.torch.use_deterministic_algorithms(True)
        def guard():assert time.time()<deadline-5
        def prohibited(*args,**kwargs):raise AssertionError('T429 audit forbids native, optimizer, backward and model export')
        C.E.L.Session=prohibited;C.E.Pool=prohibited;C.Pool=prohibited
        if hasattr(C,'PREVIOUS'):C.PREVIOUS.Pool=prohibited
        C.E.N.S.P.C.B.setup=prohibited
        for optimizer in [C.torch.optim.Adam,C.torch.optim.AdamW,C.torch.optim.SGD]:optimizer.step=prohibited
        C.torch.Tensor.backward=prohibited
        C.E.export=prohibited;C.LEARNING.export_model=prohibited
        rows=jsonl(out/'cases.jsonl');replays=jsonl(out/'replays.jsonl');commands=jsonl(out/'commands.jsonl')
        arrays={}
        for row in rows:
            guard();assert row['full'] and row['pid'] in absent
            assert Path(row['path']).resolve().is_relative_to(out.resolve())
            a=H.checked_array(C,row)
            assert set(a)==set(row['array_hashes'])
            assert row['path'] not in arrays;arrays[row['path']]=a
        for row in replays:
            guard();assert row['pid'] in absent
            assert row['complete'] and not row['administrative_censor']
            assert row['physics']==row['physics_steps']
            assert row['exact'] and row['all_arrays_exact'] and not row['independent_sample']
        details=diagnostic_audit(C,p,out,summary,rows,replays,arrays,guard)
        assert [row['id'] for row in commands]==list(range(1,len(commands)+1))
        assert len(commands)==summary['commands']==14==4+len(rows)+len(replays)<=p['max_commands']
        assert [(row['role'],row['physics']) for row in commands[:4]]==[('SETUP',0)]*4
        expected=[(row['role'],row['physics'],row['pid']) for row in rows+replays]
        assert [(row['role'],row['physics'],row['pid']) for row in commands[4:]]==expected
        assert all(row['pid'] in absent for row in commands)
        assert all(type(row['physics']) is int and row['physics']>=0 for row in commands)
        assert sum(row['physics'] for row in commands)==summary['physics']<=p['max_physics']
        assert p['max_physics']<=360000 and p['max_commands']<=20
        assert summary['elapsed_s']<=p['main_seconds']<=600
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
        print('T429_NATIVE_AUDIT_PASS',flush=True)
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':
    assert len(sys.argv)==1
    main()
