"""Independent frozen-policy/history/ownership audit, zero native and Adam."""
from t411_core import *

def jsonl(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x] if Path(path).exists() else []

def main():
    N.offline();p=plan();base=verify(p);s=read(OUT/'summary.json');assert not (OUT/'audit.json').exists() and not Path('/proc',str(s['pid'])).exists()
    for row in read(OUT/'local_manifest.json')['files']:assert sha(ROOT/row['path'])==row['sha256']
    rows=jsonl(OUT/'cases.jsonl');reps=jsonl(OUT/'replays.jsonl');commands=jsonl(OUT/'commands.jsonl')
    partial=None
    if (OUT/'partial.npz').exists():
        a=N.S.arrays(OUT/'partial.npz');partial=dict(path=str(OUT/'partial.npz'),sha256=sha(OUT/'partial.npz'),recorded_physics=int(a['native_counts'].sum()),administrative_censor=True)
    if s['state']!='COMPLETE':
        save(OUT/'audit.json',dict(state='FAILED_PRODUCER_PARTIAL_PRESERVED',native=0,Adam=0,partial=partial,recorded_cases=len(rows),recorded_replays=len(reps),physics=s['physics'],commands=s['commands']));N.manifest(OUT);raise AssertionError('No PASS for incomplete producer')
    assert partial is None and len(rows)==len(reps)==len(p['cases'])
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);student,teacher=actors(base,p);install_no_learning();fixed_student=student.phash();fixed_teacher=teacher.phash()
    for case,row,repeat in zip(p['cases'],rows,reps):
        assert row['case_id']==repeat['case_id']==case['id'] and row['first']==case['first']
        assert row['role']==case['id'] and repeat['role']==case['id']+'_exact'
        assert sha(row['path'])==row['sha256'];a=N.S.arrays(row['path']);assert digest(a)==row['array_digest']
        assert digest(a['final_native_integration'])==row['integration_sha256']
        assert int(a['native_counts'].sum())==row['physics_steps']==row['physics']
        assert bool(a['end_flags'][-1,1])==row['success'] and bool(a['end_flags'][:,0].any())==row['collision'] and bool(a['end_flags'][-1,2])==row['timeout']
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].any()
        assert row['student_policy_hash']==fixed_student and row['teacher_policy_hash']==fixed_teacher
        audit_arrays(student,teacher,a,case,p)
        for key in ['array_digest','integration_sha256','physics_steps','success','collision','timeout','student_policy_hash','teacher_policy_hash']:assert row[key]==repeat[key]
        assert repeat['exact'] is True and repeat['independent_sample'] is False
    assert student.phash()==fixed_student and teacher.phash()==fixed_teacher
    assert s['Adam']==s['backward']==s['Actor_updates']==0 and s['no_six_metre_acceptance'] and not s['seven_eight_entered']
    assert len(commands)==s['commands']==1+2*len(rows)
    assert [row['id'] for row in commands]==list(range(1,len(commands)+1))
    assert sum(row['physics'] for row in commands)==sum(row['physics_steps'] for row in rows+reps)==s['physics']
    assert not any(row['error'] for row in commands)
    assert s['physics']<=p['max_physics'] and s['commands']<=p['max_commands'] and s['elapsed_s']<=p['main_seconds']
    assert sha(p['initial_model'])==s['student_model_sha256'] and sha(p['teacher_model'])==s['teacher_model_sha256']
    save(OUT/'audit.json',dict(state='PASS',native=0,Adam=0,backward=0,Actor_updates=0,student_and_teacher_full_current_history_exact=True,teacher_hidden_prefix_reconstruction_and_no_handoff_reset=True,teacher_suffix_actions_closed_loop_exact=True,ownership_and_original_clock_reward_fullvehicle_exact=True,source_control_shared_physics_exact=True,all_raw_SHA_and_replays_exact=True,physics=s['physics'],commands=s['commands'],unique_conditions=len(rows),exact_native_replays=len(reps),student_policy_hash=fixed_student,teacher_policy_hash=fixed_teacher,no_six_metre_acceptance=True,seven_eight_entered=False));N.manifest(OUT);print('T411_INDEPENDENT_AUDIT_PASS',flush=True)

if __name__=='__main__':main()
