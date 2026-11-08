"""Independent bounded zero-native evidence and Adam audit for Stage T427.

Training-only teachers are replayed on their actual observed histories.  The
new66-parameter optimizer is independently reconstructed; all other weights
must remain exactly fixed.  Import performs no simulation or optimization.
"""
from pathlib import Path
import importlib.util,json,sys,time

ROOT=Path(__file__).resolve().parents[3]
_spec=importlib.util.spec_from_file_location('stage427_original425_audit_helpers',
    ROOT/'stage/experiments/t425_offline_day_chain/audit.py')
T=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(T)
sha,jsonl=T.sha,T.jsonl


def full_policy(C,arrays,row,p):
    E,np,torch=C.E,C.np,C.torch;request=row['request']
    q=C.frozen(p,request['model']).requires_grad_(False)
    assert q.phash()==row['policy_hash']
    initial=C.digest(q.state_dict())
    if request.get('mode','student')!='actor_teacher':
        T.check_full_policy(C,arrays,row,p)
    else:
        first=request['first'];assert first in p['allowed_teacher_first']
        assert row['teacher'] and row['seed'] is None
        assert not arrays['stochastic'].any()
        assert not arrays['ownership'][:first].any() and arrays['ownership'][first:].all()
        E.audit_array(q,arrays,p,first,None)
        source=Path(request['teacher_model'])
        assert str(source)==p['teacher_A_model']
        assert str(source) in p['inputs'] and sha(source)==p['inputs'][str(source)]==request['teacher_sha256']==p['teacher_A_sha256']
        teacher=C.teacher_actor(str(source),p)
        assert teacher.phash()==C.model_decode(str(source))['policy_hash']
        before=C.digest(teacher.state_dict());hidden=None;n=len(arrays['states'])
        assert arrays['teacher_actions'].shape==arrays['teacher_locations'].shape==(n,2)
        assert arrays['teacher_hidden_before'].shape==arrays['teacher_hidden_after'].shape==(n,16)
        with torch.no_grad():
            for index,state in enumerate(arrays['states']):
                prior=np.zeros(16) if hidden is None else hidden[0,0].numpy().copy()
                action,location,hidden,_=teacher.step(torch.tensor(state[None]),hidden)
                assert E.equal(prior,arrays['teacher_hidden_before'][index])
                assert E.equal(hidden[0,0].numpy(),arrays['teacher_hidden_after'][index])
                assert E.equal(action[0].numpy(),arrays['teacher_actions'][index])
                assert E.equal(location[0].numpy(),arrays['teacher_locations'][index])
                if index>=first:assert E.equal(action[0].numpy(),arrays['actions'][index])
        assert C.digest(teacher.state_dict())==before
        assert np.all(arrays['feedback_progress']==-1) and np.all(arrays['feedback_speed']==0)
    assert C.digest(q.state_dict())==initial


def data_audit(C,p,out,summary,rows,replays,guard):
    E=C.E;result=summary['result'];source=result['source_model']
    assert sha(source['path'])==source['sha256']==p['student_start_sha256']
    assert source['policy_hash']==p['expected_policy_hash'];T.snapshot(E,source)
    anchors=C.read(out/'anchors.json');targets=C.read(out/'targets.json')
    assert anchors==result['anchors'] and targets==result['targets']
    initial=[row for row in rows if row['role']=='ANCHOR57']
    assert len(initial)==57 and [r['job_id'] for r in initial]==[j['id'] for j in p['anchor_jobs']]
    assert len(anchors)==57 and [entry['row'] for entry in anchors]==initial
    proof={row['job_id']:row for row in p['anchor_proofs']};assert len(proof)==57
    for entry,row in zip(anchors,initial):
        guard();assert entry['path']==row['path'] and entry['sha256']==row['sha256']
        assert row['full'] and not row['teacher'] and row['mode']=='student' and row['seed'] is None
        assert row['loaded_model_sha256']==source['sha256'] and row['policy_hash']==source['policy_hash']
        assert row['success'] and not row['collision'] and not row['timeout']
        for key in ['array_digest','array_hashes','integration_sha256','physics_steps','success','collision','timeout']:
            assert row[key]==proof[row['job_id']][key],(row['job_id'],key)
    selected={entry['id']:entry for entry in targets};assert len(selected)==len(targets)
    expected=list(initial);expected_replays=[];six_ids={j['id'] for j in p['main_jobs']}
    assert p['allowed_teacher_first']==[1,0]
    for job in p['recovery_jobs']:
        identity=job['id'];success=None;mode='actor_teacher' if identity in six_ids else 'safe_reset'
        for first in p['allowed_teacher_first']:
            values=[row for row in rows if row['role']==f'QUERY_{identity}_{first}']
            assert len(values)==1
            row=values[0];expected.append(row);request=row['request']
            assert request['model']==source['path'] and request['job']==job
            assert request['mode']==mode and request['first']==first and request['seed'] is None
            assert row['teacher'] and not row['full'] and row['mode']==mode
            assert row['policy_hash']==source['policy_hash'] and row['loaded_model_sha256']==source['sha256']
            if mode=='actor_teacher':
                assert request['teacher_model']==p['teacher_A_model']
                assert sha(request['teacher_model'])==p['inputs'][request['teacher_model']]==request['teacher_sha256']==p['teacher_A_sha256']
            if row['success'] and not row['collision']:
                success=row
                for later in p['allowed_teacher_first'][p['allowed_teacher_first'].index(first)+1:]:
                    assert not any(other['role']==f'QUERY_{identity}_{later}' for other in rows)
                break
        if success is None:
            assert identity not in selected
            assert not any(row['role']=='EXACT_'+identity for row in replays)
            continue
        entry=selected[identity];upgraded=dict(success,path=entry['path'],sha256=entry['sha256'],full=True)
        assert entry['row']==upgraded and entry['request']==success['request']
        assert entry['first']==success['request']['first'] and entry['exact']
        guard();arrays=T.checked_array(C,upgraded);full_policy(C,arrays,upgraded,p)
        assert arrays['end_flags'][-1,1] and not arrays['end_flags'][:,0].any()
        repeated=[r for r in replays if r['role']=='EXACT_'+identity]
        assert len(repeated)==1;repeat=repeated[0];expected_replays.append(repeat)
        assert repeat['request']==entry['request'] and repeat['reference_path']==entry['path']
        assert repeat['exact'] and repeat['all_arrays_exact'] and not repeat['independent_sample']
        for key in ['array_digest','array_hashes','integration_sha256','physics_steps','success','collision','timeout','policy_hash']:
            assert repeat[key]==success[key],(identity,key)
    assert rows==expected and replays==expected_replays
    assert set(selected)<={j['id'] for j in p['recovery_jobs']} and len(p['recovery_jobs'])==7
    assert result['teacher_targets_complete']==result['learning_data_available']==(len(targets)==7)
    assert result['current_anchors_rebuilt_exact'] is True
    assert result['historical_anchor_executions_reconstructed']==57
    return dict(historical_discarded_Adam=0,current120_full57_array_hashes_exact=True,
        reconstructed_historical_anchor_executions=57,independent_new_anchor_samples=0,
        verified_actual_safe_teacher_targets=len(targets),real_first1_then_t0_fallback_checked=True,
        neural_A_teacher_actual_observations_and_hidden_reconstructed=True,
        every_selected_teacher_control_and_exact_native_repeat_verified=True,
        teacher_recovery_independence_not_asserted=True,
        every_native_execution_charged_even_if_historical_behavior_repeats=True)


def learn_audit(C,p,out,summary,rows,replays,guard):
    E,L=C.E,C.LEARNING;result=summary['result'];dataset=C.read(C.OUT/'data/summary.json')['result']
    assert C.read(C.OUT/'data/audit.json')['state']=='PASS' and dataset['teacher_targets_complete']
    assert not replays
    q=C.frozen(p,result['initial_model']['path']);opt=L.make_optimizer(q,p)
    T.check_model(E,q,opt,result['initial_model']);fixed=L.fixedhash(q)
    assert not opt.state and q.phash()==p['expected_policy_hash']
    assert sum(v.numel() for v in L.params(q))==66
    assert result['initial_model']==result['before_model']
    anchors=[(E.N.S.arrays(entry['path']),entry['row']) for entry in dataset['anchors']]
    targets=[dict(arrays=E.N.S.arrays(entry['path']),first=entry['first'],id=entry['id']) for entry in dataset['targets']]
    groups=L.groups(q,p,anchors,targets)
    events=jsonl(out/'optimizer_calls.jsonl');assert len(events)==480
    tags={row['tag'] for row in events};assert len(tags)==1;tag=next(iter(tags))
    assert tag=='fit120'
    fit=lambda event:L.fit(q,opt,groups,p,guard,event,steps=120)
    rebuilt=T.reconstruct_pair(E,q,opt,fit,result['fit'],guard,events,tag,C.gradient_sha)
    assert events==rebuilt['events'];T.check_model(E,q,opt,result['proposed_model'])
    assert L.fixedhash(q)==fixed
    check=C.read(result['check_file'])
    assert len(rows)==329 and len({row['role'] for row in rows})==1
    role=rows[0]['role']
    assert role=='PROPOSAL120'
    ids,actual=T.complete_development(C,p,check,rows,role,result['proposed_model'])
    assert actual==rows
    fresh_ids={job['id'] for job in p['fresh_jobs']}
    reference_ids={job['id'] for job in p['reload_jobs']}-fresh_ids
    assert len(reference_ids)==62
    assert {row['job_id'] for row in rows if row['full']}==reference_ids
    initial=set(p['initial_success_ids']);assert len(initial)==322
    accepted=initial<=ids and bool(ids-initial)
    assert result['accepted']==accepted
    chosen=result['proposed_model'] if accepted else result['initial_model']
    E.L.load(q,opt,C.model_decode(chosen['path'])['state'])
    T.check_model(E,q,opt,result['model']);assert L.fixedhash(q)==fixed
    selected=ids if accepted else initial
    assert result['selected_success_ids']==sorted(selected)
    assert result['updates']==dict(live=120,shadow=120)
    assert result['development_all_safe']==(accepted and check['all_safe'])
    assert result['development_file']==(result['check_file'] if accepted else None)
    assert result['frozen_representation_all_but66_verified'] is True
    assert result['full62_references_saved_during_development'] is True
    return dict(historical_discarded_Adam=240,actual_recorded_live_steps=120,
        actual_recorded_shadow_steps=120,every_Actor_Adam_and_clipped_gradient_event_exact=True,
        fixed_representation_and_all_but66_readout_parameters_exact=True,
        actual57_anchor_histories_and7_teacher_masks_checked=True,
        complete329_closed_loop_acceptance_and_final_Actor_Adam_selection_checked=True,
        full62_reference_payloads_saved_during_same329_evaluation=True,
        final_development_all_safe=result['development_all_safe'])


def gate_audit(C,p,out,summary,rows,replays,guard):
    result=summary['result'];model=result['model'];assert sha(model['path'])==model['sha256']
    assert C.read(C.OUT/'learn/audit.json')['state']=='PASS'
    prior=C.read(C.OUT/'learn/summary.json')['result']
    assert prior['development_all_safe'] and prior['model']['policy_hash']==model['policy_hash']
    development=C.read(result['development_file'])
    assert development['all_safe'] and len(development['success_ids'])==329
    assert development['policy_hash']==model['policy_hash']
    fresh_ids={job['id'] for job in p['fresh_jobs']}
    reference_ids={job['id'] for job in p['reload_jobs']}-fresh_ids
    references={row['job_id']:row for row in development['rows'] if row['job_id'] in reference_ids}
    assert len(references)==62
    for row in references.values():
        guard();assert row['full'] and row['success'] and not row['collision']
        assert row['loaded_model_sha256']==model['sha256'] and row['policy_hash']==model['policy_hash']
        arrays=T.checked_array(C,row);full_policy(C,arrays,row,p)
    assert sorted(result['reference62_paths'])==sorted(row['path'] for row in references.values())
    for file,value in p['fresh_ledger'].items():assert sha(file)==value
    assert len(rows)==16 and [row['job_id'] for row in rows]==[job['id'] for job in p['fresh_jobs']]
    assert all(row['role']=='FRESH16' and row['full'] and not row['teacher'] and row['seed'] is None
               and row['loaded_model_sha256']==model['sha256'] and row['policy_hash']==model['policy_hash'] for row in rows)
    assert result['fresh16']==C.E.tally(rows)
    passed=all(row['success'] and not row['collision'] for row in rows)
    references.update({row['job_id']:row for row in rows})
    if passed:
        assert len(replays)==78 and [row['job_id'] for row in replays]==[job['id'] for job in p['reload_jobs']]
        for row in replays:
            guard();old=references[row['job_id']]
            assert row['role']=='RELOAD78' and row['reference_path']==old['path']
            assert row['loaded_model_sha256']==model['sha256'] and not row['teacher'] and row['seed'] is None
            assert row['exact'] and row['all_arrays_exact'] and not row['independent_sample']
            for key in ['array_digest','array_hashes','integration_sha256','physics_steps','success','collision','timeout','policy_hash']:
                assert row[key]==old[key],(row['job_id'],key)
        assert result['reload78']==dict(exact=78,all_arrays_exact=True,independent_sample=False)
    else:assert not replays and result['reload78'] is None
    assert result['six_gate']==passed
    return dict(historical_discarded_Adam=0,six_gate=passed,
        complete329_predecessor_proof_checked=True,full62_reference_payloads_reused_without_new_physics=True,
        heldout_navigation_cases=16,independent_native_reloads=len(replays),
        independent_native_reload_every_recorded_array_hash_exact=True)


def main(name):
    import t427_core as C
    assert name in C.ALIASES
    p=C.read(C.OUT/'protocol.json');C.verify(p);out=C.OUT/name;summary=C.read(out/'summary.json')
    assert summary['state']=='COMPLETE' and summary['all_workers_stopped'] and summary['all_sessions_closed']
    assert not (out/'audit.json').exists()
    absent=T.stopped(out);members=T.manifest(C.ROOT,out)
    assert not list(out.rglob('partial*.npz'))
    start=time.time();budget=p['stage_budgets'][name]
    C.E.N.offline();C.torch.set_num_threads(1);C.torch.use_deterministic_algorithms(True)
    def guard():assert time.time()<min(start+budget['audit_seconds']-5,p['deadline_all']-5)
    def no_native(*args,**kwargs):raise AssertionError('Independent audit cannot launch native simulation')
    C.E.L.Session=no_native;C.E.Pool=no_native;C.Pool=no_native
    actual_Adam=[0];original_Adam=C.torch.optim.Adam.step
    def counted_Adam(optimizer,*args,**kwargs):
        guard();assert name=='learn' and actual_Adam[0]<240
        value=original_Adam(optimizer,*args,**kwargs);actual_Adam[0]+=1;return value
    def no_other_optimizer(*args,**kwargs):raise AssertionError('Only bounded historical Adam is allowed')
    C.torch.optim.Adam.step=counted_Adam
    C.torch.optim.AdamW.step=no_other_optimizer;C.torch.optim.SGD.step=no_other_optimizer
    C.E.export=no_other_optimizer;C.LEARNING.export_model=no_other_optimizer
    rows=jsonl(out/'cases.jsonl');replays=jsonl(out/'replays.jsonl');commands=jsonl(out/'commands.jsonl')
    for row in rows:
        guard();arrays=T.checked_array(C,row)
        if row['full']:full_policy(C,arrays,row,p)
        assert row['pid'] in absent
    for row in replays:
        assert row['physics']==row['physics_steps'] and row['pid'] in absent
        assert row['complete'] and not row['administrative_censor']
        assert row['exact'] and row['all_arrays_exact'] and not row['independent_sample']
    if name=='data':details=data_audit(C,p,out,summary,rows,replays,guard)
    elif name=='learn':details=learn_audit(C,p,out,summary,rows,replays,guard)
    elif name=='gate':details=gate_audit(C,p,out,summary,rows,replays,guard)
    else:details=T.audit_eight(C,p,out,summary,rows,replays,guard)
    assert actual_Adam[0]==details['historical_discarded_Adam']
    assert [row['id'] for row in commands]==list(range(1,len(commands)+1))
    assert len(commands)==summary['commands']==4+len(rows)+len(replays)<=budget['max_commands']
    assert [(row['role'],row['physics']) for row in commands[:4]]==[('SETUP',0)]*4
    expected=[(row['role'],row['physics']) for row in rows+replays]
    actual=[(row['role'],row['physics']) for row in commands[4:]]
    assert sorted(actual)==sorted(expected)
    for role in {entry[0] for entry in expected}:
        assert [entry for entry in actual if entry[0]==role]==[entry for entry in expected if entry[0]==role]
    assert sum(row['physics'] for row in commands)==summary['physics']<=budget['max_physics']
    assert summary['elapsed_s']<=budget['main_seconds'];guard();C.verify(p)
    C.save(out/'audit.json',dict(state='PASS',stage=name,native=0,new_navigation_training=0,
        elapsed_s=time.time()-start,native_processes_absent=absent,
        manifest_members_before_audit=members,physics=summary['physics'],commands=summary['commands'],
        raw_controls_counts_terminal_and_command_ledger_exact=True,
        actual_full_payload_history_hidden_actions_rewards_checked=True,
        compact_evidence_limitation='Compact development/query cases retain actions/counts/flags and original full array hashes; zero-native audit does not reconstruct absent full120 observations.',
        original_T422_history_audit_not_repaired=True,**details))
    C.E.N.manifest(out);print('T427_STAGE_AUDIT_PASS',name,flush=True)


if __name__=='__main__':
    assert len(sys.argv)==2
    main(sys.argv[1])
