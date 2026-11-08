"""Independent zero-native evidence and actual Adam replay for Stage T426.

The original T425 validators are immutable inputs. New code below validates
the changed dataset, fixed two-checkpoint trajectory and final model choice.
Import is inert; invocation never launches a simulator or retains a model.
"""
from pathlib import Path
import importlib.util,json,sys,time

ROOT=Path(__file__).resolve().parents[3]
_spec=importlib.util.spec_from_file_location('stage426_original425_audit_helpers',
    ROOT/'stage/experiments/t425_offline_day_chain/audit.py')
T=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(T)
sha,jsonl=T.sha,T.jsonl


def data_audit(C,p,out,summary,rows,replays,guard):
    E=C.E;result=summary['result'];source=result['source_model']
    assert sha(source['path'])==source['sha256']==p['student_start_sha256']
    assert source['policy_hash']==p['expected_policy_hash']
    T.snapshot(E,source)
    anchors=C.read(out/'anchors.json');targets=C.read(out/'targets.json')
    assert anchors==result['anchors'] and targets==result['targets']
    initial=[row for row in rows if row['role']=='ANCHOR61']
    assert len(initial)==61 and [row['job_id'] for row in initial]==[j['id'] for j in p['anchor_jobs']]
    assert len(anchors)==61 and [entry['row'] for entry in anchors]==initial
    proof={row['job_id']:row for row in p['anchor_proofs']}
    assert len(proof)==61
    for entry,row in zip(anchors,initial):
        guard();assert entry['path']==row['path'] and entry['sha256']==row['sha256']
        assert row['full'] and not row['teacher'] and row['mode']=='student' and row['seed'] is None
        assert row['loaded_model_sha256']==source['sha256'] and row['policy_hash']==source['policy_hash']
        assert row['success'] and not row['collision'] and not row['timeout']
        prior=proof[row['job_id']]
        for key in ['array_digest','array_hashes','integration_sha256','physics_steps','success','collision','timeout']:
            assert row[key]==prior[key],(row['job_id'],key)
    selected={entry['id']:entry for entry in targets};assert len(selected)==len(targets)
    expected=list(initial);expected_replays=[]
    assert p['allowed_teacher_first']==[1,0]
    for job in p['recovery_jobs']:
        identity=job['id'];success=None
        for first in p['allowed_teacher_first']:
            role=f'QUERY_{identity}_{first}'
            values=[row for row in rows if row['role']==role]
            assert len(values)==1
            row=values[0];expected.append(row);request=row['request']
            assert request==dict(model=source['path'],job=job,mode='safe_reset',first=first,seed=None)
            assert row['teacher'] and not row['full'] and row['mode']=='safe_reset'
            assert row['policy_hash']==source['policy_hash'] and row['loaded_model_sha256']==source['sha256']
            if row['success'] and not row['collision']:
                success=row
                # No additional attempt is allowed once the real recovery succeeds.
                for later in p['allowed_teacher_first'][p['allowed_teacher_first'].index(first)+1:]:
                    assert not any(other['role']==f'QUERY_{identity}_{later}' for other in rows)
                break
        if success is None:
            assert identity not in selected
            assert not any(row['role']=='EXACT_'+identity for row in replays)
            continue
        entry=selected[identity]
        upgraded=dict(success,path=entry['path'],sha256=entry['sha256'],full=True)
        assert entry['row']==upgraded and entry['request']==success['request']
        assert entry['first']==success['request']['first'] and entry['exact']
        guard();arrays=T.checked_array(C,upgraded);T.check_full_policy(C,arrays,upgraded,p)
        assert arrays['end_flags'][-1,1] and not arrays['end_flags'][:,0].any()
        repeated=[row for row in replays if row['role']=='EXACT_'+identity]
        assert len(repeated)==1;repeat=repeated[0];expected_replays.append(repeat)
        assert repeat['request']==entry['request'] and repeat['reference_path']==entry['path']
        assert repeat['exact'] and repeat['all_arrays_exact'] and not repeat['independent_sample']
        for key in ['array_digest','array_hashes','integration_sha256','physics_steps','success','collision','timeout','policy_hash']:
            assert repeat[key]==success[key],(identity,key)
    assert rows==expected and replays==expected_replays
    assert set(selected)<={job['id'] for job in p['recovery_jobs']}
    assert len(p['recovery_jobs'])==11
    assert result['teacher_targets_complete']==result['learning_data_available']==(len(targets)==11)
    assert result['current_A_anchors_rebuilt_exact'] is True
    assert result['historical_anchor_executions_reconstructed']==61
    assert result['independent_new_anchor_samples']==0
    return dict(historical_discarded_Adam=0,current_A_full61_array_hashes_exact=True,
        reconstructed_historical_anchor_executions=61,independent_new_anchor_samples=0,
        verified_actual_safe_teacher_targets=len(targets),real_first1_then_t0_fallback_checked=True,
        every_selected_teacher_control_and_exact_native_repeat_verified=True,
        teacher_recovery_independence_not_asserted=True,
        every_native_execution_charged_even_if_historical_behavior_repeats=True)


def learn_audit(C,p,out,summary,rows,replays,guard):
    E,L=C.E,C.LEARNING;result=summary['result'];dataset=C.read(C.OUT/'data/summary.json')['result']
    assert C.read(C.OUT/'data/audit.json')['state']=='PASS' and dataset['teacher_targets_complete']
    assert not replays
    q=C.frozen(p,result['initial_model']['path']);opt=L.make_optimizer(q,p)
    T.check_model(E,q,opt,result['initial_model']);fixed=E.original_hash(q)
    assert not opt.state and q.phash()==p['expected_policy_hash']
    anchors=[(E.N.S.arrays(entry['path']),entry['row']) for entry in dataset['anchors']]
    targets=[dict(arrays=E.N.S.arrays(entry['path']),first=entry['first'],id=entry['id']) for entry in dataset['targets']]
    g=L.groups(q,p,anchors,targets)
    assert len(g['six'])==54 and len(g['old_centres'])==7 and len(g['corrections'])==11
    assert g['no_legacy_left_right_query_auxiliary'] is True
    performance=C.read(out/'performance.json');assert performance==result['performance']
    assert performance['new_Adam']==performance['native']==0
    duration=performance['elapsed_s'];assert 0<duration<p['stage_budgets']['learn']['main_seconds']
    assert performance['projected_240_updates_seconds']==duration*240
    assert performance['audit_estimate_s']==duration*240+120
    assert performance['main_estimate_s']==duration*240+2150
    admitted=(performance['audit_estimate_s']<=p['stage_budgets']['learn']['audit_seconds']
              and performance['main_estimate_s']<=p['stage_budgets']['learn']['main_seconds'])
    assert performance['admitted']==admitted
    # Reconstruct the production-data gradient, without any Adam step.
    # Wall-clock timings are checked arithmetically, not expected to repeat.
    weights=C.digest(q.state_dict());variables=E.params(q);opt.zero_grad(set_to_none=True)
    losses=L.support_losses(q,g,p,0)
    reference=E.vector_gradient(losses[0]+losses[1],variables)
    broad=E.protect_vector(E.vector_gradient(losses[2],variables),reference)
    correction=E.protect_vector(E.vector_gradient(losses[3],variables),reference)
    E.install_vector(variables,reference+broad+correction)
    C.torch.nn.utils.clip_grad_norm_(variables,1.,error_if_nonfinite=True)
    assert weights==C.digest(q.state_dict())==performance['before_weights']==performance['after_weights']
    assert C.gradient_sha(q)==performance['gradient_sha256'];guard()
    events=jsonl(out/'optimizer_calls.jsonl');records=result['records'];expected_rows=[];expected_events=[]
    assert len(records)<=2 and p['maximum_steps']==120
    if not admitted:assert not records and not events and not rows
    if records:assert C.read(out/'records.json')==records
    else:assert not (out/'records.json').exists()
    initial=set(p['initial_success_ids']);assert len(initial)==318
    selected=result['initial_model'];success=set(initial);bestfile=None;all_safe=False;steps=0
    for index,record in enumerate(records):
        guard();assert record['start']==index*60 and record['count']==60
        assert C.read(out/'blocks'/str(index)/'result.json')==record
        T.check_model(E,q,opt,record['before_model'])
        tag='block'+str(index);tag_events=[row for row in events if row['tag']==tag]
        assert len(tag_events)==240
        run=lambda event:L.stepfit(q,opt,g,p,index*60,60,guard,event)
        rebuilt=T.reconstruct_pair(E,q,opt,run,record['fit'],guard,tag_events,tag,C.gradient_sha)
        expected_events.extend(rebuilt['events']);steps+=60
        T.check_model(E,q,opt,record['proposed_model']);assert E.original_hash(q)==fixed
        check=C.read(record['development_file'])
        assert {key:value for key,value in check.items() if key!='rows'}==record['development']
        ids,actual=T.complete_development(C,p,check,rows,'CHECK_'+str(index*60+60),record['proposed_model'])
        expected_rows.extend(actual)
        eligible=initial<=ids and bool(ids-initial)
        assert record['accepted']==eligible
        if eligible and len(ids)>len(success):
            selected=record['proposed_model'];success=ids
            bestfile=record['development_file'];all_safe=check['all_safe']
        if all_safe:assert index==len(records)-1
        # The next checkpoint continues this research Actor+Adam even when
        # this checkpoint is ineligible; final selection is checked below.
    assert events==expected_events and rows==expected_rows
    E.L.load(q,opt,C.model_decode(selected['path'])['state'])
    T.check_model(E,q,opt,result['model']);assert E.original_hash(q)==fixed
    assert result['updates']==dict(live=steps,shadow=steps)
    assert result['selected_success_ids']==sorted(success)
    assert result['development_all_safe']==all_safe and result['development_file']==bestfile
    assert result['continuous_research_trajectory_not_checkpoint_rollback'] is True
    assert result['final_exact_Actor_Adam_selected_snapshot'] is True
    assert result['old_T422_audit_NOT_repaired'] is True
    return dict(historical_discarded_Adam=2*steps,actual_recorded_live_steps=steps,
        actual_recorded_shadow_steps=steps,every_Actor_Adam_and_clipped_gradient_event_exact=True,
        fixed_starting61_history_targets_and_real11_teacher_masks_checked=True,
        all_two_checkpoint_full329_results_and_final_eligible_selection_checked=True,
        no_teacher_latent_or_onpolicy_density_invented=True,final_development_all_safe=all_safe,
        production_profile_gradient_reconstructed_exact_without_Adam=True,
        independent_zero_Adam_profile_gradients=1,production_learning_admitted=admitted)


def main(name):
    import t426_core as C
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
    def no_other_optimizer(*args,**kwargs):raise AssertionError('Only bounded historical Adam reconstruction is allowed')
    C.torch.optim.Adam.step=counted_Adam
    C.torch.optim.AdamW.step=no_other_optimizer;C.torch.optim.SGD.step=no_other_optimizer
    C.E.export=no_other_optimizer;C.LEARNING.export_model=no_other_optimizer
    rows=jsonl(out/'cases.jsonl');replays=jsonl(out/'replays.jsonl');commands=jsonl(out/'commands.jsonl')
    for row in rows:
        guard();arrays=T.checked_array(C,row)
        if row['full']:T.check_full_policy(C,arrays,row,p)
        assert row['pid'] in absent
    for row in replays:
        assert row['physics']==row['physics_steps'] and row['pid'] in absent
        assert row['complete'] and not row['administrative_censor']
        assert row['exact'] and row['all_arrays_exact'] and not row['independent_sample']
    if name=='data':details=data_audit(C,p,out,summary,rows,replays,guard)
    elif name=='learn':details=learn_audit(C,p,out,summary,rows,replays,guard)
    elif name=='gate':
        assert C.read(C.OUT/'learn/audit.json')['state']=='PASS'
        details=T.audit_gate(C,p,out,summary,rows,replays,guard)
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
        compact_evidence_limitation='Compact development/query cases retain actions/counts/flags and original full array hashes; zero-native audit does not reconstruct missing full120 observations.',
        original_T422_history_audit_not_repaired=True,**details))
    C.E.N.manifest(out);print('T426_STAGE_AUDIT_PASS',name,flush=True)


if __name__=='__main__':
    assert len(sys.argv)==2
    main(sys.argv[1])
