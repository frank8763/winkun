"""Discarded historical reconstruction after the offline producer has exited.

No simulator is advanced. Rejections and the finite correction warmups are
reproduced from original stage evidence; they are not new navigation training.
"""
from day_core import *
import importlib.util
_SPEC=importlib.util.spec_from_file_location('stage410_actual_producer',HERE/'run.py')
RUN=importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(RUN)


def jsonl(path):
    path=Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def model_readback(q,opt,record):
    assert sha(record['path'])==record['sha256']
    stored=torch.load(io.BytesIO(gzip.decompress(Path(record['path']).read_bytes())),weights_only=False,map_location='cpu')
    actual=L.statehash(q,opt)
    assert stored['hashes']==record['hashes']==actual
    assert {key:digest(value) for key,value in stored['state'].items()}==actual
    assert stored['policy_hash']==record['policy_hash']==q.phash()
    assert stored['deployed_actor_count']==1 and stored['trainable_parameters']==4450
    assert stored['architecture']=='frozen_T404_plus_136x32x2_zero_initialized_logit_residual'


def joint_check(rows,role,q,p,record):
    jobs=[*p['old8_jobs'],*p['left5_jobs'],p['right_job']]
    actual=[row for row in rows if row['role'].startswith(role+'_')]
    assert [row['job_id'] for row in actual]==[job['id'] for job in jobs]
    assert len(actual)==14 and all(not row['teacher'] and row['seed'] is None for row in actual)
    assert all(row['policy_hash']==q.phash() for row in actual)
    ids=sorted(row['job_id'] for row in actual if row['success'])
    expected=dict(policy_hash=q.phash(),success_ids=ids,all14=len(ids)==14,
                  old8=tally(actual[:8]),left5=tally(actual[8:13]),right1=tally(actual[13:]))
    assert record==expected
    return ids


def raw_ledger(p,summary):
    rows=jsonl(OUT/'cases.jsonl');replays=jsonl(OUT/'replays.jsonl');commands=jsonl(OUT/'commands.jsonl')
    for row in rows:
        assert sha(row['path'])==row['sha256']
        arrays=N.S.arrays(row['path'])
        assert int(arrays['native_counts'].sum())==row['physics_steps']==row['physics']
        assert bool(arrays['end_flags'][-1,1])==row['success']
        assert bool(arrays['end_flags'][:,0].any())==row['collision']
        assert bool(arrays['end_flags'][-1,2])==row['timeout']
        assert arrays['end_flags'][-1].any() and not arrays['end_flags'][:-1].any()
        assert not (row['success'] and row['collision'])
        if row['full']:
            assert digest(arrays)==row['array_digest']
            assert digest(arrays['final_native_integration'])==row['integration_sha256']
            assert equal(arrays['states'][1:],arrays['next_states'][:-1])
    partial=[]
    for path in sorted(OUT.rglob('partial.npz')):
        record=dict(path=str(path),sha256=sha(path),administrative_censor=True)
        try:record['recorded_physics']=int(N.S.arrays(path)['native_counts'].sum())
        except BaseException:record.update(recorded_physics=None,parse_error=traceback.format_exc())
        partial.append(record)
    for row in replays:
        role=row['role']
        if role.startswith('train_') and '_exact_' in role:
            candidates=[value for value in rows if value['role']==role.replace('_exact_','_')]
        elif role.startswith('teacher_') and '_exact_' in role:
            candidates=[value for value in rows if value['role']==role.replace('_exact_','_')]
        elif role.startswith('reload78_'):
            candidates=[value for value in rows if value['job_id']==row['job_id'] and
                        any(value['role'].startswith(prefix) for prefix in ['main54_','old275_','fresh16_'])]
        else:raise AssertionError(('Undeclared replay role',role))
        assert len(candidates)==1
        reference=candidates[0]
        assert row['exact'] is True and row['independent_sample'] is False
        for key in ['job_id','array_digest','integration_sha256','physics_steps','success','collision','timeout','policy_hash','seed','teacher','force_first']:
            assert row[key]==reference[key],(role,key)
    recorded=sum(row['physics_steps'] for row in rows+replays)
    ledger=dict(cases=len(rows),exact_native_replays=len(replays),commands=len(commands),
                recorded_complete_physics=recorded,summary_physics=summary['physics'],
                unledgered_physics=summary['physics']-recorded,
                unledgered_commands=summary['commands']-len(commands),partials=partial)
    if summary['state']=='COMPLETE':
        assert not partial and len(commands)==summary['commands']==4+len(rows)+len(replays)
        assert [row['id'] for row in commands]==list(range(1,len(commands)+1))
        assert sum(row['physics'] for row in commands)==summary['physics']==recorded
        assert not any(row['error'] for row in commands)
        assert summary['commands']<=p['max_commands'] and summary['physics']<=p['max_physics']
        assert summary['elapsed_s']<=p['main_seconds']
    return rows,replays,commands,ledger


def optimizer_events(tag,count):
    result=[]
    for which in ['live','shadow']:
        for step in range(count):result.extend([[tag,which,'BEFORE',step],[tag,which,'AFTER',step]])
    return result


def training_batch(rows,q,p,rnd,job):
    arrays=[];actual=[]
    for index in range(4):
        row=next(value for value in rows if value['role']==f'train_{rnd}_{index}')
        assert row['job_id']==job['id'] and row['full'] and not row['teacher']
        assert row['seed']==[p['seed'],rnd,index] and row['policy_hash']==q.phash()
        value=N.S.arrays(row['path']);audit_array(q,value,p,0,row['seed'])
        assert not value['ownership'].any()
        arrays.append(value);actual.append(row)
    return arrays,tally(actual)


def controls_for(reference,variant):
    indices=np.minimum(np.arange(400)+variant,len(reference['actions'])-1)
    return reference['actions'][indices].copy()


def teacher_evidence(rows,replays,q,p,rnd,job,index,block):
    queries=block['teacher_queries'];variants=p['teacher_advance_holds']
    assert 1<=len(queries)<=len(variants)
    assert [query['variant'] for query in queries]==variants[:len(queries)]
    reference=N.S.arrays(p['left_history_files'][index]);chosen=None
    for position,query in enumerate(queries):
        role=f'teacher_{rnd}_{query["variant"]}_{job["id"]}'
        row=next(value for value in rows if value['role']==role)
        assert row==query['row'] and row['job_id']==job['id'] and row['teacher']
        assert row['seed'] is None and row['force_first']==p['teacher_first']
        assert row['policy_hash']==q.phash()
        if row['success']:
            assert position==len(queries)-1 and row['full'] and not row['collision'] and not row['timeout']
            value=N.S.arrays(row['path']);audit_mixed(q,value,p,p['teacher_first'])
            controls=controls_for(reference,query['variant'])
            first=p['teacher_first']
            assert equal(value['actions'][first:],controls[first:len(value['actions'])])
            exact=[rep for rep in replays if rep['role']==f'teacher_{rnd}_{query["variant"]}_exact_{job["id"]}']
            assert len(exact)==1 and exact[0]['array_digest']==row['array_digest']
            chosen=(value,row)
        else:
            assert not row['full']
    if chosen is None:
        assert len(queries)==len(variants) and block['teacher_selected'] is None
        return []
    assert block['teacher_selected']==chosen[1]
    value=chosen[0];x=torch.tensor(value['states'])
    return [dict(x=x,b=q.base_actions(x),target=torch.tensor(value['actions'],dtype=torch.float64),first=p['teacher_first'])]


def main(folder):
    global OUT
    N.offline()
    folder=Path(folder)
    if folder.is_absolute():assert folder.parent==ROOT_OUT
    p,base=RUN.configuration(folder.name);OUT=Path(p['out']);set_out(OUT)
    summary=read(OUT/'summary.json')
    assert not (OUT/'audit.json').exists()
    assert not Path('/proc',str(summary['pid'])).exists()
    for pid in read(OUT/'workers.json')['workers']:assert not Path('/proc',str(pid)).exists()
    for row in read(OUT/'local_manifest.json')['files']:assert sha(ROOT/row['path'])==row['sha256']
    rows,replays,commands,ledger=raw_ledger(p,summary)
    if summary['state']!='COMPLETE':
        save(OUT/'audit.json',dict(state='FAILED_PRODUCER_PARTIAL_PRESERVED',native=0,new_navigation_training=0,
             historical_discarded_Adam=0,raw_ledger=ledger,optimizer_reconstruction='NOT_EXECUTED'))
        N.manifest(OUT)
        raise AssertionError('Failed producer: no scientific acceptance or retry')
    start=time.time();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    q,opt=learner(base,p);frozen=original_hash(q)
    support=groups(q,[N.S.arrays(file) for file in p['old_history_files']],p,N.old_data(q,base,lambda:None))
    selected=joint_check(rows,'initial',q,p,summary['result']['initial'])
    mandatory={job['id'] for job in p['old8_jobs']}|{p['right_job']['id']}
    assert mandatory<=set(selected)
    blocks=summary['blocks'];assert blocks==read(OUT/'blocks.json') if blocks else not (OUT/'blocks.json').exists()
    events=[];historical=0;expected_roles=[f'initial_{job["id"]}' for job in [*p['old8_jobs'],*p['left5_jobs'],p['right_job']]]
    expected_replays=[]
    def check_block(block,before,actual,role):
        nonlocal selected
        assert block==read(OUT/'blocks'/str(block['tag'])/'result.json')
        assert actual==block['fit']
        model_readback(q,opt,block['proposed_model'])
        safe=joint_check(rows,role,q,p,block['checks'])
        numeric=actual.get('numeric_accept',True)
        accepted=bool(numeric and mandatory<=set(safe) and set(selected)<=set(safe))
        assert accepted==block['accepted']
        if accepted:selected=safe
        else:L.load(q,opt,before)
        assert selected==block['selected_success_ids'] and L.statehash(q,opt)==block['retained_hashes']
        assert original_hash(q)==frozen
        expected_roles.extend(f'{role}_{job["id"]}' for job in [*p['old8_jobs'],*p['left5_jobs'],p['right_job']])
        return accepted
    if p['kind'] in ['lateral','projected']:
        assert blocks and blocks[0]['tag']=='warm'
        before=L.state(q,opt);function=fit_A if p['kind']=='lateral' else fit_B
        actual=function(q,opt,support,p,lambda:None,lambda *_:None)
        historical+=p['consolidation_steps'];events+=optimizer_events('warm',p['consolidation_steps'])
        accepted=check_block(blocks[0],before,actual,'warm_proposal')
        if accepted:
            assert summary['result']['fresh_Adam_after_warm'] is True
            opt=torch.optim.Adam(params(q),lr=p['lr'])
            assert len(blocks)-1<=len(p['rl_jobs'])
            for rnd,block in enumerate(blocks[1:]):
                assert block['tag']==rnd and block['job']==p['rl_jobs'][rnd]
                assert block['seeds']==[[p['seed'],rnd,index] for index in range(4)]
                before=L.state(q,opt);batch,sampling=training_batch(rows,q,p,rnd,block['job'])
                assert sampling==block['sampling']
                expected_roles.extend(f'train_{rnd}_{index}' for index in range(4));expected_replays.extend(f'train_{rnd}_exact_{index}' for index in range(4))
                actual=ppo_fit(q,opt,batch,support,p,rnd,lambda:None,lambda *_:None,projected=p['kind']=='projected')
                historical+=p['epochs'];events+=optimizer_events(rnd,p['epochs'])
                check_block(block,before,actual,f'proposal_{rnd}')
        else:
            assert len(blocks)==1 and not any(row['role'].startswith('train_') for row in rows)
    elif p['kind']=='recovery':
        assert len(blocks)<=p['cycles']
        rejections=0
        for rnd,block in enumerate(blocks):
            assert block['tag']==rnd and rejections<p['reject_stop']
            before=L.state(q,opt)
            missing=[index for index,job in enumerate(p['left5_jobs']) if job['id'] not in selected]
            index=missing[0] if missing else rnd%5;job=p['left5_jobs'][index]
            assert block['job']==job and block['seeds']==[[p['seed'],rnd,k] for k in range(4)]
            support['corrections']=teacher_evidence(rows,replays,q,p,rnd,job,index,block)
            for query in block['teacher_queries']:
                expected_roles.append(f'teacher_{rnd}_{query["variant"]}_{job["id"]}')
                if query['row']['success']:expected_replays.append(f'teacher_{rnd}_{query["variant"]}_exact_{job["id"]}')
            if support['corrections']:
                actual=fit_B(q,opt,support,p,lambda:None,lambda *_:None,steps=p['correction_steps'])
                assert actual==block['warm_fit']
                historical+=p['correction_steps'];events+=optimizer_events('cw'+str(rnd),p['correction_steps'])
            else:assert block['warm_fit'] is None
            batch,sampling=training_batch(rows,q,p,rnd,job);assert sampling==block['sampling']
            expected_roles.extend(f'train_{rnd}_{k}' for k in range(4));expected_replays.extend(f'train_{rnd}_exact_{k}' for k in range(4))
            actual=ppo_fit(q,opt,batch,support,p,rnd,lambda:None,lambda *_:None,projected=True)
            historical+=p['epochs'];events+=optimizer_events(rnd,p['epochs'])
            accepted=check_block(block,before,actual,f'proposal_{rnd}')
            rejections=0 if accepted else rejections+1
    elif p['kind']=='validation':
        assert not blocks and summary['updates']==dict(live=0,shadow=0)
        assert summary['result']['diagnostic_only_not_new_Actor_training'] is True
        dev=[row for row in rows if row['role'].startswith('main54_')]
        old=[row for row in rows if row['role'].startswith('old275_')]
        assert [row['job_id'] for row in dev]==[job['id'] for job in p['main_jobs']]
        assert [row['job_id'] for row in old]==[job['id'] for job in base['final_old_jobs']]
        assert summary['result']['main54']==tally(dev) and summary['result']['old275']==tally(old)
        assert all(row['policy_hash']==q.phash() and not row['teacher'] and row['seed'] is None for row in dev+old)
        expected_roles.extend(f'main54_{job["id"]}' for job in p['main_jobs']);expected_roles.extend(f'old275_{job["id"]}' for job in base['final_old_jobs'])
        fresh=[row for row in rows if row['role'].startswith('fresh16_')]
        reloads=[row for row in replays if row['role'].startswith('reload78_')]
        if fresh:
            assert all(row['success'] for row in dev+old)
            for file,digest_value in p['fresh_ledger'].items():assert sha(file)==digest_value
            assert [row['job_id'] for row in fresh]==[job['id'] for job in p['fresh_jobs']]
            assert summary['result']['fresh16']==tally(fresh)
            assert all(row['policy_hash']==q.phash() and not row['teacher'] and row['seed'] is None for row in fresh)
            expected_roles.extend(f'fresh16_{job["id"]}' for job in p['fresh_jobs'])
        else:assert summary['result']['fresh16'] is None
        stable=bool(fresh and all(row['success'] for row in fresh) and len(reloads)==78)
        assert summary['result']['stable_wide_six']==stable
        if stable:
            assert [row['job_id'] for row in reloads]==[job['id'] for job in p['reload_jobs']]
            assert summary['result']['reload78']==dict(exact=78,independent_sample=False)
            assert all(row['loaded_model_sha256']==summary['selected_model']['sha256'] for row in reloads)
            expected_replays.extend(f'reload78_{job["id"]}' for job in p['reload_jobs'])
        else:assert not reloads and summary['result']['reload78'] is None
    else:raise AssertionError(p['kind'])
    final=joint_check(rows,'final_joint',q,p,summary['result']['final_joint'])
    assert set(selected)<=set(final);selected=final
    expected_roles.extend(f'final_joint_{job["id"]}' for job in [*p['old8_jobs'],*p['left5_jobs'],p['right_job']])
    assert [row['role'] for row in rows]==expected_roles
    assert [row['role'] for row in replays]==expected_replays
    assert jsonl(OUT/'optimizer_calls.jsonl')==events
    assert summary['updates']==dict(live=historical,shadow=historical)
    assert selected==summary['selected_success_ids']
    assert summary['selection_score']==[len(set(selected)&{job['id'] for job in p['left5_jobs']}),len(set(selected)&{job['id'] for job in p['old8_jobs']}),int(p['right_job']['id'] in selected)]
    assert summary['selected_joint_all']==(len(selected)==14)
    assert summary['selected_model']==summary['result']['model']
    model_readback(q,opt,summary['selected_model']);assert original_hash(q)==frozen
    save(OUT/'audit.json',dict(state='PASS',native=0,new_navigation_training=0,historical_discarded_Adam=historical,
         actual_source_and_model_SHA=True,original_parent_parameters_frozen_exact=True,
         proposal_Actor_Adam_and_wholecycle_rollback_exact=True,
         warm_and_PPO_live_shadow_event_order_exact=True,
         teacher_current_prefix_and_verified_control_suffix_exact=True,
         teacher_data_not_used_as_onpolicy=True,training_full_history_density_rewards_exact=True,
         raw_native_command_physics_and_replay_ledger=ledger,
         final_checkpoint_original_bytes_and_stored_Actor_Adam_exact=True,
         selected_success_ID_gate_exact=True,selected_model=summary['selected_model'],
         selected_success_ids=selected,selection_score=summary['selection_score'],
         full54_old275_fresh16_reload78=summary['result'] if p['kind']=='validation' else None,
         elapsed_s=time.time()-start,remote_backup_verified=False,seven_eight_entered=False))
    N.manifest(OUT)
    print('T410_'+p['id']+'_INDEPENDENT_AUDIT_PASS',historical,flush=True)


if __name__=='__main__':main(sys.argv[1])
