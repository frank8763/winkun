"""Independent bounded evidence/optimizer audit for the offline T425 chain.

Import is inert: no simulation, optimizer step or file write.  Scientific
stages and their historical reconstructions are accounted separately.
"""
from pathlib import Path
import copy,gzip,hashlib,io,json,sys,time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path):
    file=Path(path)
    return [json.loads(line) for line in file.read_text().splitlines() if line] if file.exists() else []


def stopped(folder):
    """The finalizer may live, while every recorded scientific PID is gone."""
    folder=Path(folder);pids=set()
    for name in ['main_started.json','workers.json','summary.json']:
        file=folder/name
        if not file.exists():continue
        row=json.loads(file.read_text())
        pids.update(row[key] for key in ['pid','parent'] if type(row.get(key)) is int)
        pids.update(row.get('workers',[]))
    assert pids
    assert all(not Path('/proc',str(pid)).exists() for pid in pids),pids
    return sorted(pids)


def manifest(root,folder):
    rows=json.loads((Path(folder)/'local_manifest.json').read_text())['files']
    for row in rows:
        file=Path(root)/row['path']
        assert file.stat().st_size==row['bytes'] and sha(file)==row['sha256'],file
    return len(rows)


def snapshot(E,record):
    assert sha(record['path'])==record['sha256']
    data=E.torch.load(io.BytesIO(gzip.decompress(Path(record['path']).read_bytes())),
                      weights_only=False,map_location='cpu')
    assert data['policy_hash']==record['policy_hash']
    assert {key:E.digest(value) for key,value in data['state'].items()}==data['hashes']==record['hashes']
    return data


def check_model(E,q,opt,record):
    data=snapshot(E,record)
    assert E.L.statehash(q,opt)==data['hashes'] and q.phash()==data['policy_hash']
    return data


def episode(E,arrays,row,step_cap,full):
    """Check recorded controls/termination; full history adds real rewards."""
    np=E.np;counts=arrays['native_counts'].astype(np.int64);flags=arrays['end_flags'].astype(bool)
    assert counts.ndim==1 and len(counts)>0 and np.all((counts>0)&(counts<=100))
    assert flags.shape==(len(counts),3) and flags[-1].any() and not flags[:-1].any()
    assert np.all(flags.sum(1)<=1)
    assert arrays['actions'].shape==(len(counts),2)
    assert np.isfinite(arrays['actions']).all() and np.abs(arrays['actions']).max()<=1
    native=int(counts.sum())
    assert native==row['physics_steps']<=step_cap
    assert row['success']==bool(flags[-1,1]) and row['collision']==bool(flags[:,0].any()) and row['timeout']==bool(flags[-1,2])
    assert row['complete'] and not row['administrative_censor']
    if row['timeout']:assert native==step_cap
    if full:
        assert E.digest(arrays)==row['array_digest']
        assert E.digest(arrays['final_native_integration'])==row['integration_sha256']
        assert arrays['states'].shape==arrays['next_states'].shape==(len(counts),120)
        assert E.equal(arrays['states'][1:],arrays['next_states'][:-1])
        assert E.equal(arrays['poses'][1:],arrays['next_poses'][:-1])
        elapsed=np.r_[0,np.cumsum(counts)]
        assert np.max(np.abs(arrays['states'][:,112]-(step_cap-elapsed[:-1])/step_cap))<=2e-6
        assert np.max(np.abs(arrays['next_states'][:,112]-(step_cap-elapsed[1:])/step_cap))<=2e-6
        goal=arrays['task_goal_xy'];d0=np.linalg.norm(arrays['poses'][:,:2]-goal,axis=1)
        d1=np.linalg.norm(arrays['next_poses'][:,:2]-goal,axis=1)
        gamma=E.N.K.R.GAMMA;discount=gamma**counts
        reward=-.000002*(1-discount)/(1-gamma)+.03*d0-discount*.03*d1*(~flags.any(1))
        reward+=gamma**(counts-1)*(flags[:,1].astype(float)-flags[:,0]-.5*flags[:,2])
        assert np.max(np.abs(reward-arrays['rewards']))<2e-12
        assert abs(float(d1[-1])-row['remaining_m'])<1e-12
        if row['success']:assert float(d1[-1])<=.20000001 and not row['collision']
    return native


def reconstruct_pair(E,q,opt,run,expected_fit,guard,event_rows,tag,gradient_hash):
    """Actually execute live and shadow historical fits, matching each event.

    The registered producer provides the exact clipped-gradient hash schema;
    this function never substitutes a norm for a full gradient/state digest.
    """
    before=E.L.state(q,opt);results=[];states=[];events=[]
    for which in ['live','shadow']:
        E.L.load(q,opt,before);local=[]
        def event(where,k):
            guard();assert where in ['BEFORE','AFTER']
            row=dict(tag=tag,which=which,where=where,k=k,
                     hashes=E.L.statehash(q,opt),gradient_sha256=gradient_hash(q))
            local.append(row)
            expected=event_rows[len(events)+len(local)-1]
            assert row==expected,('Historical optimizer event mismatch',tag,which,where,k)
        result=run(event)
        assert result==expected_fit
        results.append(result);states.append(E.L.state(q,opt));events.extend(local)
    assert E.digest(states[0])==E.digest(states[1])
    E.L.load(q,opt,states[0]);return dict(events=events,results=results)


def check_teacher(C,q,arrays,request,p):
    """Independently reconstruct executed teacher controls on actual states."""
    E,np=C.E,C.np;first=int(request.get('first',0));mode=request['mode']
    assert mode in ['safe_reset','feedback'] and request.get('seed') is None
    assert not arrays['ownership'][:first].any() and arrays['ownership'][first:].all()
    assert not arrays['stochastic'].any()
    E.audit_array(q,arrays,p,first,None)
    if mode=='safe_reset':
        with C.torch.no_grad():
            actions=np.concatenate([q.base_actions(C.torch.tensor(row[None])).clamp(-1,1).float().numpy() for row in arrays['states']])
        assert E.equal(actions[first:],arrays['actions'][first:])
        assert C.np.all(arrays['feedback_progress']==-1) and C.np.all(arrays['feedback_speed']==0)
        return
    ref=C.R.load_reference(request['reference_path'],request['reference_sha256'])
    assert E.equal(ref['goal'],arrays['task_goal_xy'])
    # The neighbouring successful reference intentionally has a different
    # perturbed start; no native/hidden/history state is copied from it.
    n=len(ref['arrays']['actions']);progress=first
    elapsed=np.r_[0,np.cumsum(arrays['native_counts'].astype(np.int64))]*.002
    gains=np.asarray(request['gains'],np.float64)
    for i in range(first,len(arrays['actions'])):
        pose=arrays['poses'][i];speed=float(arrays['feedback_speed'][i])
        indices=np.arange(progress,min(n,progress+12)+1)
        delta=ref['poses'][indices,:2]-pose[:2]
        yaw=(ref['poses'][indices,2]-pose[2]+np.pi)%(2*np.pi)-np.pi
        costs=np.square(delta).sum(1)+.04*np.square(yaw)
        progress=int(indices[int(np.argmin(costs))])
        goal_delta=ref['goal']-pose[:2];distance=float(np.linalg.norm(goal_delta))
        if progress==n:
            target=ref['goal'];target_yaw=float(np.arctan2(goal_delta[1],goal_delta[0]));target_speed=.025;command=np.zeros(2)
        else:
            target=ref['poses'][progress,:2];target_yaw=float(ref['poses'][progress,2]);target_speed=float(ref['refspeed'][progress]);command=ref['arrays']['actions'][progress].astype(np.float64)
        diff=target-pose[:2];c,s=np.cos(pose[2]),np.sin(pose[2])
        forward=float(c*diff[0]+s*diff[1]);left=float(-s*diff[0]+c*diff[1]);angle=float((target_yaw-pose[2]+np.pi)%(2*np.pi)-np.pi)
        ky,kpsi,kx,kv,kyawslow,kendslow=gains
        action=np.clip([command[0]+kx*forward+kv*(target_speed-speed)-kyawslow*abs(angle)-kendslow*max(0.,1.-distance/.6),command[1]+ky*left+kpsi*angle],-1,1).astype(np.float32)
        original,actual_progress,_=C.R.feedback(ref,pose,speed,float(elapsed[i]),int(arrays['feedback_progress'][i-1]) if i>first else first,gains,'progress')
        assert progress==actual_progress==int(arrays['feedback_progress'][i])
        assert E.equal(action,original) and E.equal(action,arrays['actions'][i]),i
    assert np.all(arrays['feedback_progress'][:first]==-1)
    assert np.all(arrays['feedback_speed'][:first]==0)
    assert np.array_equal(arrays['feedback_speed'][first:],arrays['states'][first:,56])


def checked_array(C,row):
    E=C.E
    assert sha(row['path'])==row['sha256']
    arrays=E.N.S.arrays(row['path'])
    episode(E,arrays,row,row['request']['job']['case']['step_cap'],row['full'])
    assert set(arrays)<=set(row['array_hashes'])
    for key,value in arrays.items():assert E.digest(value)==row['array_hashes'][key],(row['role'],row['job_id'],key)
    assert row['job_id']==row['request']['job']['id']
    assert row['mode']==row['request'].get('mode','student')
    assert row['loaded_model_sha256']==sha(row['request']['model'])
    assert row['seed']==row['request'].get('seed')
    assert row['physics']==row['physics_steps']
    return arrays


def check_full_policy(C,arrays,row,p):
    q=C.frozen(p,row['request']['model']).requires_grad_(False)
    assert q.phash()==row['policy_hash']
    fixed=C.digest(q.state_dict())
    if row['teacher']:check_teacher(C,q,arrays,row['request'],p)
    else:
        assert row['mode']=='student' and not arrays['ownership'].any()
        C.E.audit_array(q,arrays,p,0,row['seed'])
    assert C.digest(q.state_dict())==fixed


def complete_development(C,p,check,rows,role,model):
    expected=[*p['main_jobs'],*p['old275_jobs']]
    chosen=[row for row in rows if row['role']==role]
    assert len(expected)==329 and [row['job_id'] for row in chosen]==[job['id'] for job in expected]
    assert check['rows']==chosen
    assert all(not row['teacher'] and row['seed'] is None and row['policy_hash']==model['policy_hash']
               and row['loaded_model_sha256']==model['sha256'] for row in chosen)
    successes=sorted(row['job_id'] for row in chosen if row['success'] and not row['collision'])
    assert check['success_ids']==successes and check['all_safe']==(len(successes)==329)
    assert check['wide54']==C.E.tally(chosen[:54]) and check['old275']==C.E.tally(chosen[54:])
    assert check['policy_hash']==model['policy_hash']
    return set(successes),chosen


def audit_data(C,p,out,summary,rows,replays,guard):
    E=C.E;result=summary['result'];source=result['model']
    assert sha(source['path'])==source['sha256']
    targets=C.read(out/'targets.json');assert targets==result['targets']
    assert result['learning_data_available']==bool(targets)
    assert result['complete_target_coverage']==(len(targets)==19)
    assert result['unique_safe_recovery_targets']==len(targets)
    expected_rows=[];expected_replays=[];target_lookup={entry['id']:entry for entry in targets}
    assert len(target_lookup)==len(targets)
    configs={}
    for target in p['targets']:
        identity=target['job']['id'];gains=p['feedback_gains'] if target['kind']=='feedback' else [None]
        trials=[]
        for index,gain in enumerate(gains):
            role='QUERY_'+identity+'_'+str(index);chosen=[row for row in rows if row['role']==role]
            assert len(chosen)==1;row=chosen[0];expected_rows.append(row)
            req=row['request'];assert req['job']==target['job'] and req['model']==source['path']
            assert req['mode']==target['kind'] and req['first']==(0 if target['kind']=='safe_reset' else p['teacher_first'])
            assert req['seed'] is None and row['teacher'] and not row['full']
            if gain is not None:
                assert req['gains']==gain and req['reference_path']==target['reference_path'] and req['reference_sha256']==target['reference_sha256']
            if row['success'] and not row['collision']:trials.append(row)
        if not trials:
            assert identity not in target_lookup
            assert not any(row['role']=='EXACT_'+identity for row in replays)
            continue
        best=min(trials,key=lambda row:(row['physics'],row['remaining_m']));entry=target_lookup[identity]
        assert entry['request']==best['request'] and entry['first']==best['request']['first'] and entry['exact']
        selected=dict(best,path=entry['path'],sha256=entry['sha256'],full=True)
        assert entry['row']==selected
        guard();arrays=checked_array(C,selected);check_full_policy(C,arrays,selected,p)
        assert arrays['end_flags'][-1,1] and not arrays['end_flags'][:,0].any()
        repeats=[row for row in replays if row['role']=='EXACT_'+identity];assert len(repeats)==1
        repeat=repeats[0];expected_replays.append(repeat)
        assert repeat['request']==entry['request'] and repeat['reference_path']==entry['path']
        assert repeat['exact'] and repeat['all_arrays_exact'] and not repeat['independent_sample']
        for key in ['array_digest','array_hashes','integration_sha256','physics_steps','success','collision','timeout','policy_hash']:
            assert repeat[key]==best[key],(identity,key)
        if target['kind']=='feedback':configs[identity]=entry['request']['gains']
    assert rows==expected_rows and replays==expected_replays
    assert configs==result['feedback_configs']
    assert result['development_all_safe'] is False
    return dict(new_training_Adam=0,historical_discarded_Adam=0,
                full_successful_teacher_targets_verified=len(targets),
                actual_teacher_controls_reconstructed=True,
                all_selected_teacher_native_repeats_exact=True,
                new_navigation_gate=False)


def audit_arm(C,p,out,summary,rows,replays,guard,kind):
    E=C.E;L=C.LEARNING;result=summary['result'];data=C.read(C.OUT/'data/summary.json')['result']
    assert C.read(C.OUT/'data/audit.json')['state']=='PASS'
    q=C.frozen(p,result['initial_model']['path']);opt=L.make_optimizer(q,p)
    check_model(E,q,opt,result['initial_model'])
    fixed=E.original_hash(q)
    initial=set(p['initial_success_ids']);selected=set(initial)
    all_safe=False;development_file=None
    if kind=='b' and (C.OUT/'a/summary.json').exists():
        prior=C.read(C.OUT/'a/summary.json')['result'];selected=set(prior['selected_success_ids'])
        all_safe=prior['development_all_safe'];development_file=prior.get('development_file')
        assert result['initial_model']['policy_hash']==prior['model']['policy_hash']
    targets=copy.deepcopy(data['targets']);expected_rows=[];expected_replays=[];events=jsonl(out/'optimizer_calls.jsonl');expected_events=[];steps=0;consecutive=0
    rounds=result['rounds']
    if rounds:assert C.read(out/'rounds.json')==rounds
    else:assert not (out/'rounds.json').exists()
    assert len(rounds)<=p['maximum_rounds'][kind]
    for rnd,record in enumerate(rounds):
        guard();assert C.read(out/'blocks'/str(rnd)/'result.json')==record
        assert record['kind']==kind and record['round']==rnd and record['selected_before']==sorted(selected)
        before=check_model(E,q,opt,record['before_model'])['state'];refresh=[]
        if rnd or kind=='b':
            for target in p['targets']:
                identity=target['job']['id']
                if target['kind']!='feedback' or identity not in data['feedback_configs']:continue
                role='REFRESH_'+str(rnd)+'_'+identity
                chosen=[row for row in rows if row['role']==role];assert len(chosen)==1
                row=chosen[0];expected_rows.append(row)
                assert row['full'] and row['teacher'] and row['request']['model']==record['before_model']['path']
                assert row['request']['gains']==data['feedback_configs'][identity]
                assert row['request']['job']==target['job'] and row['request']['mode']=='feedback' and row['request']['first']==p['teacher_first']
                if row['success'] and not row['collision']:
                    repeated=[row2 for row2 in replays if row2['role']=='REFRESH_EXACT_'+str(rnd)+'_'+identity];assert len(repeated)==1
                    repeat=repeated[0];expected_replays.append(repeat)
                    assert repeat['reference_path']==row['path'] and repeat['exact'] and repeat['all_arrays_exact'] and not repeat['independent_sample']
                    for key in ['array_digest','array_hashes','integration_sha256','physics_steps','success','collision','timeout','policy_hash']:
                        assert repeat[key]==row[key],key
                    entry=dict(id=identity,path=row['path'],sha256=row['sha256'],first=row['request']['first'],row=row,request=row['request'],exact=True)
                    targets=[entry0 for entry0 in targets if entry0['id']!=identity]+[entry];refresh.append(entry)
        assert record['refresh']==refresh and record['targets']==targets
        source_targets=[]
        for entry in targets:
            assert sha(entry['path'])==entry['sha256']
            arrays=E.N.S.arrays(entry['path']);assert arrays['end_flags'][-1,1] and not arrays['end_flags'][:,0].any()
            source_targets.append(dict(arrays=arrays,first=entry['first'],id=entry['id']))
        groups=L.make_groups(q,p,C.anchors(p),source_targets)
        items=[]
        if kind=='b':
            training=[row for row in rows if row['role']=='TRAIN_'+str(rnd)]
            assert training==record['training'] and len(training)==4
            assert [row['job_id'] for row in training]==[job['id'] for job in p['wide_failed_jobs']]
            for k,row in enumerate(training):
                assert row['seed']==[p['seed'],rnd,k] and not row['teacher'] and row['full']
                assert row['loaded_model_sha256']==record['before_model']['sha256'] and row['policy_hash']==q.phash()
                arrays=E.N.S.arrays(row['path']);E.audit_array(q,arrays,p,0,row['seed'])
                assert not arrays['ownership'].any() and arrays['stochastic'].all();items.append(arrays)
            expected_rows.extend(training)
            repeated=[row for row in replays if row['role']=='TRAIN_EXACT_'+str(rnd)]
            assert len(repeated)==4
            for old,new in zip(training,repeated):
                assert new['reference_path']==old['path'] and new['seed']==old['seed']
                assert new['exact'] and new['all_arrays_exact'] and not new['independent_sample']
                for key in ['array_digest','array_hashes','integration_sha256','physics_steps','success','collision','timeout','policy_hash']:
                    assert new[key]==old[key],key
            expected_replays.extend(repeated)
        else:assert record['training']==[]
        tag=kind+str(rnd);tag_events=[row for row in events if row['tag']==tag]
        expected_count=p['a_steps'] if kind=='a' else 8
        assert len(tag_events)==expected_count*4
        run=(lambda event:L.fit_a(q,opt,groups,p,guard,event)) if kind=='a' else (lambda event:L.fit_b(q,opt,items,groups,p,rnd,guard,event))
        pair=reconstruct_pair(E,q,opt,run,record['fit'],guard,tag_events,tag,C.gradient_sha)
        expected_events.extend(pair['events']);steps+=expected_count
        check_model(E,q,opt,record['proposed_model'])
        success,development=complete_development(C,p,record['check'],rows,'PROPOSAL_'+str(rnd),record['proposed_model']);expected_rows.extend(development)
        accepted=(selected<=success and initial<=success and bool(success-selected)
                  and (kind=='a' or record['fit']['numeric_accept']))
        assert accepted==record['accepted']
        if accepted:
            selected=success;all_safe=record['check']['all_safe'];development_file=str(out/'blocks'/str(rnd)/'development.json')
            assert C.read(development_file)==record['check']
        else:E.L.load(q,opt,before)
        assert E.L.statehash(q,opt)==record['retained_hashes']
        assert E.original_hash(q)==fixed
        consecutive=0 if accepted else consecutive+1
        assert record['selected_after']==sorted(selected) and record['consecutive_rejections']==consecutive
        if consecutive>=2 or all_safe:assert rnd==len(rounds)-1
    assert events==expected_events and rows==expected_rows and replays==expected_replays
    assert result['updates']==dict(live=steps,shadow=steps)
    check_model(E,q,opt,result['model'])
    assert result['selected_success_ids']==sorted(selected)
    assert result['development_all_safe']==all_safe and result['development_file']==development_file
    assert result['old_training_audit_not_repaired'] is True
    return dict(new_training_Adam=0,historical_discarded_Adam=2*steps,
                actual_recorded_live_steps=steps,actual_recorded_shadow_steps=steps,
                complete_Actor_Adam_and_clipped_gradient_every_event_exact=True,
                full_actual_history_and_label_mask_preserved=True,
                no_external_teacher_controls_enter_PPO_ratio=True,
                every_proposal_full329_success_ID_gate_reconstructed=True,
                final_development_all_safe=all_safe)


def audit_gate(C,p,out,summary,rows,replays,guard):
    E=C.E;result=summary['result'];model=result['model'];assert sha(model['path'])==model['sha256']
    development=C.read(result['development_file'])
    assert development['all_safe'] and len(development['success_ids'])==329
    assert development['policy_hash']==model['policy_hash']
    lookup={row['job_id']:row for row in development['rows']}
    ids={job['id'] for job in p['reload_jobs']}
    jobs=[job for job in [*p['main_jobs'],*p['old275_jobs']] if job['id'] in ids]
    rebuild=[row for row in rows if row['role']=='REBUILD62']
    assert len(jobs)==62 and [row['job_id'] for row in rebuild]==[job['id'] for job in jobs]
    for row in rebuild:
        guard();assert row['full'] and row['loaded_model_sha256']==model['sha256']
        prior=lookup[row['job_id']]
        for key in ['array_digest','array_hashes','integration_sha256','physics_steps','success','collision','timeout']:
            assert row[key]==prior[key],key
    for file,hashvalue in p['fresh_ledger'].items():assert sha(file)==hashvalue
    fresh=[row for row in rows if row['role']=='FRESH16']
    assert [row['job_id'] for row in fresh]==[job['id'] for job in p['fresh_jobs']]
    assert rows==rebuild+fresh and all(row['full'] and not row['teacher'] and row['seed'] is None
                                      and row['loaded_model_sha256']==model['sha256'] and row['policy_hash']==model['policy_hash'] for row in rows)
    assert result['fresh16']==E.tally(fresh)
    passed=all(row['success'] and not row['collision'] for row in fresh)
    if passed:
        assert len(replays)==78 and [row['job_id'] for row in replays]==[job['id'] for job in p['reload_jobs']]
        reference={row['job_id']:row for row in rows}
        for row in replays:
            guard();prior=reference[row['job_id']]
            assert row['role']=='RELOAD78' and row['reference_path']==prior['path']
            assert row['loaded_model_sha256']==model['sha256'] and not row['teacher'] and row['seed'] is None
            assert row['exact'] and row['all_arrays_exact'] and not row['independent_sample']
            for key in ['array_digest','array_hashes','integration_sha256','physics_steps','success','collision','timeout','policy_hash']:
                assert row[key]==prior[key],key
        assert result['reload78']==dict(exact=78,all_arrays_exact=True,independent_sample=False)
    else:assert not replays and result['reload78'] is None
    assert result['six_gate']==passed
    return dict(new_training_Adam=0,historical_discarded_Adam=0,six_gate=passed,
                complete329_predecessor_proof_checked=True,full_reference_rebuilds=62,
                heldout_navigation_cases=16,independent_native_reloads=len(replays),
                independent_native_reload_every_recorded_array_hash_exact=True,
                old_T422_training_history_not_repaired=True)


def audit_eight(C,p,out,summary,rows,replays,guard):
    result=summary['result'];assert not replays
    assert C.read(C.OUT/'gate/summary.json')['result']['six_gate']
    assert C.read(C.OUT/'gate/audit.json')['state']=='PASS'
    geometry=C.read(C.OUT/'eight_geometry.json')
    assert len(geometry)==len(p['eight_jobs'])
    assert [entry['id'] for entry in geometry]==[job['id'] for job in p['eight_jobs']]
    valid={entry['id'] for entry in geometry if entry['valid']}
    assert [row['job_id'] for row in rows]==[job['id'] for job in p['eight_jobs'] if job['id'] in valid]
    assert result['invalid_geometry']==[entry for entry in geometry if not entry['valid']]
    assert all(row['role']=='EIGHT_DIAGNOSTIC' and row['seed'] is None and not row['teacher']
               and row['loaded_model_sha256']==result['model']['sha256'] and row['policy_hash']==result['model']['policy_hash'] for row in rows)
    assert result['diagnostic']==C.E.tally(rows) and result['eight_started'] and not result['eight_training']
    assert result['seven_skipped'] and result['six_gate_previously_passed'] and result['not_eight_stable_acceptance']
    return dict(new_training_Adam=0,historical_discarded_Adam=0,eight_frozen_diagnostic_only=True,
                seven_skipped_by_explicit_authorization=True,no_eight_stability_claim=True)


def main(name):
    import t425_core as C
    assert name in C.ALIASES
    p=C.read(C.OUT/'protocol.json');C.verify(p);out=C.OUT/C.ALIASES[name]
    summary=C.read(out/'summary.json')
    assert summary['state']=='COMPLETE' and summary['all_workers_stopped'] and summary['all_sessions_closed']
    assert not (out/'audit.json').exists()
    absent=stopped(out);members=manifest(C.ROOT,out)
    assert not list(out.rglob('partial*.npz'))
    start=time.time();budget=p['stage_budgets'][name]
    C.E.N.offline();C.torch.set_num_threads(1);C.torch.use_deterministic_algorithms(True)
    def guard():assert time.time()<min(start+budget['audit_seconds']-5,p['deadline_all']-5)
    def no_native(*args,**kwargs):raise AssertionError('Independent audit cannot launch native simulation')
    C.E.L.Session=no_native;C.E.Pool=no_native;C.Pool=no_native
    rows=jsonl(out/'cases.jsonl');replays=jsonl(out/'replays.jsonl');commands=jsonl(out/'commands.jsonl')
    for row in rows:
        guard();arrays=checked_array(C,row)
        if row['full']:check_full_policy(C,arrays,row,p)
        assert row['pid'] in absent
    # Every repeated native execution remains charged separately. The
    # producer compares all arrays; this audit compares the persisted hashes.
    for row in replays:
        assert row['physics']==row['physics_steps'] and row['pid'] in absent
        assert row['complete'] and not row['administrative_censor']
        assert row['exact'] and row['all_arrays_exact'] and not row['independent_sample']
    if name=='data':details=audit_data(C,p,out,summary,rows,replays,guard)
    elif name in ['arm_a','arm_b']:details=audit_arm(C,p,out,summary,rows,replays,guard,'a' if name=='arm_a' else 'b')
    elif name=='gate':details=audit_gate(C,p,out,summary,rows,replays,guard)
    else:details=audit_eight(C,p,out,summary,rows,replays,guard)
    assert [row['id'] for row in commands]==list(range(1,len(commands)+1))
    assert len(commands)==summary['commands']==4+len(rows)+len(replays)<=budget['max_commands']
    assert [(row['role'],row['physics']) for row in commands[:4]]==[('SETUP',0)]*4
    expected=[(row['role'],row['physics']) for row in rows+replays]
    actual=[(row['role'],row['physics']) for row in commands[4:]]
    # Cases and exact replays are interleaved by the registered stage; retain
    # the exact command order while matching each distinct role's row order.
    assert sorted(actual)==sorted(expected)
    for role in {row[0] for row in expected}:
        assert [item for item in actual if item[0]==role]==[item for item in expected if item[0]==role]
    assert sum(row['physics'] for row in commands)==summary['physics']<=budget['max_physics']
    assert summary['elapsed_s']<=budget['main_seconds']
    guard();C.verify(p)
    result=dict(state='PASS',stage=name,native=0,new_navigation_training=0,
                elapsed_s=time.time()-start,native_processes_absent=absent,
                manifest_members_before_audit=members,physics=summary['physics'],commands=summary['commands'],
                raw_controls_counts_terminal_and_command_ledger_exact=True,
                actual_full_payload_history_hidden_actions_rewards_checked=True,
                compact_evidence_limitation='Compact acceptance/search cases retain actual actions/counts/flags and all original array hashes. This independent zero-native audit does not reconstruct absent full120 observations.',
                original_T422_training_history_not_repaired=True,**details)
    C.save(out/'audit.json',result);C.E.N.manifest(out)
    print('T425_STAGE_AUDIT_PASS',name,flush=True)


if __name__=='__main__':
    assert len(sys.argv)==2
    main(sys.argv[1])
