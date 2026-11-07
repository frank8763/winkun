"""Independent raw-data and historical Actor/Adam reconstruction, no physics.

Teacher labels are checked against the frozen expert on actual observations,
including the unexecuted expert queries during the student's owned prefix.
"""
from t412_learning import *

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

def joint_check(rows,role,q,p,record):
    actual=[]
    for suffix,jobs in [('old8',p['old8_jobs']),('left5',p['left5_jobs']),('right',[p['right_job']])]:
        selected=[r for r in rows if r['role'].startswith(role+'_'+suffix+'_')]
        assert [r['job_id'] for r in selected]==[j['id'] for j in jobs]
        actual.extend(selected)
    assert len(actual)==14 and all(r['policy_hash']==q.phash() for r in actual)
    assert all(r.get('seed') is None and not r.get('teacher',False) for r in actual)
    ids=sorted(r['job_id'] for r in actual if r['success'])
    expected=dict(policy_hash=q.phash(),success_ids=ids,all14=len(ids)==14,
                  old8=tally(actual[:8]),left5=tally(actual[8:13]),right1=tally(actual[13:]))
    assert record==expected
    return ids

def raw_ledger(p,s):
    rows=jsonl(OUT/'cases.jsonl');reps=jsonl(OUT/'replays.jsonl');commands=jsonl(OUT/'commands.jsonl')
    for row in rows:
        assert sha(row['path'])==row['sha256'];a=N.S.arrays(row['path'])
        assert int(a['native_counts'].sum())==row['physics_steps']==row['physics']
        assert bool(a['end_flags'][-1,1])==row['success']
        assert bool(a['end_flags'][:,0].any())==row['collision']
        assert bool(a['end_flags'][-1,2])==row['timeout']
        assert a['end_flags'][-1].any() and not a['end_flags'][:-1].any()
        if row['full']:
            assert digest(a)==row['array_digest']
            assert digest(a['final_native_integration'])==row['integration_sha256']
            assert equal(a['states'][1:],a['next_states'][:-1])
    partials=[]
    for file in sorted(OUT.rglob('partial*.npz')):
        a=N.S.arrays(file);partials.append(dict(path=str(file),sha256=sha(file),recorded_physics=int(a['native_counts'].sum()),administrative_censor=True))
    for row in reps:
        role=row['role']
        if role.startswith('teacher_') and '_exact_' in role:
            matched=[r for r in rows if r['role']==role.replace('_exact_','_')]
        elif role.startswith('train_') and '_exact_' in role:
            matched=[r for r in rows if r['role']==role.replace('_exact_','_')]
        elif role.startswith('final_centres_exact_'):
            matched=[r for r in rows if r['role']==role.replace('final_centres_exact_','final_centres_',1)]
        elif role.startswith('reload78_'):
            matched=[r for r in rows if r['job_id']==row['job_id'] and any(r['role'].startswith(prefix) for prefix in ['main54_','old275_','fresh16_'])]
        else:raise AssertionError(('Undeclared replay',role))
        assert len(matched)==1
        ref=matched[0]
        for key in ['array_digest','integration_sha256','physics_steps','success','collision','timeout']:
            assert row[key]==ref[key],(role,key)
        assert row['exact'] is True and row['independent_sample'] is False
    recorded=sum(r['physics_steps'] for r in rows+reps)
    ledger=dict(cases=len(rows),exact_replays=len(reps),command_records=len(commands),summary_commands=s['commands'],complete_physics=recorded,summary_physics=s['physics'],unledgered_physics=s['physics']-recorded,unledgered_commands=s['commands']-len(commands),partials=partials)
    if s['state']=='COMPLETE':
        assert not partials
        assert len(commands)==s['commands']==4+len(rows)+len(reps)
        assert [r['id'] for r in commands]==list(range(1,len(commands)+1))
        assert sum(r['physics'] for r in commands)==recorded==s['physics']
        assert not any(r['error'] for r in commands)
        assert s['physics']<=p['max_physics'] and s['commands']<=p['max_commands'] and s['elapsed_s']<=p['main_seconds']
    return rows,reps,commands,ledger

def events_for(tag,count):
    output=[]
    for which in ['live','shadow']:
        for k in range(count):output.extend([[tag,which,'BEFORE',k],[tag,which,'AFTER',k]])
    return output

def main():
    N.offline();p=read(OUT/'protocol.json');assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for file,h in p['inputs'].items():assert sha(file)==h,file
    base=N.science(p);s=read(OUT/'summary.json');assert not (OUT/'audit.json').exists() and not Path('/proc',str(s['pid'])).exists()
    for pid in read(OUT/'workers.json')['workers']:assert not Path('/proc',str(pid)).exists()
    for f in read(OUT/'local_manifest.json')['files']:assert sha(ROOT/f['path'])==f['sha256']
    rows,reps,commands,ledger=raw_ledger(p,s)
    if s['state']!='COMPLETE':
        save(OUT/'audit.json',dict(state='FAILED_PRODUCER_PARTIAL_PRESERVED',native=0,historical_discarded_Adam=0,raw_ledger=ledger));N.manifest(OUT);raise AssertionError('Incomplete producer; no PASS')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);q,opt=learner(base,p);frozen=original_hash(q)
    initial=[r for r in rows if r['role'].startswith('initial_old8_')];assert [r['job_id'] for r in initial]==[j['id'] for j in p['old8_jobs']];arrays=[]
    for r in initial:
        assert r['full'] and r['success'] and r['policy_hash']==q.phash();a=N.S.arrays(r['path']);audit_array(q,a,p,0,None);assert not a['ownership'].any();arrays.append(a)
    selected=joint_check(rows,'initial',q,p,s['result']['initial']);mandatory={j['id'] for j in p['old8_jobs']}|{p['right_job']['id']};assert mandatory<=set(selected)
    g=groups(q,arrays,p,N.old_data(q,base,lambda:None));events=[];calls=0;anyaccepted=False
    expected=[f'initial_old8_{j["id"]}' for j in p['old8_jobs']]+[f'initial_left5_{j["id"]}' for j in p['left5_jobs']]+[f'initial_right_{p["right_job"]["id"]}'];expected_reps=[]
    def joint_roles(role):return [f'{role}_old8_{j["id"]}' for j in p['old8_jobs']]+[f'{role}_left5_{j["id"]}' for j in p['left5_jobs']]+[f'{role}_right_{p["right_job"]["id"]}']
    for rnd,record in enumerate(s['rounds']):
        assert rnd<len(p['query_pairs']) and record['round']==rnd and record==read(OUT/'blocks'/str(rnd)/'result.json');before=L.state(q,opt);corrections=[]
        assert len(record['queries'])==len(p['query_pairs'][rnd])==2
        for index,query in zip(p['query_pairs'][rnd],record['queries']):
            case=query['case'];job=p['left5_jobs'][index];assert case==dict(id=f'query_{rnd}_{index}',job=job,first=p['teacher_first'],source_control=False);role=f'teacher_{rnd}_{index}_{job["id"]}';r=next(x for x in rows if x['role']==role);assert r==query['row'] and r['full'] and r['teacher'] and r['policy_hash']==q.phash() and r['seed'] is None
            a=N.S.arrays(r['path']);audit_query(q,a,case,p);expected.append(role)
            if r['success']:
                assert not r['collision'] and not r['timeout'];corrections.append(query_group(q,a));expected_reps.append(f'teacher_{rnd}_{index}_exact_{job["id"]}')
        g['corrections']=corrections
        if not corrections:
            assert record['bc'] is record['rl'] is None and record['selected_success_ids']==selected;assert rnd==len(s['rounds'])-1 and s['reason']=='NO_VERIFIED_RECOVERY_LABELS';break
        actual=fit_bc(q,opt,g,p,lambda:None,lambda *_:None);calls+=p['bc_steps'];events+=events_for('BC'+str(rnd),p['bc_steps']);bc=record['bc'];assert bc['fit']==actual;model_readback(q,opt,bc['proposed_model']);safe=joint_check(rows,f'BC_{rnd}_proposal',q,p,bc['checks']);expected+=joint_roles(f'BC_{rnd}_proposal');keep=mandatory<=set(safe) and set(selected)<=set(safe);assert keep==bc['accepted']
        if keep:selected=safe;anyaccepted=True
        else:L.load(q,opt,before)
        assert L.statehash(q,opt)==bc['retained_hashes'] and original_hash(q)==frozen
        if keep:
            opt=torch.optim.Adam(params(q),lr=p['lr']);postbc=L.state(q,opt);rl=record['rl'];assert rl is not None;job=p['left5_jobs'][p['query_pairs'][rnd][0]];assert rl['job']==job and rl['seeds']==[[p['seed'],rnd,k] for k in range(4)];batch=[];batchrows=[]
            for k in range(4):
                r=next(x for x in rows if x['role']==f'train_{rnd}_{k}');assert r['full'] and not r['teacher'] and r['job_id']==job['id'] and r['policy_hash']==q.phash() and r['seed']==rl['seeds'][k];a=N.S.arrays(r['path']);audit_array(q,a,p,0,r['seed']);assert not a['ownership'].any();batch.append(a);batchrows.append(r);expected.append(r['role']);expected_reps.append(f'train_{rnd}_exact_{k}')
            actual=fit_ppo(q,opt,batch,g,p,rnd,lambda:None,lambda *_:None);calls+=p['epochs'];events+=events_for(rnd,p['epochs']);assert actual==rl['fit'] and tally(batchrows)==rl['sampling'];model_readback(q,opt,rl['proposed_model']);safe=joint_check(rows,f'RL_{rnd}_proposal',q,p,rl['checks']);expected+=joint_roles(f'RL_{rnd}_proposal');keepRL=actual['numeric_accept'] and mandatory<=set(safe) and set(selected)<=set(safe);assert keepRL==rl['accepted']
            if keepRL:selected=safe;anyaccepted=True
            else:L.load(q,opt,postbc)
            assert L.statehash(q,opt)==rl['retained_hashes']
        else:assert record['rl'] is None
        assert record['selected_success_ids']==selected
    assert s['updates']==dict(live=calls,shadow=calls) and events==jsonl(OUT/'optimizer_calls.jsonl');assert original_hash(q)==frozen
    result=s['result'];model_readback(q,opt,result['model']);assert result['any_learning_accepted']==anyaccepted
    centres=[r for r in rows if r['role'].startswith('final_centres_')];assert [r['job_id'] for r in centres]==[j['id'] for j in p['centres']] and tally(centres)==result['main_centres'];expected.extend(r['role'] for r in centres);expected_reps.extend(f'final_centres_exact_{j["id"]}' for j in p['centres']);final=joint_check(rows,'final_joint',q,p,result['final_joint']);assert set(selected)<=set(final);expected+=joint_roles('final_joint')
    dev=[r for r in rows if r['role'].startswith('main54_')];old=[r for r in rows if r['role'].startswith('old275_')];fresh=[r for r in rows if r['role'].startswith('fresh16_')];reloads=[r for r in reps if r['role'].startswith('reload78_')]
    if anyaccepted:
        assert [r['job_id'] for r in dev]==[j['id'] for j in p['main_jobs']] and [r['job_id'] for r in old]==[j['id'] for j in base['final_old_jobs']];assert tally(dev)==result['main54'] and tally(old)==result['old275'];expected.extend(r['role'] for r in dev+old)
    else:assert not dev and not old and result['main54'] is result['old275'] is None
    if fresh:
        assert anyaccepted and all(r['success'] for r in dev+old);assert [r['job_id'] for r in fresh]==[j['id'] for j in p['fresh_jobs']] and tally(fresh)==result['fresh16'];expected.extend(r['role'] for r in fresh)
        for file,h in p['fresh_ledger'].items():assert sha(file)==h
    else:assert result['fresh16'] is None
    stable=bool(fresh and all(r['success'] for r in fresh) and len(reloads)==78);assert result['stable_wide_six']==stable
    if stable:assert result['reload78']==dict(exact=78,independent_sample=False) and [r['job_id'] for r in reloads]==[j['id'] for j in p['reload_jobs']];expected_reps.extend(r['role'] for r in reloads)
    else:assert not reloads and result['reload78'] is None
    for r in [*centres,*dev,*old,*fresh]:assert r['policy_hash']==q.phash()
    for r in [x for x in reps if x['role'].startswith(('final_centres_exact_','reload78_'))]:assert r['loaded_model_sha256']==result['model']['sha256'] and r['policy_hash']==q.phash()
    assert [r['role'] for r in rows]==expected and [r['role'] for r in reps]==expected_reps
    save(OUT/'audit.json',dict(state='PASS',native=0,new_training=0,historical_discarded_Adam=calls,raw_ledger=ledger,all_models_Actor_Adam_rejections_and_fresh_Adam_exact=True,all_teacher_current_history_labels_and_both_hidden_exact=True,expert_queries_include_student_prefix=True,pure_student_training_density_rewards_exact=True,all_live_shadow_events_and_final_serialized_model_exact=True,safe_parent_original_parameters_frozen=True,old8_RIGHT_retained_LEFT_ID_gate_exact=True,stable_wide_six=stable,main54=result['main54'],old275=result['old275'],fresh16=result['fresh16'],reload78=result['reload78'],seven_eight_entered=False));N.manifest(OUT);print('T412_INDEPENDENT_AUDIT_PASS',calls,flush=True)
if __name__=='__main__':main()

