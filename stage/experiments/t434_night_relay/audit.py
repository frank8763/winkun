"""Independent history/optimizer reconstruction; zero new native decisions."""
from pathlib import Path
import sys,time,json,copy,fcntl,traceback,importlib.util
ROOT=Path(__file__).resolve().parents[3]
_spec=importlib.util.spec_from_file_location('stage434_immutable432_audit',ROOT/'stage/experiments/t432_offline_night/audit.py');A=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(A)
H=A.H

def learning_audit(C,p,name,out,s,rows,replays,guard):
    import run as producer
    E,L=C.E,C.LEARNING;r=s['result'];pp=dict(p,relay_stage=name)
    q=C.frozen(pp,r['initial_model']['path']);opt=L.make_optimizer(q,pp);H.check_model(E,q,opt,r['initial_model']);whole=E.L.state(q,opt);fixed=L.fixedhash(q)
    assert not opt.state and r['source_Adam_not_inherited']
    source,source_rows,parent=producer.stage_source(p,name)
    assert source==r['source_model'] and parent==r['parent_success_ids']
    assert q.phash()==source['policy_hash']
    panel={v['job_id']:v for v in source_rows if v['job_id'] in p['panel_ids']};assert len(panel)==len(p['panel_ids'])
    g=producer.dataset(p,q,name);dynamic=[];all_events=H.jsonl(out/'optimizer_calls.jsonl');kept=0;calls=0;rejects=0
    for rr in r['rounds']:
        guard();rnd=rr['round'];before=E.L.state(q,opt);H.check_model(E,q,opt,rr['before_model']);teacher=rr['teacher']
        if teacher:
            row=teacher['row'];arr=H.checked_array(C,row);diag=teacher['diagnostic'];aa=H.checked_array(C,diag)
            ref=C.R.load_reference(row['request']['reference_path'],row['request']['reference_sha256'])
            old=next(v for v in C.read(p['verified_target_file']) if v['id']==rr['focus_id'])
            first,why=producer.dynamic_first(aa,ref,old['first'],p)
            assert first==teacher['first']==row['request']['first'] and why==teacher['selection']
            producer.exact_prefix(arr,aa,first)
            repeat=[v for v in replays if v['role']==f'RECOVERY_EXACT_{rnd}'];assert len(repeat)==1;A.PREVIOUS.same_replay(repeat[0],row)
            assert teacher['safe_verified']==bool(row['success'] and not row['collision'])
            if teacher['safe_verified']:
                dynamic.append(dict(arrays=arr,first=first,id=f"{rr['focus_id']}:iteration{rnd}",path=row['path'],sha256=row['sha256']))
                g=producer.dataset(p,q,name,dynamic)
        assert rr['dynamic_target_paths']==[v['path'] for v in dynamic]
        for step in rr['stages']:
            tag=step['tag'];ev=[v for v in all_events if v['tag']==tag]
            count=40 if step['kind']=='bc' else 8
            assert len(ev)==4*count
            if step['kind']=='bc':
                assert teacher and teacher['safe_verified']
                fit=lambda event:L.fit_bc(q,opt,g,pp,guard,event)
            else:
                train=[v for v in rows if v['role']==f'TRAIN_{rnd}'];assert len(train)==4
                assert [v['path'] for v in train]==step['training_paths']
                assert {v['policy_hash'] for v in train}=={q.phash()}
                assert {v['loaded_model_sha256'] for v in train}=={step['sampling_model']['sha256']}
                for row in train:
                    assert row['seed'] in [p['seed_base']+rnd*4+i for i in range(4)]
                    match=next(v for v in replays if v['role']==f'TRAIN_EXACT_{rnd}' and v['seed']==row['seed']);A.PREVIOUS.same_replay(match,row)
                arrays=[H.checked_array(C,v) for v in train]
                fit=lambda event:L.fit_ppo(q,opt,arrays,g,pp,step['rnd'],guard,event)
            H.reconstruct_pair(E,q,opt,fit,step['fit'],guard,ev,tag,C.gradient_sha);calls+=2*count
            H.check_model(E,q,opt,step['after_model']);assert L.fixedhash(q)==fixed
        proposed_rows={v['job_id']:v for v in rows if v['role']==f'PANEL_{rnd}'};assert set(proposed_rows)==set(p['panel_ids'])
        prior={i for i,v in panel.items() if v['success'] and not v['collision']};after={i for i,v in proposed_rows.items() if v['success'] and not v['collision']}
        progress=bool(after-prior) or producer.score(proposed_rows[rr['focus_id']])>producer.score(panel[rr['focus_id']])
        accepted=bool(rr['stages'][-1]['fit']['numeric_accept'] and prior<=after and progress)
        assert accepted==rr['accepted'] and rr['previous_panel_success_ids']==sorted(prior) and rr['proposed_panel_success_ids']==sorted(after)
        if accepted:panel=proposed_rows;kept+=1;rejects=0
        else:E.L.load(q,opt,before);rejects+=1
        H.check_model(E,q,opt,rr['kept_model']);assert rr['consecutive_rejects']==rejects
    assert kept==r['retained_research_rounds'] and calls==r['updates']['live']+r['updates']['shadow']
    assert r['updates']['live']==r['updates']['shadow'] and calls*2==len(all_events)
    assert len(r['rounds'])<=p['rounds'][name]
    if len(r['rounds'])<p['rounds'][name]:assert rejects>=2
    accepted=False
    if kept:
        check=C.read(r['check_file']);final=[v for v in rows if v['role']=='FINAL329_'+name]
        ids,_=H.complete_development(C,p,check,final,'FINAL329_'+name,dict(path=final[0]['request']['model'],sha256=final[0]['loaded_model_sha256'],policy_hash=q.phash()))
        accepted=set(parent)<=ids and bool(ids-set(parent));assert r['development_all_safe']==bool(accepted and check['all_safe'])
        assert r['full329_executed'] and len(final)==329
    else:
        assert r['check_file'] is None and not r['full329_executed']
        assert not any(v['role'].startswith('FINAL329_') for v in rows)
    assert accepted==r['accepted'] and r['whole_stage_Actor_Adam_rollback']==(not accepted)
    if not accepted:E.L.load(q,opt,whole)
    H.check_model(E,q,opt,r['model']);assert L.fixedhash(q)==fixed
    assert r['selected_success_ids']==(check['success_ids'] if accepted else parent)
    assert r['trainable_parameters']==sum(v.numel() for v in L.params(q))
    return dict(historical_discarded_Adam=calls,live=r['updates']['live'],shadow=r['updates']['shadow'],all_Actor_Adam_gradient_events_exact=True,all_full_history_iid_densities_rewards_checked=True,teacher_only_verified_success_suffix=True,all_rejections_whole_Actor_Adam_rollback=True,full329_executed=r['full329_executed'],unexecuted_is_not_failure_rate=True)

def gate_audit(C,p,out,s,rows,replays):
    r=s['result'];name,prior=C.selected_arm(p);assert prior['development_all_safe'] and C.read(C.OUT/name/'audit.json')['state']=='PASS'
    check=C.read(prior['development_file']);assert check['all_safe'] and len(check['success_ids'])==329
    originals={v['job_id']:v for v in check['rows']};rebuilt=[v for v in rows if v['role']=='REBUILD62'];fresh=[v for v in rows if v['role']=='FRESH16'];assert len(rebuilt)==62 and len(fresh)==16
    for v in rebuilt:
        for k in ['array_digest','array_hashes','integration_sha256','physics_steps','success','collision','timeout','remaining_m','policy_hash']:assert v[k]==originals[v['job_id']][k]
    assert r['fresh16']==C.E.tally(fresh)
    good=all(v['success'] and not v['collision'] for v in fresh)
    assert r['six_gate']==good
    if good:
        assert len(replays)==78;refs={v['job_id']:v for v in rows}
        for v in replays:A.PREVIOUS.same_replay(v,refs[v['job_id']])
    else:assert not replays and r['reload78'] is None
    return dict(historical_discarded_Adam=0,full329_same_candidate_safe=True,reference62_rebuilt_exact=True,fresh16=C.E.tally(fresh),reload78_exact=len(replays),six_gate=good)

def main(name):
    import t434_core as C
    p=C.read(C.OUT/'protocol.json');C.verify(p);out=C.OUT/name;s=C.read(out/'summary.json');assert s['state']=='COMPLETE' and s['all_workers_stopped'] and s['all_sessions_closed'];assert not (out/'audit.json').exists()
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        absent=H.stopped(out);members=H.manifest(ROOT,out);assert not list(out.rglob('partial*.npz'))
        start=time.time();budget=p['stage_budgets'][name];C.E.N.offline();C.torch.set_num_threads(1);C.torch.use_deterministic_algorithms(True)
        def guard():assert time.time()<min(start+budget['audit_seconds']-5,p['deadline_all']-5)
        def no(*a,**k):raise AssertionError('T434 independent audit forbids native/new models')
        C.E.L.Session=no;C.Pool=no;C.E.Pool=no;C.E.N.S.P.C.B.setup=no;C.E.export=no;C.LEARNING.export_model=no
        calls=[0];original=C.torch.optim.Adam.step
        def counted(opt,*a,**k):
            guard();assert name in ['fresh_rl','iterative_recovery','representation_rl'];assert calls[0]<p['maximum_audit_Adam'][name];v=original(opt,*a,**k);calls[0]+=1;return v
        C.torch.optim.Adam.step=counted;C.torch.optim.AdamW.step=no;C.torch.optim.SGD.step=no
        rows=H.jsonl(out/'cases.jsonl');replays=H.jsonl(out/'replays.jsonl');commands=H.jsonl(out/'commands.jsonl')
        for row in rows:
            guard();arrays=H.checked_array(C,row);assert row['pid'] in absent
            if row['full']:A.full_policy(C,p,arrays,row)
        for row in replays:assert row['pid'] in absent and row['exact'] and row['all_arrays_exact'] and not row['independent_sample']
        if name in ['fresh_rl','iterative_recovery','representation_rl']:detail=learning_audit(C,p,name,out,s,rows,replays,guard)
        elif name=='gate':detail=gate_audit(C,p,out,s,rows,replays)
        else:
            assert C.read(C.OUT/'gate/audit.json')['six_gate'];assert len(rows)==56 and not replays
            assert s['result']['diagnostic']==C.E.tally(rows);detail=dict(historical_discarded_Adam=0,eight_training=False,seven_skipped=True)
        assert calls[0]==detail['historical_discarded_Adam']
        assert [v['id'] for v in commands]==list(range(1,len(commands)+1));assert len(commands)==s['commands']==4+len(rows)+len(replays)
        assert [(v['role'],v['physics']) for v in commands[:4]]==[('SETUP',0)]*4
        assert sorted((v['role'],v['physics'],v['pid']) for v in commands[4:])==sorted((v['role'],v['physics'],v['pid']) for v in rows+replays)
        assert sum(v['physics'] for v in commands)==s['physics']<=budget['max_physics'];assert s['commands']<=budget['max_commands'] and s['elapsed_s']<=budget['main_seconds']
        assert C.E.N.K.S.used(out)<=budget['max_bytes'];guard();C.verify(p)
        C.save(out/'audit.json',dict(state='PASS',stage=name,native=0,new_navigation_training=0,elapsed_s=time.time()-start,native_processes_absent=absent,manifest_members_before_audit=members,physics=s['physics'],commands=s['commands'],compact_development_limitation='Absent full120 dev arrays cannot be reconstructed in zero-native audit; actions, counts, flags and full array hashes retained.',**detail));C.E.N.manifest(out)
        print('T434_AUDIT_PASS',name,flush=True)
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':
    try:main(sys.argv[1])
    except BaseException:
        import t434_core as C
        f=C.OUT/sys.argv[1]/'audit_error.json'
        if not f.exists():C.save(f,dict(state='FAILED',error=traceback.format_exc(),native=0,no_restart=True))
        raise
