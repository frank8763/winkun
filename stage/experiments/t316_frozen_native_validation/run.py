from t316_core import *
import argparse,multiprocessing as mp

def preflight():
    assert not any((OUT/n).exists() for n in ['preflight.json','launch.json','status.json'])
    p=read(HERE/'protocol.json');verify(p);start=time.time();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.optim.Adam.step=forbid_step
    qs=policies(p);before={k:K.S.digest(q.state_dict()) for k,q in qs.items()};folder=OUT/'preflight';folder.mkdir()
    with (folder/'factory.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        native,create=K.S.native_setup(folder);geometry=[]
        for j in p['acceptance_guard_jobs']+p['development_jobs']:
            assert time.time()<start+p['preflight_seconds'];env=None
            try:
                env,obs,_=K.R.task_env(create,j);obs=K.C.perturb_anchor(env,obs,j);assert not env._collision()
                x=K.T.L.packet(env,obs,0,j['case']['step_cap'],np.zeros(2,np.float32),0)
                if j.get('expected_initial120_sha256'):assert K.S.digest(x)==j['expected_initial120_sha256']
                integration=K.R.arrays_hash(K.R.integration(env))
                actions={}
                with torch.no_grad():
                    for name,q in qs.items():
                        action=q.step(torch.tensor(x[None]))[0][0];assert torch.isfinite(action).all() and bool((action.abs()<=1).all());actions[name]=action.tolist()
                assert integration==K.R.arrays_hash(K.R.integration(env))
                geometry.append(dict(id=j['id'],initial120_sha256=K.S.digest(x),integration_sha256=integration,scene_sha256=K.C.model_hash(env),goal_xy=env.goal_xy.tolist(),valid=True,physics_steps=0,first_actions=actions))
            finally:
                if env is not None:env.close()
    assert len(geometry)==74 and before=={k:K.S.digest(q.state_dict()) for k,q in qs.items()}
    # Worst-case incompressible compact arrays for each registered episode cap.
    rng=np.random.default_rng(316);sizes={}
    for cap,limit in [(23334,p['old_compact_bytes']),(28000,p['new_compact_bytes'])]:
        n=(cap+99)//100;a=dict(actions=rng.normal(size=(n,2)).astype(np.float32),native_counts=rng.integers(1,101,size=n,dtype=np.uint8),end_flags=rng.integers(0,2,size=(n,3),dtype=np.uint8),environment_rewards=rng.normal(size=n))
        b=io.BytesIO();np.savez_compressed(b,**a);assert len(b.getvalue())<=limit;sizes[str(cap)]=len(b.getvalue())
    K.C.save(OUT/'geometry.json',geometry)
    K.C.save(OUT/'preflight.json',dict(state='PASS',geometry_commands=74,physics_steps=0,Adam_calls=0,gradient_calls=0,all_four_frozen_models_and_first_actions_checked=True,
        original_native_initial_history_unchanged=True,compact_worst_incompressible_bytes=sizes,protocol_sha256=K.C.sha(HERE/'protocol.json'),full_worstcase_reserved_bytes=K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes'],elapsed_s=time.time()-start))
    print('T316 preflight PASS:74 geometry,0physics/Adam/gradient,4fixedActors')

def main():
    assert not (OUT/'status.json').exists();p=read(HERE/'protocol.json');start=time.time();deadline=start+p['wall_seconds'];phase='STARTING';state='FAILED';reason='STARTING';meter=None;workers=[];pipes=[];raw=[];inflight=[];last=0;geometry=read(OUT/'geometry.json')
    ctx=mp.get_context('spawn');physical=ctx.Array('q',4);current=None;torch.optim.Adam.step=forbid_step
    def status(force=False):
        nonlocal last
        if not force and time.time()-last<10:return
        last=time.time();K.C.save(OUT/'status.json',dict(state=phase,pid=os.getpid(),main_epoch=start,deadline_epoch=deadline,updated_epoch=last,commands=74+(0 if meter is None else meter.commands),physics_steps=sum(physical[:]),completed_cases=len(raw),inflight=inflight,candidate=current,Adam_calls=0))
    def receive(pipe):
        while not pipe.poll(.25):status();assert time.time()<deadline-20
        r=pipe.recv();assert 'error' not in r,r;return r
    def alarm(*_):raise TimeoutError('T316 absolute frozen evaluation budget')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(p['wall_seconds']-5)
    try:
        verify(p);assert read(OUT/'preflight.json')['state']=='PASS';torch.set_num_threads(1);torch.use_deterministic_algorithms(True);meter=K.Budget(ROOT,OUT,p,start_epoch=start);qs=policies(p)
        for i in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(i+1);w=ctx.Process(target=worker,args=(b,i,p,physical));w.start();b.close();pipes.append(a);workers.append(w)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(pipe) for pipe in pipes];assert all(r['ready'] and r['CPU_affinity']==[i+1] for i,r in enumerate(ready))
        K.C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[w.pid for w in workers],ready=ready));status(True)
        for name,q in qs.items():
            current=name
            for role,key in [('old_guard','acceptance_guard_jobs'),('six_development','development_jobs')]:
                phase=role
                for off in range(0,len(p[key]),4):
                    group=[]
                    for base in p[key][off:off+4]:
                        j=dict(base,id=name+'_'+role+'_'+base['id'],source_reset_id=base['id'],role=role,arm=name,round=None,mode='deterministic',seed=0,aux_eta=0.,policy_hash=p['candidates'][name]['policy_hash'],deadline_epoch=deadline-p['final_reserve_seconds'],policy_np=K.S.state_np(q));group.append(j)
                    meter.checkpoint();assert 74+meter.commands+len(group)<=p['max_commands'];assert sum(physical[:])+sum(j['case']['step_cap'] for j in group)<=p['max_physics'];assert time.time()<deadline-p['final_reserve_seconds']
                    meter.commands+=len(group);inflight=[j['id'] for j in group];status(True)
                    for pipe,j in zip(pipes,group):pipe.send(j)
                    batch=[receive(pipe) for pipe in pipes[:len(group)]];meter.physics=sum(physical[:]);inflight=[]
                    for item in batch:
                        r=item['row'];a=item['arrays'];r.pop('original_job',None);r.pop('forced_first_action_diagnostic',None)
                        if 'job' in r:r['job'].pop('policy_np',None)
                        data=compact(a);b=io.BytesIO();np.savez_compressed(b,**data);blob=b.getvalue();limit=p['old_compact_bytes'] if role=='old_guard' else p['new_compact_bytes'];assert len(blob)<=limit
                        meter.charge_payload(len(blob));f=OUT/'episodes'/(r['id']+'.npz');assert not f.exists();f.write_bytes(blob);r.update(path=str(f),sha256=K.C.sha(f),saved_array_keys=list(data));raw.append(r)
                        line=json.dumps(r,allow_nan=False,separators=(',',':'))+'\n';assert len(line.encode())<=p['case_row_bytes']
                        with (OUT/'cases.jsonl').open('a') as log:log.write(line)
                    status(True);assert all(outcome(x['row']) not in ['INVALID','CENSORED'] for x in batch),'Invalid/partial preserved; no continuation'
                K.C.save(OUT/'counts.json',counts(raw,p))
            assert K.S.digest(q.state_dict())==p['candidates'][name]['policy_hash']
        meter.checkpoint();state='COMPLETE';reason='ALL4_FIXED_CANDIDATES_EVALUATED'
    except BaseException:
        reason='ENGINEERING_OR_BUDGET_FAILURE_'+phase;K.C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),phase=phase,inflight=inflight,completed_cases=len(raw),commands=74+(0 if meter is None else meter.commands),physics_steps=sum(physical[:])))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except OSError:pass
        for w in workers:
            w.join(5)
            if w.is_alive():w.kill();w.join(3)
        signal.alarm(0)
        if meter:meter.close()
    s=dict(state=state,stop_reason=reason,pid=os.getpid(),main_epoch=start,elapsed_s=time.time()-start,deadline_epoch=deadline,commands=74+(0 if meter is None else meter.commands),geometry_commands=74,native_commands=0 if meter is None else meter.commands,physics_steps=sum(physical[:]),raw_record_physics=sum(r['physics_steps'] for r in raw),completed_cases=len(raw),inflight=inflight,counts=counts(raw,p),all_workers_stopped=all(not w.is_alive() for w in workers),Adam_calls=0,gradient_calls=0,deployed_Actor_changed=False,old275_fresh16_reload78_executed=False,six_metre_stable=False,seven_eight_entered=False,automatic_training=False,own_bytes=K.S.used(OUT),global_bytes=K.S.used(ROOT/'stage/runs'))
    K.C.save(OUT/'summary.json',s);K.C.save(OUT/'status.json',s);print(json.dumps(s),flush=True)

def audit():
    dest=DOC/'T316_AUDIT.json';assert not dest.exists();p=read(HERE/'protocol.json');s=read(OUT/'summary.json');start=time.time();assert s['state']=='COMPLETE' and s['all_workers_stopped'] and not s['inflight']
    w=read(OUT/'workers.json')
    for pid in [s['pid'],read(OUT/'launch.json')['pid'],*w['workers']]:
        try:os.kill(pid,0)
        except ProcessLookupError:continue
        raise AssertionError(('Live or reused PID',pid))
    verify(p);torch.optim.Adam.step=forbid_step;qs=policies(p);rows=[json.loads(v) for v in (OUT/'cases.jsonl').read_text().splitlines()];geometry={g['id']:g for g in read(OUT/'geometry.json')}
    assert len(rows)==296 and len(geometry)==74 and s['commands']==370 and s['native_commands']==296 and s['physics_steps']==s['raw_record_physics']==sum(r['physics_steps'] for r in rows)<=p['max_physics']
    saved=read(OUT/'counts.json');assert saved==s['counts']==counts(rows,p)
    for r in rows:
        assert time.time()<start+p['audit_seconds'];f=Path(r['path']);assert K.C.sha(f)==r['sha256'];assert r['policy_hash']==p['candidates'][r['arm']]['policy_hash']
        assert r['complete'] and not r.get('error');g=geometry[r['source_reset_id']];assert r['initial120_sha256']==g['initial120_sha256'] and r['scene_sha256']==g['scene_sha256'] and r['goal_xy']==g['goal_xy']
        with np.load(f) as z:a={k:z[k].copy() for k in z.files}
        assert set(a)=={'actions','native_counts','end_flags','environment_rewards'}
        for k,v in a.items():assert K.S.digest(v)==r['array_hashes'][k]
        assert int(a['native_counts'].astype(np.int64).sum())==r['physics_steps'] and not a['end_flags'][:-1].any() and a['end_flags'][-1].any()
        flags=a['end_flags'][-1];assert tuple(map(bool,flags))==(r['collision'],r['success'],r['timeout'])
        assert r['mode']=='deterministic' and r['aux_eta']==0 and r['seed']==0 and len(r['integration_sha256'])==64
        base=next(j for j in p['acceptance_guard_jobs' if r['role']=='old_guard' else 'development_jobs'] if j['id']==r['source_reset_id'])
        for key,v in base.items():
            if key not in ['id','source_reset_id','mode','seed']:assert r['job'][key]==v
        assert r['job']['case']['step_cap']>=r['physics_steps']
    assert all(v['executed']==v['planned'] and v['outcomes']['INVALID']==v['outcomes']['CENSORED']==0 for v in saved)
    K.C.save(dest,dict(state='PASS',native_added=0,Adam_calls=0,gradient_calls=0,all4_frozen_input_models_source_SHA_exact=True,all296_compact_arrays_and_native_counts_terminal_flags_exact=True,
        all74_original_native_initial_and_scene_goal_hashes_match=True,counts=saved,commands=370,physics_steps=s['physics_steps'],all_processes_absent=True,
        scope='Compact4arrays directly checked, full12array and terminalintegration hashes recorded by native producer; no new physical replay/full intermediate120 reconstruction in this audit.',summary_sha256=K.C.sha(OUT/'summary.json'),elapsed_s=time.time()-start))
    print('T316 independent compact/raw accounting audit PASS,0newnative0Adam')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['preflight','main','audit']);a=parser.parse_args();globals()[a.mode]()
