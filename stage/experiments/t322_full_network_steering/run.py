from t322_learning import *
import argparse,multiprocessing as mp

def preflight():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    q=K.policy(p).eval();opt=K.T.L.optimizer(q,p);before=K.S.digest(q.state_dict());oh=K.S.digest(opt.state_dict())
    parent=copy.deepcopy(q).requires_grad_(False);data=dataset(p,parent)
    with torch.no_grad():
        a,b=mode_raw(q,data['x']);aa,bb=mode_raw(q,data['x'],False)
        error=float((b-bb).abs().max());assert error<=3e-6
    losses=backward(p,q,data)
    norms={group:sum(float(v.grad.square().sum()) for name,v in q.named_parameters() if name.startswith(group) and v.grad is not None) for group in ['base.','encoder.','core.','residual.']}
    assert all(np.isfinite(v) and v>0 for v in norms.values()),norms
    q.zero_grad(set_to_none=True);assert K.S.digest(q.state_dict())==before and K.S.digest(opt.state_dict())==oh
    # Four disposable synthetic Adam calls: exact paired update plus full state
    # rollback. No actual navigation parameter proposal or density substitution.
    torch.manual_seed(p['seed']);m=torch.nn.Linear(3,2);n=copy.deepcopy(m)
    o=torch.optim.Adam(m.parameters());oo=torch.optim.Adam(n.parameters());initial=copy.deepcopy(m.state_dict());empty=copy.deepcopy(o.state_dict())
    for _ in range(2):
        for model,optimizer in [(m,o),(n,oo)]:
            optimizer.zero_grad();model(torch.ones(2,3)).square().mean().backward();optimizer.step()
    assert K.S.digest(m.state_dict())==K.S.digest(n.state_dict()) and K.S.digest(o.state_dict())==K.S.digest(oo.state_dict())
    m.load_state_dict(initial);o.load_state_dict(empty);assert K.S.digest(m.state_dict())==K.S.digest(initial) and K.S.digest(o.state_dict())==K.S.digest(empty)
    K.C.save(OUT/'preflight.json',dict(state='PASS',native_steps=0,geometry_commands=0,real_Actor_Adam_calls=0,synthetic_disposable_Adam_calls=4,
        real_full_objective_backward_calls=1,full_parameter_group_gradient_squared_norms=norms,batch64_vs_canonical1_raw_max_error=error,
        density_claim=False,actual_Actor_and_Adam_unchanged=True,paired_Adam_and_full_rollback_exact=True,geometry74_reused_from_T316=True,
        original_objective=losses,change_rows=int(data['changed'].sum()),protocol_sha256=K.C.sha(HERE/'protocol.json'),
        worst_reserved_global_bytes=K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']))
    print('T322 preflight PASS: 0native/0realActorAdam;4syntheticdiscardedAdam',flush=True)

def main():
    assert not (OUT/'status.json').exists();p=read(HERE/'protocol.json');start=time.time();deadline=start+p['wall_seconds'];phase='STARTING';meter=None;workers=[];pipes=[];raw=[];replays=[];inflight=[];last=0;state='FAILED';q=None;opt=None;report=None
    ctx=mp.get_context('spawn');physical=ctx.Array('q',4)
    def status(force=False):
        nonlocal last
        if not force and time.time()-last<10:return
        last=time.time();K.C.save(OUT/'status.json',dict(state=phase,pid=os.getpid(),main_epoch=start,updated_epoch=last,deadline_epoch=deadline,
            commands=0 if meter is None else meter.commands,physics_steps=sum(physical[:]),cases=len(raw),replays=len(replays),inflight=inflight,Adam_calls=optimizer_count(),RL_updates=0))
    def guard():assert time.time()<deadline-p['final_reserve_seconds'];meter.hold();status()
    def receive(pipe):
        while not pipe.poll(.25):status();assert time.time()<deadline-15
        item=pipe.recv();assert 'error' not in item,item;return item
    def alarm(*_):raise TimeoutError('T322 finite initialization and native validation budget')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(p['wall_seconds']-5)
    try:
        verify(p);assert read(OUT/'preflight.json')['state']=='PASS';torch.set_num_threads(1);torch.use_deterministic_algorithms(True);original_step=torch.optim.Adam.step
        meter=K.Budget(ROOT,OUT,p,start_epoch=start);phase='FULL_NETWORK_STEERING_INITIALIZATION';status(True)
        def fit_guard():
            guard();assert time.time()<start+p['fit_seconds'],'Initialization cutoff; preserve partial Actor/Adam'
        q,opt,report=fit(p,fit_guard,lambda e: append(OUT/'optimizer_calls.jsonl',e));K.C.save(OUT/'fit.json',report);blob=checkpoint(q,opt,p);meter.charge_payload(len(blob));(OUT/'final_checkpoint.pt.gz').write_bytes(blob)
        for i in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(i+1);w=ctx.Process(target=worker,args=(b,i,p,physical));w.start();b.close();pipes.append(a);workers.append(w)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(v) for v in pipes];assert all(v['ready'] and v['CPU_affinity']==[i+1] for i,v in enumerate(ready))
        K.C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[w.pid for w in workers],ready=ready))
        for role,jobs in [('old_guard',p['acceptance_guard_jobs']),('six_development',p['development_jobs'])]:
            phase=role
            for offset in range(0,len(jobs),4):
                group=[dict(j,id=role+'_'+j['id'],source_reset_id=j['id'],role=role,arm='full_network_steering',round=None,mode='deterministic',seed=0,aux_eta=0.,policy_hash=report['policy_hash'],policy_np=K.S.state_np(q),deadline_epoch=deadline-p['final_reserve_seconds']) for j in jobs[offset:offset+4]]
                reference=None
                for replay in [False,True]:
                    meter.checkpoint();guard();assert meter.commands+len(group)<=p['max_commands'] and sum(physical[:])+sum(j['case']['step_cap'] for j in group)<=p['max_physics']
                    meter.commands+=len(group);inflight=[j['id'] for j in group];status(True)
                    for pipe,j in zip(pipes,group):pipe.send(j)
                    items=[receive(pipe) for pipe in pipes[:len(group)]];meter.physics=sum(physical[:]);inflight=[]
                    for k,item in enumerate(items):
                        r,a=item['row'],item['arrays'];r.pop('original_job',None);r.pop('forced_first_action_diagnostic',None)
                        if 'job' in r:r['job'].pop('policy_np',None)
                        if replay:
                            old=reference[k];assert r['array_hashes']==old['row']['array_hashes'] and r['integration_sha256']==old['row']['integration_sha256']
                            assert all(np.array_equal(v,old['arrays'][key]) for key,v in a.items())
                            rep=dict(id=r['id'],physics_steps=r['physics_steps'],all12_arrays_exact=True,integration_sha256=r['integration_sha256'],not_independent_sample=True)
                            replays.append(rep)
                            with (OUT/'replays.jsonl').open('a') as f:f.write(json.dumps(rep,separators=(',',':'))+'\n')
                        else:
                            data=U.compact(a);buf=io.BytesIO();np.savez_compressed(buf,**data);blob=buf.getvalue();assert len(blob)<=p['compact_bytes'];meter.charge_payload(len(blob))
                            f=OUT/'episodes'/(r['id']+'.npz');assert not f.exists();f.write_bytes(blob);r.update(path=str(f),sha256=K.C.sha(f),saved_array_keys=list(data));raw.append(r)
                            line=json.dumps(r,allow_nan=False,separators=(',',':'))+'\n';assert len(line.encode())<=p['case_row_bytes']
                            with (OUT/'cases.jsonl').open('a') as f:f.write(line)
                    if not replay:reference=items
                    status(True);assert all(outcome(item['row']) not in ['INVALID','CENSORED'] for item in items),'Partial/invalid raw preserved; stop'
        meter.checkpoint();state='COMPLETE'
    except BaseException:
        K.C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),phase=phase,commands=0 if meter is None else meter.commands,physics_steps=sum(physical[:]),inflight=inflight,original_models_untouched=True))
    finally:
        if not (OUT/'final_checkpoint.pt.gz').exists() and HOLDER:
            try:
                blob=checkpoint(HOLDER['q'],HOLDER['opt'],p);meter.charge_payload(len(blob));(OUT/'partial_checkpoint.pt.gz').write_bytes(blob)
            except BaseException:K.C.save(OUT/'checkpoint_failure.json',dict(error=traceback.format_exc(),state_hashes=holder_hashes()))
        for pipe in pipes:
            try:pipe.send(None)
            except OSError:pass
        for w in workers:
            w.join(5)
            if w.is_alive():w.kill();w.join(3)
        signal.alarm(0)
        if meter:meter.close()
    s=dict(state=state,pid=os.getpid(),main_epoch=start,elapsed_s=time.time()-start,deadline_epoch=deadline,commands=0 if meter is None else meter.commands,
        physics_steps=sum(physical[:]),completed_cases=len(raw),exact_replays=len(replays),counts=counts(raw),all_workers_stopped=all(not w.is_alive() for w in workers),
        closed_form_parameter_proposals=0,Adam_calls=optimizer_count(),RL_updates=0,Actor_updates=optimizer_count(),policy_hash=None if report is None else report['policy_hash'],
        checkpoint_sha256=K.C.sha(OUT/'final_checkpoint.pt.gz') if (OUT/'final_checkpoint.pt.gz').exists() else None,
        human_successes_in_training=0,deployed_Actor_changed=False,six54_executed=sum(r['role']=='six_development' for r in raw)==54,old275_fresh16_reload78_executed=False,six_metre_stable=False,seven_eight_entered=False,
        own_bytes=K.S.used(OUT),global_bytes=K.S.used(ROOT/'stage/runs'))
    K.C.save(OUT/'summary.json',s);K.C.save(OUT/'status.json',s);print(json.dumps(s),flush=True)

def audit():
    p=read(HERE/'protocol.json');verify(p);s=read(OUT/'summary.json');dest=DOC/'T322_AUDIT.json'
    assert s['state']=='COMPLETE' and not dest.exists()
    for pid in [s['pid'],read(OUT/'launch.json')['pid'],*read(OUT/'workers.json')['workers']]:
        assert not Path(f'/proc/{pid}').exists(),('Live/reused producer',pid)
    start=time.time();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);events=[]
    def guard():assert time.time()-start<p['audit_seconds']
    q,opt,report=fit(p,guard,events.append);assert report==read(OUT/'fit.json')
    actual=[json.loads(v) for v in (OUT/'optimizer_calls.jsonl').read_text().splitlines()];assert events==actual
    saved=torch.load(io.BytesIO(gzip.decompress((OUT/'final_checkpoint.pt.gz').read_bytes())),map_location='cpu',weights_only=False)
    assert K.S.digest(saved['policy'])==K.S.digest(q.state_dict())==s['policy_hash']
    assert K.S.digest(saved['optimizer'])==K.S.digest(opt.state_dict())==report['optimizer_hash']
    assert K.C.sha(OUT/'final_checkpoint.pt.gz')==s['checkpoint_sha256'] and optimizer_count()==p['initialization_steps']==s['Adam_calls']
    rows=[json.loads(v) for v in (OUT/'cases.jsonl').read_text().splitlines()];reps=[json.loads(v) for v in (OUT/'replays.jsonl').read_text().splitlines()]
    assert len(rows)==len(reps)==74 and counts(rows)==s['counts'] and s['commands']==148
    assert sum(r['physics_steps'] for r in rows)+sum(r['physics_steps'] for r in reps)==s['physics_steps']<=p['max_physics']
    gs={r['id']:r for r in p['reference_geometry']}
    for r,rep in zip(rows,reps):
        guard();assert r['id']==rep['id'] and rep['all12_arrays_exact'] and rep['integration_sha256']==r['integration_sha256'] and rep['physics_steps']==r['physics_steps']
        g=gs[r['source_reset_id']];assert r['initial120_sha256']==g['initial120_sha256'] and r['scene_sha256']==g['scene_sha256'] and r['goal_xy']==g['goal_xy']
        assert r['policy_hash']==s['policy_hash'] and outcome(r) not in ['INVALID','CENSORED'] and K.C.sha(r['path'])==r['sha256']
        with np.load(r['path']) as z:a={k:z[k].copy() for k in z.files}
        assert set(a)=={'actions','native_counts','end_flags','environment_rewards'}
        for k,v in a.items():assert K.S.digest(v)==r['array_hashes'][k]
        assert int(a['native_counts'].astype('int64').sum())==r['physics_steps'] and not a['end_flags'][:-1].any() and a['end_flags'][-1].any()
    K.C.save(dest,dict(state='PASS',native_added=0,disposable_exact_historical_Adam_reconstruction=p['initialization_steps'],
        all_Actor_Adam_metrics_and200_events_exact=True,all74_compact_raw_and74_full12_native_replay_receipts_verified=True,
        commands=148,physics_steps=s['physics_steps'],all_original_processes_absent=True,summary_sha256=K.C.sha(OUT/'summary.json'),
        scope='Compact4 direct audit; full12 and integration exact during separate registered native replay. Unstored intermediate120 not reconstructed by this zero-native audit.'))
    print('T322 independent exact Actor/Adam reconstruction/native ledger PASS',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['preflight','main','audit']);globals()[parser.parse_args().mode]()
