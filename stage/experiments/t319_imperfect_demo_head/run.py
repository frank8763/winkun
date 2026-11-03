from t319_core import *
import argparse,multiprocessing as mp

def preflight():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists()
    torch.set_num_threads(1);torch.optim.Adam.step=U.forbid_step
    q=K.policy(p).eval().requires_grad_(False);before=K.S.digest(q.state_dict());x,y=demo(p['train_demo']);phi,raw,_=features(q,x[:8])
    # Real fixed-feature head identity without assigning any candidate weights.
    assert np.max(np.abs(phi[:,:-1]@q.residual.weight.detach().numpy().T+q.residual.bias.detach().numpy()-(raw-np.stack([K.C.ParentPolicy.forward(q.base,torch.tensor(v[None,:117]).contiguous())[0].detach().numpy() for v in x[:8]]))))<3e-6
    m=np.array([[2.,.2],[.2,1.]]);v=np.array([[1.,2.],[3.,4.]])
    assert np.max(np.abs(m@np.linalg.solve(m,v)-v))<1e-12
    assert K.S.digest(q.state_dict())==before==p['policy_hash']
    rng=np.random.default_rng(319);n=280;buf=io.BytesIO()
    np.savez_compressed(buf,actions=rng.normal(size=(n,2)).astype('float32'),native_counts=rng.integers(1,101,n,dtype='uint8'),end_flags=rng.integers(0,2,(n,3),dtype='uint8'),environment_rewards=rng.normal(size=n))
    assert len(buf.getvalue())<=p['compact_bytes']
    K.C.save(OUT/'preflight.json',dict(state='PASS',native_steps=0,geometry_commands=0,Adam_calls=0,gradient_calls=0,real_parameter_proposals=0,
        real_frozen_features_and_existing_head_identity=True,synthetic_float64_solve_identity=True,initial_policy_exact=True,
        original22_geometry_hashes_reused=True,compact_incompressible_bytes=len(buf.getvalue()),protocol_sha256=K.C.sha(HERE/'protocol.json'),
        worst_reserved_global_bytes=K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']))
    print('T319 preflight PASS: fixed features/head identity,0native0Adam0candidate')

def main():
    assert not (OUT/'status.json').exists();p=read(HERE/'protocol.json');start=time.time();deadline=start+p['wall_seconds'];phase='STARTING';meter=None;workers=[];pipes=[];raw=[];replays=[];inflight=[];last=0;state='FAILED';q=None;report=None
    ctx=mp.get_context('spawn');physical=ctx.Array('q',4)
    def status(force=False):
        nonlocal last
        if not force and time.time()-last<10:return
        last=time.time();K.C.save(OUT/'status.json',dict(state=phase,pid=os.getpid(),main_epoch=start,updated_epoch=last,deadline_epoch=deadline,
            commands=0 if meter is None else meter.commands,physics_steps=sum(physical[:]),cases=len(raw),replays=len(replays),inflight=inflight,Adam_calls=0,RL_updates=0))
    def guard():assert time.time()<deadline-p['final_reserve_seconds'];meter.hold();status()
    def receive(pipe):
        while not pipe.poll(.25):status();assert time.time()<deadline-15
        item=pipe.recv();assert 'error' not in item,item;return item
    def alarm(*_):raise TimeoutError('T319 finite initialization and native validation budget')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(p['wall_seconds']-5)
    try:
        verify(p);assert read(OUT/'preflight.json')['state']=='PASS';torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.optim.Adam.step=U.forbid_step
        meter=K.Budget(ROOT,OUT,p,start_epoch=start);phase='ONE_CLOSED_FORM_INITIALIZATION';status(True)
        q,report=science(p,guard);K.C.save(OUT/'fit.json',report);blob=patch(q,p);meter.charge_payload(len(blob));(OUT/'residual_head.pt.gz').write_bytes(blob)
        for i in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(i+1);w=ctx.Process(target=worker,args=(b,i,p,physical));w.start();b.close();pipes.append(a);workers.append(w)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(v) for v in pipes];assert all(v['ready'] and v['CPU_affinity']==[i+1] for i,v in enumerate(ready))
        K.C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[w.pid for w in workers],ready=ready))
        for role,jobs in [('old_guard',p['acceptance_guard_jobs']),('six_centers',p['six_jobs'])]:
            phase=role
            for offset in range(0,len(jobs),4):
                group=[dict(j,id=role+'_'+j['id'],source_reset_id=j['id'],role=role,arm='imperfect_demo_head',round=None,mode='deterministic',seed=0,aux_eta=0.,policy_hash=report['policy_hash'],policy_np=K.S.state_np(q),deadline_epoch=deadline-p['final_reserve_seconds']) for j in jobs[offset:offset+4]]
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
        closed_form_parameter_proposals=0 if report is None else 1,Adam_calls=0,RL_updates=0,policy_hash=None if report is None else report['policy_hash'],
        patch_sha256=K.C.sha(OUT/'residual_head.pt.gz') if (OUT/'residual_head.pt.gz').exists() else None,
        human_successes_in_training=0,deployed_Actor_changed=False,old275_six54_fresh16_reload78_executed=False,six_metre_stable=False,seven_eight_entered=False,
        own_bytes=K.S.used(OUT),global_bytes=K.S.used(ROOT/'stage/runs'))
    K.C.save(OUT/'summary.json',s);K.C.save(OUT/'status.json',s);print(json.dumps(s),flush=True)

def audit():
    p=read(HERE/'protocol.json');verify(p);s=read(OUT/'summary.json');dest=DOC/'T319_AUDIT.json';assert s['state']=='COMPLETE' and not dest.exists()
    for pid in [s['pid'],read(OUT/'launch.json')['pid'],*read(OUT/'workers.json')['workers']]:
        try:os.kill(pid,0)
        except ProcessLookupError:continue
        raise AssertionError(('Original process exists',pid))
    start=time.time();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.optim.Adam.step=U.forbid_step
    def guard():assert time.time()<start+p['audit_seconds']
    q,report=science(p,guard);assert report==read(OUT/'fit.json')
    saved=torch.load(io.BytesIO(gzip.decompress((OUT/'residual_head.pt.gz').read_bytes())),map_location='cpu',weights_only=False)
    assert K.S.digest(saved['residual'])==K.S.digest(q.residual.state_dict()) and saved['policy_hash']==K.S.digest(q.state_dict())==s['policy_hash']
    assert K.C.sha(OUT/'residual_head.pt.gz')==s['patch_sha256']
    rows=[json.loads(v) for v in (OUT/'cases.jsonl').read_text().splitlines()];reps=[json.loads(v) for v in (OUT/'replays.jsonl').read_text().splitlines()]
    assert len(rows)==len(reps)==22 and counts(rows)==s['counts'] and s['commands']==44
    assert sum(r['physics_steps'] for r in rows)+sum(r['physics_steps'] for r in reps)==s['physics_steps']<=p['max_physics']
    geometries={r['id']:r for r in p['reference_geometry']}
    for r,rep in zip(rows,reps):
        guard();assert r['id']==rep['id'] and rep['all12_arrays_exact'] and rep['integration_sha256']==r['integration_sha256'] and rep['physics_steps']==r['physics_steps']
        g=geometries[r['source_reset_id']];assert r['initial120_sha256']==g['initial120_sha256'] and r['scene_sha256']==g['scene_sha256'] and r['goal_xy']==g['goal_xy']
        assert r['policy_hash']==s['policy_hash'] and outcome(r) not in ['INVALID','CENSORED'] and K.C.sha(r['path'])==r['sha256']
        with np.load(r['path']) as z:a={k:z[k].copy() for k in z.files}
        assert set(a)=={'actions','native_counts','end_flags','environment_rewards'}
        for k,v in a.items():assert K.S.digest(v)==r['array_hashes'][k]
        assert int(a['native_counts'].astype('int64').sum())==r['physics_steps'] and not a['end_flags'][:-1].any() and a['end_flags'][-1].any()
    K.C.save(dest,dict(state='PASS',native_added=0,Adam_calls=0,disposable_closed_form_reconstruction=1,
        all_head_fit_frozen_state_and_patch_policy_SHA_exact=True,all22_compact_raw_and22_full12_native_replay_receipts_verified=True,
        commands=44,physics_steps=s['physics_steps'],all_original_processes_absent=True,summary_sha256=K.C.sha(OUT/'summary.json'),
        scope='Compact4arrays checked directly; all12arrays and native integration compared in registered physical replay. No new native replay or reconstruction of unsaved middle120 in this independent audit.'))
    print('T319 independent fit/native-ledger audit PASS,0newnative0Adam')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['preflight','main','audit']);globals()[a.parse_args().mode]()
