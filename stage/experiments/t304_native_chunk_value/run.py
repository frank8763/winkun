from t304_core import *
import multiprocessing as mp, signal, subprocess, builtins, symtable


def preflight():
    assert not any((OUT/n).exists() for n in ['preflight.json','status.json','launch.json'])
    p=read(HERE/'protocol.json');verify(p);torch.set_num_threads(1)
    q=actor(p)
    assert sum(v.numel() for v in Head().parameters())==4609
    for source in p['sources']:
        chunks=source['chunks'];a=np.asarray(chunks['center'],np.float32)
        assert a.shape==(3,2) and source['branch_decision']+3<(source['reference']['physics_steps']+99)//100
        assert np.array_equal(a[:,0],np.array(chunks['minus'],np.float32)[:,0]) and np.array_equal(a[:,0],np.array(chunks['plus'],np.float32)[:,0])
    torch.manual_seed(p['seed']);head=Head();x=torch.arange(8*126,dtype=torch.float32).reshape(8,126)/1000
    with torch.no_grad():
        altered=x.clone();altered[:,120:]=7
        assert torch.equal(head(x,True),head(altered,True))
    head(x).square().mean().backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in head.parameters())
    for mod in [sys.modules[__name__],sys.modules['t304_core']]:
        table=symtable.symtable(Path(mod.__file__).read_text(),str(mod.__file__),'exec');missing=set()
        def walk(t):
            for s in t.get_symbols():
                if s.is_referenced() and s.is_global() and s.get_name() not in vars(mod) and not hasattr(builtins,s.get_name()):missing.add(s.get_name())
            for child in t.get_children():walk(child)
        walk(table);assert not missing,(mod.__name__,missing)
    reserve=S.used(ROOT/'stage/runs')-S.used(OUT)+p['final_bytes'];assert reserve<p['global_bytes']
    C.save(OUT/'preflight.json',dict(state='PASS',native_steps=0,optimizer_calls=0,Actor_updates=0,
        policy_hash=S.digest(q.state_dict()),head_parameters_each=4609,blind_exact_invariance=True,
        full_worstcase_reserved_bytes=reserve,protocol_sha256=C.sha(HERE/'protocol.json')))
    print('T304 preflight PASS, zero native/optimizer/Actor')


def main():
    assert not (OUT/'status.json').exists()
    p=read(HERE/'protocol.json');start=time.time();deadline=start+p['wall_seconds'];native_deadline=deadline-p['final_reserve_seconds']
    counts=mp.get_context('spawn').Array('q',4);workers=[];pipes=[];meter=None;raw=[];replays=[];inflight=[];models={};opts={};exports={};calls=[];state='FAILED';last_status=0;holder={}
    def status(phase,force=False):
        nonlocal last_status
        if not force and time.time()-last_status<10:return
        last_status=time.time();C.save(OUT/'status.json',dict(state=phase,pid=os.getpid(),main_epoch=start,
            updated_epoch=last_status,absolute_deadline_epoch=deadline,native_deadline_epoch=native_deadline,
            physics_steps=sum(counts[:]),commands=0 if meter is None else meter.commands,
            complete_unique_cases=len(raw),exact_replays=len(replays),inflight=[j['id'] for j in inflight],
            head_optimizer_calls=len(calls),Actor_updates=0))
    def guard():
        assert time.time()<deadline-10;meter.checkpoint()
    def receive(pipe):
        while not pipe.poll(.25):
            status('NATIVE_ACTIVE')
            assert time.time()<native_deadline+5,'NATIVE_RESPONSE_DEADLINE'
        item=pipe.recv();assert 'error' not in item,item;return item
    def save_item(item,partial_replay=False):
        a=item['arrays'];r=item['row'];buf=io.BytesIO();np.savez_compressed(buf,**pack_full(a));blob=buf.getvalue()
        assert len(blob)<=p['episode_bytes'];meter.charge_payload(len(blob))
        f=OUT/'episodes'/(r['id']+('_partial_replay' if partial_replay else '')+'.npz');assert not f.exists();f.write_bytes(blob)
        r.update(path=str(f),sha256=C.sha(f),saved_array_keys=list(pack_full(a)))
        raw.append(r);C.save(OUT/'cases.json',raw)
        return r
    def alarm(*_):raise TimeoutError('T304 total absolute deadline')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(p['wall_seconds']-5)
    try:
        verify(p);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        meter=Budget(ROOT,OUT,p,start_epoch=start);status('STARTED',True);ctx=mp.get_context('spawn')
        for index in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(index+1)
            w=ctx.Process(target=worker,args=(b,index,p,counts));w.start();b.close();workers.append(w);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0'
        ready=[receive(pipe) for pipe in pipes];assert all(r['ready'] and r['policy_hash']==p['policy_hash'] and r['CPU_affinity']==[i+1] for i,r in enumerate(ready))
        C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[w.pid for w in workers],ready=ready))
        jobs=[]
        for source in p['sources']:
            for arm in p['arms']:
                jobs.append(dict(source['job'],id=source['id']+'_'+arm,source_reset_id=source['id'],family=source['id'],arm=arm,
                    branch_decision=source['branch_decision'],forced_chunk=source['chunks'][arm],mode='deterministic',seed=0,aux_eta=0.,deadline_epoch=native_deadline))
        by_id={}
        for replay in [False,True]:
            for off in range(0,len(jobs),4):
                guard();batch=jobs[off:off+4];assert meter.commands+len(batch)<=p['max_commands']
                assert sum(counts[:])+sum(j['case']['step_cap'] for j in batch)<=p['max_physics']
                meter.commands+=len(batch);inflight=batch
                for pipe,j in zip(pipes,batch):pipe.send(j)
                items=[receive(pipe) for pipe in pipes[:len(batch)]];meter.physics=sum(counts[:]);inflight=[]
                for item in items:
                    r=item['row'];complete=r['complete'] and not r['error']
                    if not replay or not complete:
                        save_item(item,partial_replay=replay)
                    if replay and complete:
                        original=by_id[r['id']];assert r['array_hashes']==original['array_hashes'] and r['integration_sha256']==original['integration_sha256']
                        old=arrays(original['path']);assert all(np.array_equal(v,item['arrays'][k]) for k,v in old.items())
                        r['original_id']=original['id'];replays.append(r);C.save(OUT/'replays.json',replays)
                    elif not replay:
                        by_id[r['id']]=r
                        if r['arm']=='center':
                            reference=next(s['reference'] for s in p['sources'] if s['id']==r['family'])
                            assert r['array_hashes']==reference['array_hashes'] and r['integration_sha256']==reference['integration_sha256']
                    assert complete,'Partial raw retained; no continuation/restart'
                meter.checkpoint();status('REPLAY' if replay else 'COLLECTION',True)
        assert len(raw)==len(replays)==24 and sum(r['physics_steps'] for r in raw+replays)==sum(counts[:])
        for pipe in pipes:pipe.send(None)
        for w in workers:w.join(5);assert not w.is_alive()
        data=value_data(p,raw);C.save(OUT/'value_labels.json',data['details']);status('FITTING_HEADS',True)
        def event(name,step):
            calls.append(dict(head=name,step=step));assert len(calls)<=1000
            with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps(calls[-1])+'\n')
            status('FITTING_HEADS')
        models,opts,losses,pred,initial=fit(data,p,guard,event,holder)
        report=metrics(data,pred,p);C.save(OUT/'metrics.json',report)
        buf=io.BytesIO();np.savez_compressed(buf,losses=losses,raw_x=data['raw_x'],raw_y=data['raw_y'],train=data['train'],
            action_prediction=pred['action'],blind_prediction=pred['blind'],**data['normalization'])
        blob=buf.getvalue();assert len(blob)<p['evidence_bytes'];meter.charge_payload(len(blob));(OUT/'head_evidence.npz').write_bytes(blob)
        for name,q in models.items():
            payload=dict(model=q.state_dict(),optimizer=opts[name].state_dict(),steps=500,initial_hash=initial,normalization=data['normalization'])
            buf=io.BytesIO();torch.save(payload,buf);blob=gzip.compress(buf.getvalue(),compresslevel=6,mtime=0)
            assert len(blob)<p['model_bytes'];meter.charge_payload(len(blob));f=OUT/(name+'_value_head.pt.gz');f.write_bytes(blob)
            exports[name]=dict(path=str(f),sha256=C.sha(f),model_hash=S.digest(q.state_dict()),optimizer_hash=S.digest(opts[name].state_dict()))
        verify(p);state='COMPLETE'
    except BaseException:
        C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),inflight_jobs=[j['id'] for j in inflight],
            real_physics=sum(counts[:]),commands=0 if meter is None else meter.commands,head_optimizer_calls=len(calls)))
        # Preserve failed head/Adam states as well as their calls and native data.
        for name,q in holder.get('models',{}).items():
            f=OUT/(name+'_value_head.pt.gz')
            if f.exists():continue
            try:
                payload=dict(model=q.state_dict(),optimizer=holder['optimizers'][name].state_dict(),
                    steps_invoked=sum(c['head']==name for c in calls),partial_or_interrupted=True,
                    initial_hash=holder['initial'],normalization=data['normalization'])
                buf=io.BytesIO();torch.save(payload,buf);blob=gzip.compress(buf.getvalue(),compresslevel=6,mtime=0)
                assert len(blob)<p['model_bytes'];meter.charge_payload(len(blob));f.write_bytes(blob)
                exports[name]=dict(path=str(f),sha256=C.sha(f),model_hash=S.digest(q.state_dict()),
                    optimizer_hash=S.digest(payload['optimizer']),partial_or_interrupted=True)
            except BaseException:C.save(OUT/(name+'_partial_export_failure.json'),dict(traceback=traceback.format_exc()))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except OSError:pass
        for w in workers:
            w.join(3)
            if w.is_alive():w.kill();w.join(3)
        signal.alarm(0)
        if meter is not None:meter.close()
    result=dict(state=state,pid=os.getpid(),commands=0 if meter is None else meter.commands,physics_steps=sum(counts[:]),
        unique_cases=len(raw),exact_replays=len(replays),raw_record_physics=sum(r['physics_steps'] for r in raw+replays),
        Actor_updates=0,head_optimizer_calls=len(calls),models=exports,all_workers_stopped=all(not w.is_alive() for w in workers),
        inflight_jobs=[j['id'] for j in inflight],main_epoch=start,absolute_deadline_epoch=deadline,elapsed_s=time.time()-start,
        own_bytes=S.used(OUT),global_bytes=S.used(ROOT/'stage/runs'),automatic_next_training=False,
        screen=read(OUT/'metrics.json')['screen'] if (OUT/'metrics.json').exists() else None,limitation=p['contract'])
    C.save(OUT/'summary.json',result);C.save(OUT/'status.json',result);print(json.dumps(result),flush=True)


def audit():
    dest=DOC/'T304_AUDIT.json';assert not dest.exists();p=read(HERE/'protocol.json');s=read(OUT/'summary.json')
    assert s['state']=='COMPLETE' and s['all_workers_stopped'] and not s['inflight_jobs']
    w=read(OUT/'workers.json')
    for pid in [w['parent'],*w['workers'],read(OUT/'launch.json')['pid']]:
        try:os.kill(pid,0)
        except ProcessLookupError:continue
        raise AssertionError(('Live/reused',pid))
    start=time.time();verify(p);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);q=actor(p)
    raw,replays=read(OUT/'cases.json'),read(OUT/'replays.json');by={r['id']:r for r in raw}
    assert len(raw)==len(replays)==24 and len(raw)+len(replays)==s['commands']==48
    assert sum(r['physics_steps'] for r in raw+replays)==s['physics_steps']==s['raw_record_physics']<=p['max_physics']
    semantics=[]
    for source in p['sources']:
        center=by[source['id']+'_center'];baseline=arrays(center['path']);b=source['branch_decision']
        assert center['array_hashes']==source['reference']['array_hashes'] and center['integration_sha256']==source['reference']['integration_sha256']
        for arm in p['arms']:
            r=by[source['id']+'_'+arm];a=arrays(r['path']);assert C.sha(r['path'])==r['sha256']
            assert len(a)==12 and a['states'].shape[1]==120 and all(S.digest(v)==r['array_hashes'][k] for k,v in a.items())
            assert np.array_equal(a['states'][1:],a['next_states'][:-1]) and int(a['native_counts'].astype(np.int64).sum())==r['physics_steps']
            assert np.array_equal(a['rewards'],a['environment_rewards']) and np.array_equal(a['aux_costs'],T.endpoint_cost(a,28000))
            for key in ['states','actions','native_counts','next_states','poses','next_poses','end_flags','environment_rewards']:
                assert np.array_equal(a[key][:b],baseline[key][:b]),(source['id'],arm,key)
            assert np.array_equal(a['states'][b],baseline['states'][b])
            with torch.no_grad():expected=np.concatenate([q.step(torch.tensor(x[None]))[0].numpy() for x in a['states']])
            stop=min(len(a['actions']),b+3)
            assert np.array_equal(a['actions'][b:stop],np.asarray(source['chunks'][arm],np.float32)[:stop-b])
            assert np.array_equal(a['actions'][:b],expected[:b]) and np.array_equal(a['actions'][stop:],expected[stop:])
            assert np.all(a['logp']==0) and np.all(a['means']==0)
            semantics.append(dict(id=r['id'],initial_branch120_exact=True,prefix_exact=True,all_after_chunk_actions_exact_Actor=True,executed_chunk_decisions=stop-b))
    for r in replays:
        ref=by[r['original_id']];assert r['array_hashes']==ref['array_hashes'] and r['integration_sha256']==ref['integration_sha256']
    data=value_data(p,raw);count=[]
    def guard():assert time.time()<start+p['audit_seconds']
    models,opts,losses,pred,initial=fit(data,p,guard,lambda name,step:count.append((name,step)))
    assert len(count)==s['head_optimizer_calls']==1000
    with np.load(OUT/'head_evidence.npz') as z:
        for key,a in dict(losses=losses,raw_x=data['raw_x'],raw_y=data['raw_y'],train=data['train'],action_prediction=pred['action'],blind_prediction=pred['blind'],**data['normalization']).items():assert np.array_equal(z[key],a),key
    assert read(OUT/'value_labels.json')==data['details'] and read(OUT/'metrics.json')==metrics(data,pred,p)
    for name,model in models.items():
        record=s['models'][name];assert C.sha(record['path'])==record['sha256']
        saved=torch.load(io.BytesIO(gzip.decompress(Path(record['path']).read_bytes())),map_location='cpu',weights_only=False)
        assert S.digest(model.state_dict())==S.digest(saved['model'])==record['model_hash']
        assert S.digest(opts[name].state_dict())==S.digest(saved['optimizer'])==record['optimizer_hash']
        assert saved['initial_hash']==initial and saved['steps']==500
    verify(p);C.save(dest,dict(state='PASS',native_added=0,Actor_updates=0,disposable_head_Adam_steps=1000,
        native_commands=s['commands'],native_physics=s['physics_steps'],all_original_processes_absent=True,
        all24_full120_arrays_and_replays_verified=True,all8_parent_controls_exact_T299=True,semantics=semantics,
        all_head_Adam_losses_labels_predictions_exact=True,screen=s['screen'],elapsed_s=time.time()-start,summary_sha256=C.sha(OUT/'summary.json')))
    print('T304 independent audit PASS, no new native or Actor update')


if __name__=='__main__':
    if sys.argv[1:]==['--preflight']:preflight()
    elif sys.argv[1:]==['--audit']:audit()
    else:assert not sys.argv[1:];main()
