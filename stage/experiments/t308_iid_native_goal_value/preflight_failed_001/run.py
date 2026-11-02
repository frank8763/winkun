from t308_core import *
import argparse,builtins,multiprocessing as mp,signal,subprocess,symtable

def preflight():
    assert not any((OUT/n).exists() for n in ['preflight.json','status.json','launch.json'])
    p=read(HERE/'protocol.json');verify(p);torch.set_num_threads(1);q=actor(p)
    # Worst uncompressed payload + ZIP headers bounds real 280-decision storage.
    shapes=dict(states=(280,120),actions=(280,2),means=(280,2),logp=(280,),native_counts=(280,),next_states=(280,120),poses=(280,3),next_poses=(280,3),end_flags=(280,3),rewards=(280,),environment_rewards=(280,),aux_costs=(280,),history_poses=(281,4,3),history_goals=(281,4,2),history_times=(281,4),innovations=(280,2))
    d={k:np.zeros(s,np.float32 if k in ['states','next_states','actions','means'] else np.uint8 if k in ['native_counts','end_flags'] else np.float64) for k,s in shapes.items()}
    d['next_states'][:-1]=d['states'][1:];z=packed(d);bound=sum(v.nbytes for v in z.values())+8192;assert bound<=p['episode_bytes'],bound
    restored=unpack_full({k:v for k,v in z.items() if k in NATIVE_KEYS or k=='last_next_state'})
    assert all(np.array_equal(restored[k],d[k]) for k in NATIVE_KEYS)
    # No optimizer call: sampler/density/terminal/positive equivalence/gradient checks.
    rng=np.random.default_rng(p['seed']);m=torch.tensor(rng.normal(0,.5,(32,2)),dtype=torch.float32);a=T.L.AR.sample(m,rng.normal(size=(32,2)));_,lp=D.terms(m,a)
    assert torch.isfinite(lp).all() and bool((a.abs()<1).all())
    torch.manual_seed(p['seed']);model=Critic();x=torch.tensor(rng.normal(size=(64,120)),dtype=torch.float32);b=torch.tensor(rng.normal(size=(64,64,2)),dtype=torch.float32)
    types=torch.zeros((64,64),dtype=torch.long);eq=torch.eye(64,dtype=torch.bool);eq[0,1]=eq[1,0]=True
    data=(x,torch.tensor(rng.normal(size=(64,2)),dtype=torch.float32),b,types,eq,torch.arange(64))
    loss,_,_=scores(model,data);loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
    b1=tuple(data);a2=data[1]+.1;b2=(data[0],a2,*data[2:]);assert torch.equal(scores(model,b1,True)[0],scores(model,b2,True)[0])
    assert abs(sum(e['mass'] for e in future_law([100,25],[[0,0,0],[0,1,0]],0,28000,True))-1)<2e-12
    # Complete saved-state upper bound includes all float32 Adam moments.
    critic_uncompressed_bound=sum(v.numel()*v.element_size() for v in model.state_dict().values())*3+32768
    assert critic_uncompressed_bound<=p['model_bytes']
    for module in [sys.modules[__name__],sys.modules['t308_core']]:
        table=symtable.symtable(Path(module.__file__).read_text(),str(module.__file__),'exec');missing=set()
        def walk(t):
            for s in t.get_symbols():
                if s.is_referenced() and s.is_global() and s.get_name() not in vars(module) and not hasattr(builtins,s.get_name()):missing.add(s.get_name())
            for ch in t.get_children():walk(ch)
        walk(table);assert not missing,(module.__name__,missing)
    reserved=S.used(ROOT/'stage/runs')-S.used(OUT)+p['final_bytes'];assert reserved<=p['global_bytes']
    C.save(OUT/'preflight.json',dict(state='PASS',native_steps=0,optimizer_calls=0,Actor_updates=0,
        maximum_full_uncompressed_plus_headers=bound,critic_Adam_uncompressed_bound=critic_uncompressed_bound,
        registered_jobs=len(p['jobs']),commands_including_replay=64,full_worstcase_reserved_bytes=reserved,
        all_inputs_SHA_verified=True,policy_hash=S.digest(q.state_dict()),protocol_sha256=C.sha(HERE/'protocol.json')))
    print('T308 zero-native/optimizer preflight PASS')

def main():
    assert not (OUT/'status.json').exists();p=read(HERE/'protocol.json');start=time.time();deadline=min(start+p['wall_seconds'],p['night_absolute_end_epoch'])
    native_deadline=deadline-p['final_reserve_seconds'];counts=mp.get_context('spawn').Array('q',4);workers=[];pipes=[];meter=None
    raw=[];replays=[];inflight=[];calls=[];holder={};exports={};state='FAILED';last_status=0;phase='STARTED';report=None
    def status(name,force=False):
        nonlocal last_status,phase
        phase=name
        if not force and time.time()-last_status<10:return
        last_status=time.time();C.save(OUT/'status.json',dict(state=name,pid=os.getpid(),main_epoch=start,updated_epoch=last_status,
            absolute_deadline_epoch=deadline,native_deadline_epoch=native_deadline,commands=0 if meter is None else meter.commands,
            physics_steps=sum(counts[:]),complete_unique_cases=sum(r['complete'] for r in raw),exact_replays=len(replays),
            inflight=[j['id'] for j in inflight],head_optimizer_calls=len(calls),Actor_updates=0))
    def guard():
        assert time.time()<deadline-15;meter.hold();status(phase)
    def receive(pipe):
        while not pipe.poll(.25):status('NATIVE_ACTIVE');assert time.time()<native_deadline+5
        r=pipe.recv();assert 'error' not in r,r;return r
    def store(item,replay_partial=False):
        r=item['row'];buf=io.BytesIO();np.savez_compressed(buf,**packed(item['arrays']));b=buf.getvalue()
        assert len(b)<=p['episode_bytes'];meter.charge_payload(len(b));f=OUT/'episodes'/(r['id']+('_partial_replay' if replay_partial else '')+'.npz');assert not f.exists();f.write_bytes(b)
        r.update(path=str(f),sha256=C.sha(f));raw.append(r);C.save(OUT/'cases.json',raw)
    def event(name,step):
        assert len(calls)<2*p['head_steps'];calls.append(dict(head=name,step=step,invoked=True))
        with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps(calls[-1])+'\n')
    def alarm(*_):raise TimeoutError('T308 absolute finite pilot deadline')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(max(1,int(deadline-start)-5))
    try:
        verify(p);assert read(DOC/'T307_AUDIT.json')['state']=='PASS';torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        meter=Budget(ROOT,OUT,p,start_epoch=start);status('STARTED',True);ctx=mp.get_context('spawn')
        for i in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(i+1);w=ctx.Process(target=worker,args=(b,i,p,counts));w.start();b.close();workers.append(w);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(pipe) for pipe in pipes]
        assert all(r['ready'] and r['policy_hash']==p['policy_hash'] and r['CPU_affinity']==[i+1] for i,r in enumerate(ready))
        C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[w.pid for w in workers],ready=ready))
        by={};jobs=[dict(j,deadline_epoch=native_deadline) for j in p['jobs']]
        for replay in [False,True]:
            for offset in range(0,len(jobs),4):
                guard();meter.checkpoint();group=jobs[offset:offset+4];assert meter.commands+len(group)<=p['max_commands']
                assert sum(counts[:])+sum(j['case']['step_cap'] for j in group)<=p['max_physics']
                meter.commands+=len(group);inflight=group
                for pipe,j in zip(pipes,group):pipe.send(j)
                items=[receive(pipe) for pipe in pipes[:len(group)]];meter.physics=sum(counts[:]);inflight=[]
                for item in items:
                    r=item['row'];complete=r['complete'] and not r['error']
                    if not replay or not complete:store(item,replay)
                    if replay and complete:
                        original=by[r['id']];assert r['array_hashes']==original['array_hashes'] and r['integration_sha256']==original['integration_sha256']
                        old=arrays(original['path']);assert all(np.array_equal(v,item['arrays'][k]) for k,v in old.items())
                        r['original_id']=original['id'];replays.append(r);C.save(OUT/'replays.json',replays)
                    elif not replay:by[r['id']]=r
                    assert complete,'Partial/censored native evidence retained, no continuation'
                status('REPLAY' if replay else 'COLLECTION',True)
        for pipe in pipes:pipe.send(None)
        for w in workers:w.join(5);assert not w.is_alive()
        assert len(raw)==len(replays)==32 and sum(r['physics_steps'] for r in raw+replays)==sum(counts[:])
        status('FITTING_HEADS',True);models,opts,losses,initial=fit(p,raw,guard,event,holder)
        status('CONTROL_VALIDATION',True);report=evaluate(p,raw,models);C.save(OUT/'metrics.json',report)
        buf=io.BytesIO();np.savez_compressed(buf,losses=losses);blob=buf.getvalue();assert len(blob)<=p['evidence_bytes'];meter.charge_payload(len(blob));(OUT/'head_evidence.npz').write_bytes(blob)
        verify(p);state='COMPLETE'
    except BaseException:
        C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),phase=phase,inflight=[j['id'] for j in inflight],physics_steps=sum(counts[:]),commands=0 if meter is None else meter.commands,head_optimizer_calls=len(calls),Actor_updates=0))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except OSError:pass
        for w in workers:
            w.join(3)
            if w.is_alive():w.kill();w.join(3)
        for name,q in holder.get('models',{}).items():
            try:
                payload=dict(model=q.state_dict(),optimizer=holder['optimizers'][name].state_dict(),initial_hash=holder['initial'],steps_invoked=sum(c['head']==name for c in calls),partial=state!='COMPLETE')
                buf=io.BytesIO();torch.save(payload,buf);blob=gzip.compress(buf.getvalue(),compresslevel=6,mtime=0);assert len(blob)<=p['model_bytes'];meter.charge_payload(len(blob));f=OUT/(name+'_critic.pt.gz');f.write_bytes(blob)
                exports[name]=dict(path=str(f),sha256=C.sha(f),model_hash=S.digest(q.state_dict()),optimizer_hash=S.digest(payload['optimizer']))
            except BaseException:
                state='FAILED';C.save(OUT/(name+'_export_failure.json'),dict(traceback=traceback.format_exc()))
        signal.alarm(0)
        if meter is not None:meter.close()
    s=dict(state=state,pid=os.getpid(),commands=0 if meter is None else meter.commands,physics_steps=sum(counts[:]),
        raw_record_physics=sum(r['physics_steps'] for r in raw+replays),unique_cases=len(raw),exact_replays=len(replays),
        head_optimizer_calls=len(calls),Actor_updates=0,models=exports,all_workers_stopped=all(not w.is_alive() for w in workers),
        inflight=[j['id'] for j in inflight],main_epoch=start,absolute_deadline_epoch=deadline,native_deadline_epoch=native_deadline,elapsed_s=time.time()-start,
        own_bytes=S.used(OUT),global_bytes=S.used(ROOT/'stage/runs'),combined_value_control_gate=None if report is None else report['combined_gate'],
        six_metre_navigation_gate_evaluated=False,fresh16_executed=False,reload78_executed=False,seven_eight_entered=False,automatic_next_training=False)
    C.save(OUT/'summary.json',s);C.save(OUT/'status.json',s);print(json.dumps(s),flush=True)

def audit():
    dest=DOC/'T308_AUDIT.json';assert not dest.exists();p=read(HERE/'protocol.json');s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and s['all_workers_stopped'] and not s['inflight']
    w=read(OUT/'workers.json')
    for pid in [w['parent'],*w['workers'],read(OUT/'launch.json')['pid']]:
        try:os.kill(pid,0)
        except ProcessLookupError:continue
        raise AssertionError(('Live/reused',pid))
    start=time.time();verify(p);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);q=actor(p)
    raw=read(OUT/'cases.json');replays=read(OUT/'replays.json');by={r['id']:r for r in raw}
    assert len(raw)==len(replays)==32 and s['commands']==64 and sum(r['physics_steps'] for r in raw+replays)==s['physics_steps']==s['raw_record_physics']<=p['max_physics']
    for r in raw:
        a=arrays(r['path']);assert C.sha(r['path'])==r['sha256'];assert len(a)==16 and a['states'].shape[1]==120
        assert {k:S.digest(v) for k,v in a.items()}==r['array_hashes'] and np.array_equal(a['states'][1:],a['next_states'][:-1])
        assert int(a['native_counts'].astype(np.int64).sum())==r['physics_steps'] and not a['end_flags'][:-1].any()
        rng=np.random.default_rng(r['seed']);expected=rng.normal(size=a['innovations'].shape);expected[0,1]+=r['original_job']['first_steer_innovation_offset'];assert np.array_equal(expected,a['innovations'])
        with torch.no_grad():means=torch.cat([q.step(torch.tensor(x[None]))[1] for x in a['states']])
        assert np.array_equal(means.numpy(),a['means']) and np.array_equal(T.L.AR.sample(means,expected).numpy(),a['actions'])
        assert np.array_equal(D.terms(means,torch.tensor(a['actions']))[1].sum(-1).numpy(),a['logp'])
        points=np.concatenate([a['states'],a['next_states'][-1:]]);poses=np.concatenate([a['poses'],a['next_poses'][-1:]])
        assert len(points)==len(a['history_poses'])==len(a['history_goals'])==len(a['history_times'])
        for x,pose,hp,hg in zip(points,poses,a['history_poses'],a['history_goals']):assert np.abs(H.rebuild(x,pose,hp,r['goal_xy'],hg)-x).max()<=2e-6
        assert np.array_equal(a['rewards'],a['environment_rewards']) and np.array_equal(a['aux_costs'],T.endpoint_cost(a,28000))
        for k,n in enumerate(a['native_counts']):
            contact,goal,timeout=map(bool,a['end_flags'][k]);discount=R.GAMMA**int(n)
            d0=H.fields(a['poses'][k],r['goal_xy'])[1];d1=H.fields(a['next_poses'][k],r['goal_xy'])[1]
            reward=-.000002*(1-discount)/(1-R.GAMMA)+.03*d0-discount*.03*d1*(not(contact or goal or timeout))+(R.GAMMA**(int(n)-1)*(float(goal)-float(contact)-.5*float(timeout)) if contact or goal or timeout else 0.)
            assert abs(reward-a['environment_rewards'][k])<1e-12
    for r in replays:assert r['array_hashes']==by[r['original_id']]['array_hashes'] and r['integration_sha256']==by[r['original_id']]['integration_sha256']
    for family in sorted({r['family'] for r in raw if r['split']=='test'}):
        group=[arrays(by[family+'_'+arm]['path']) for arm in ['minus','center','plus']]
        assert all(np.array_equal(a['states'][0],group[1]['states'][0]) and np.array_equal(a['innovations'][1:min(len(a['innovations']),len(group[1]['innovations']))],group[1]['innovations'][1:min(len(a['innovations']),len(group[1]['innovations']))]) for a in group)
        assert all(a['actions'][0,0]==group[1]['actions'][0,0] for a in group)
    calls=[]
    def guard():assert time.time()<start+p['audit_seconds']
    models,opts,losses,initial=fit(p,raw,guard,lambda n,k:calls.append(dict(head=n,step=k,invoked=True)),{})
    savedcalls=[json.loads(v) for v in (OUT/'optimizer_calls.jsonl').read_text().splitlines()];assert calls==savedcalls and len(calls)==s['head_optimizer_calls']==3000
    with np.load(OUT/'head_evidence.npz') as z:assert np.array_equal(losses,z['losses'])
    assert evaluate(p,raw,models)==read(OUT/'metrics.json')
    for name,q in models.items():
        r=s['models'][name];assert C.sha(r['path'])==r['sha256'];saved=torch.load(io.BytesIO(gzip.decompress(Path(r['path']).read_bytes())),map_location='cpu',weights_only=False)
        assert S.digest(q.state_dict())==S.digest(saved['model'])==r['model_hash']
        assert S.digest(opts[name].state_dict())==S.digest(saved['optimizer'])==r['optimizer_hash'] and saved['steps_invoked']==1500 and saved['initial_hash']==initial
    verify(p);C.save(dest,dict(state='PASS',native_added=0,Actor_updates=0,disposable_head_Adam_steps=3000,
        all_original_processes_absent=True,native_commands=64,native_physics=s['physics_steps'],all32_full120_history_arrays_exact=True,
        all32_independent_native_replays_exact=True,all_iid_actions_marginal_densities_recomputed=True,all_models_Adam_losses_control_metrics_exact=True,
        combined_value_control_gate=s['combined_value_control_gate'],six_metre_navigation_gate_evaluated=False,elapsed_s=time.time()-start,summary_sha256=C.sha(OUT/'summary.json')))
    print('T308 independent native/raw/value/Adam audit PASS; navigation not evaluated')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['preflight','main','audit']);args=parser.parse_args();globals()[args.mode]()
