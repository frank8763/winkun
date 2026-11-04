from t338_core import *

def worker(pipe,index,p,counts,preflight=False):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    q=Actor(p,p['arms'][0]).requires_grad_(False);folder=OUT/'workers'/str(index)
    if preflight:
        cmd=pipe.recv();q.arm=cmd['arm'];q.load_state_dict({k:torch.tensor(v) for k,v in cmd['policy_np'].items()},strict=True)
        with torch.no_grad():
            x=torch.tensor(cmd['x']);out=q.draw(x,torch.tensor(cmd['z']))
            out={k:v.numpy() for k,v in out.items()}
        pipe.send(dict(policy_hash=digest(q.state_dict()),outputs=out));pipe.close();return
    folder.mkdir(parents=True);log=N.LimitedLog(folder/'worker.log');sys.stdout=sys.stderr=log;witness=None;s=None;total=0
    try:
        torch.optim.Adam.step=lambda *_a,**_kw:(_ for _ in ()).throw(AssertionError('No native worker optimizer'))
        spec=importlib.util.spec_from_file_location('stage338_course_native',ROOT/'stage/experiments/t331_policy_prefix_retry/recorder.py')
        M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M);M.ACTIVE=read(p['course_protocol'])
        native,create,witness,reward=M.setup(folder/'native');base_factory=M.original_task_env
        def factory(create,job):
            env,obs,actual=base_factory(create,job)
            if job.get('wide_course') or job['id']==M.ACTIVE['job']['id']:obs=M.widen(env,witness)
            return env,obs,actual
        K.R.task_env=factory
        reference,_,_=M.R.load_arrays(p['reference']);course=read(p['course_protocol'])
        pipe.send(dict(ready=True,pid=os.getpid(),CPU_affinity=sorted(os.sched_getaffinity(0)),initial_policy_hash=digest(q.state_dict())))
        while True:
            job=pipe.recv()
            if job is None:break
            q.arm=job['arm'];q.load_state_dict({k:torch.tensor(v) for k,v in job['policy_np'].items()},strict=True)
            if job.get('readback_model'):
                v=torch.load(io.BytesIO(gzip.decompress(Path(job['readback_model']).read_bytes())),weights_only=False,map_location='cpu')
                q.load_state_dict(dict(q.state_dict(),**v['state']['actor']),strict=True)
            assert digest(q.state_dict())==job['policy_hash']
            if job['role']=='old':
                item=N.generate(job,q,native,create,job['deadline_epoch'],counts,index,total);total+=item['row']['physics_steps']
                item['row'].update(role='old',policy_hash=job['policy_hash'],independent_model_readback=bool(job.get('readback_model')));witness.events.clear();pipe.send(item);continue
            local=0;last=0.
            def guard():
                nonlocal last
                now=time.time();assert now<job['deadline_epoch']
                if now-last>1:assert shutil.disk_usage(ROOT).free>=p['min_free_bytes'];last=now
            def physical():
                nonlocal local
                local+=1
                if local%100==0:
                    with counts.get_lock():counts[index]=total+local
            s=M.R.Session(course,native,create,witness,reward,guard,physical)
            prefix=int(job['prefix']);assert prefix in p['frontiers']
            for k in range(prefix):s.step(reference['actions'][k],int(reference['native_counts'][k]))
            a=s.arrays()
            for name,v in a.items():
                n=prefix+1 if name.startswith('history_') else prefix
                assert np.array_equal(v,reference[name][:n]),(name,prefix)
            startsteps=s.steps;startnative=K.R.integration(s.env).copy();startx=K.T.L.packet(s.env,s.obs,s.steps,40000,s.previous,s.pn)
            assert np.array_equal(startx,reference['states'][prefix])
            if str(prefix) in p['prefix_native_references']:
                native_reference=arrays(p['prefix_native_references'][str(prefix)])
                assert np.array_equal(startnative,native_reference['initial_native_integration'])
            # The native episode, clocks, action memory and history continue.
            # Only the local evidence accumulator now stores the suffix.
            s.rows=[];s.hist=[];innovations=[];draws=[];rng=np.random.default_rng(job['seed']);source=job.get('replay_arrays');k=0
            while not(s.term or s.trunc):
                x=K.T.L.packet(s.env,s.obs,s.steps,40000,s.previous,s.pn)
                with torch.no_grad():
                    z=torch.tensor(rng.normal(size=(1,2)),dtype=torch.float64)
                    draw=q.draw(torch.tensor(x[None]),z,job['deterministic'])
                    action=draw['actions'][0].numpy()
                if source is not None:assert np.array_equal(action,source['actions'][k])
                s.step(action);innovations.append(z[0].numpy());draws.append({name:v[0].numpy() for name,v in draw.items() if name!='actions'});k+=1
            a=s.arrays();a.update(innovations=np.asarray(innovations),
                initial_native_integration=startnative,final_native_integration=K.R.integration(s.env).copy())
            a.update({name:np.asarray([d[name] for d in draws]) for name in draws[0]})
            row=dict(id=job['id'],arm=job['arm'],role=job['role'],prefix=prefix,seed=job['seed'],deterministic=job['deterministic'],
                policy_hash=job['policy_hash'],physics_steps=local,prefix_steps=startsteps,suffix_steps=s.steps-startsteps,
                complete=bool(s.term or s.trunc),success=bool(s.info.get('reach_goal')) and not bool(s.info.get('collision')),
                collision=bool(s.info.get('collision')),timeout=bool(s.trunc),remaining_m=float(s.env._goal_state()[1]),
                initial120_sha256=digest(startx),integration_sha256=K.R.arrays_hash(K.R.integration(s.env)),
                scene_sha256=s.scene,goal_xy=s.goal.tolist(),prefix_full12_and_history_exact=True,
                array_hashes={name:digest(v) for name,v in a.items()},not_original_start_autonomy=prefix!=0)
            if source is not None:assert set(a)==set(source) and all(np.array_equal(v,source[name]) for name,v in a.items())
            total+=local
            with counts.get_lock():counts[index]=total
            s.close();s=None;pipe.send(dict(row=row,arrays=a))
    except EOFError:pass
    except BaseException:
        if s is not None:
            try:
                partial=s.arrays()
                if len(locals().get('draws',[]))==len(partial['states']) and draws:
                    partial.update({name:np.asarray([d[name] for d in draws]) for name in draws[0]});partial['innovations']=np.asarray(innovations)
                partial['final_native_integration']=K.R.integration(s.env).copy()
                buf=io.BytesIO();np.savez_compressed(buf,**packed(partial));blob=buf.getvalue()
                assert len(blob)<=p['partial_bytes'];(folder/'partial.npz').write_bytes(blob)
            except BaseException:save(folder/'partial_save_failure.json',dict(error=traceback.format_exc()))
            with counts.get_lock():counts[index]=total+local
        error=dict(error=traceback.format_exc(),actual_physics=counts[index]);save(folder/'failure.json',error)
        try:pipe.send(error)
        except OSError:pass
    finally:
        if s is not None:s.close()
        if witness is not None:witness.close()
        pipe.close();log.close()
