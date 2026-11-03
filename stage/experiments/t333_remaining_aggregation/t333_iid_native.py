"""T308 rho0 generator, copied unchanged except task-local output and partial byte bound for cap40000. No original source edit."""
from t308_core import *
OUT=ROOT/'stage/runs/t333_remaining_aggregation'
def generate(job,q,native,create,witness,deadline,counts,index,previous_total):
    env=None;rows=[];hist=[];innovations=[];steps=0;error=None;term=trunc=False;info={};initial=None;integration=None;remaining=None
    try:
        env,obs,_=R.task_env(create,job);obs=C.perturb_anchor(env,obs,job)
        assert not env._collision();scene=C.model_hash(env);goal=env.goal_xy.copy();cap=job['case']['step_cap']
        previous=np.zeros(2,np.float32);pn=0;rng=np.random.default_rng(job['seed'])
        initial=T.L.packet(env,obs,0,cap,previous,0)
        if job.get('expected_initial120_sha256'):assert S.digest(initial)==job['expected_initial120_sha256']
        while not(term or trunc) and steps<cap and time.time()<deadline:
            assert shutil.disk_usage(ROOT).free>=40*1024**3
            x=T.L.packet(env,obs,steps,cap,previous,pn);pose=np.asarray(env._pose()[:3]);meta=witness.get(env)
            assert np.abs(H.rebuild(x,pose,meta[0],goal,meta[1])-x).max()<=2e-6
            with torch.no_grad():
                mean=q.step(torch.tensor(x[None]))[1][0]
                z=rng.normal(size=2)
                if not rows:z[1]+=job['first_steer_innovation_offset']
                action=T.L.AR.sample(mean,z).numpy()
                lp=float(D.terms(mean,torch.tensor(action))[1].sum())
            d0=float(env._goal_state()[1]);before=steps
            try:
                for _ in range(min(100,cap-steps)):
                    if time.time()>=deadline:break
                    obs,_,term,trunc,info=native.step_checked(env,action);steps+=1
                    if term or trunc:break
            except BaseException:error=traceback.format_exc()
            n=steps-before
            if n:
                d1=float(env._goal_state()[1]);discount=R.GAMMA**n
                terminal=term or trunc;event=float(info.get('reach_goal',False))-float(info.get('collision',False))-.5*float(trunc)
                reward=-.000002*(1-discount)/(1-R.GAMMA)+.03*d0-discount*.03*d1*(not terminal)+(R.GAMMA**(n-1)*event if terminal else 0.)
                nxt=T.L.packet(env,obs,steps,cap,action,n)
                rows.append((x,action,mean.numpy(),lp,n,nxt,pose,np.asarray(env._pose()[:3]),[info.get('collision',False),info.get('reach_goal',False),trunc],reward))
                hist.append(meta);innovations.append(z.copy());previous=action.copy();pn=n
            with counts.get_lock():counts[index]=previous_total+steps
            if error or not n:break
        if rows:hist.append(witness.get(env))
        remaining=float(env._goal_state()[1]);integration=R.arrays_hash(R.integration(env))
        assert scene==C.model_hash(env) and np.array_equal(goal,env.goal_xy)
    except BaseException:
        # Administrative/contract interruption evidence is not navigation failure.
        error=traceback.format_exc()
        if env is not None:
            partial={name:np.asarray([r[i] for r in rows]) for i,name in enumerate(NATIVE_KEYS[:10])}
            partial.update(history_poses=np.asarray([h[0] for h in hist]),history_goals=np.asarray([h[1] for h in hist]),
                history_times=np.asarray([h[2] for h in hist]),innovations=np.asarray(innovations),
                interruption_integration=R.integration(env),interruption_pose=np.asarray(env._pose()))
            buf=io.BytesIO();np.savez_compressed(buf,**partial);blob=buf.getvalue();assert len(blob)<=768*1024
            f=OUT/'workers'/str(index)/'interruption.npz';assert not f.exists();f.write_bytes(blob)
            C.save(f.with_suffix('.json'),dict(job=job,physics_steps=steps,completed_decisions=len(rows),sha256=C.sha(f),traceback=error,administrative_censor=True))
        raise
    finally:
        with counts.get_lock():counts[index]=previous_total+steps
        if env is not None:
            witness.events.pop(id(env),None);env.close()
    assert rows,'No completed decision: initialization/deadline failure, no invented episode'
    names=NATIVE_KEYS[:10];a={k:np.asarray([r[i] for r in rows]) for i,k in enumerate(names)}
    a['native_counts']=a['native_counts'].astype(np.uint8);a['end_flags']=a['end_flags'].astype(np.uint8)
    a['environment_rewards']=a['rewards'].copy();a['aux_costs']=T.endpoint_cost(a,cap)
    a.update(history_poses=np.asarray([h[0] for h in hist]),history_goals=np.asarray([h[1] for h in hist]),history_times=np.asarray([h[2] for h in hist]),innovations=np.asarray(innovations))
    row=dict(id=job['id'],family=job['family'],side=job['side'],split=job['split'],arm=job['arm'],seed=job['seed'],
        original_job=job,physics_steps=steps,complete=bool(term or trunc),success=bool(info.get('reach_goal')),
        collision=bool(info.get('collision')),timeout=bool(trunc),remaining_m=remaining,error=error,
        censored=not bool(term or trunc),policy_hash=job['policy_hash'],initial120_sha256=S.digest(initial),
        goal_xy=goal.tolist(),scene_sha256=scene,integration_sha256=integration,array_hashes={k:S.digest(v) for k,v in a.items()},
        iid_rho=0.,forced_first_action_diagnostic=job['arm']!='center',Actor_updates=0)
    return dict(row=row,arrays=a)

