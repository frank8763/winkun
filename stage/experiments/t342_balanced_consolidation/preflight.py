from t342_core import *
import multiprocessing as mp

def main():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    obj=Learner(p);initial=obj.state();groups=dataset(p,obj.actor,lambda:None)
    assert len(groups)==9 and sum(len(g['x']) for g in groups[:8])==28629
    assert np.array_equal(obj.actor.step(groups[-1]['x'])[0].detach().numpy(),arrays(p['new_success_states'])['actions'])
    tests=[];maxerr=0.
    for g in groups:
        idx=np.linspace(0,len(g['x'])-1,min(16,len(g['x'])),dtype=np.int64)
        with torch.no_grad():
            direct=obj.actor.stats(g['x'][idx],tuple(v[idx] for v in g['cache']))[0];fast=predict(obj.actor,g,idx)
        err=float((direct-fast).abs().max());assert err<=p['numerical_supervised_batch_vs_deployment_tolerance'];maxerr=max(maxerr,err);tests.append(g['x'][idx])
    clones=[]
    for replica in ['live','shadow']:
        v=Learner(p);v.load(initial);tr=fit(p,v,groups,lambda:None,lambda _:None,steps=2);clones.append((v,tr))
    assert clones[0][1]==clones[1][1] and clones[0][0].hashes()==clones[1][0].hashes()
    # A rejected/interrupted proposal restores full trainable Actor and Adam.
    v=clones[0][0];v.load(initial);assert v.hashes()=={k:digest(s) for k,s in initial.items()}
    try:
        with torch.no_grad():next(v.actor.correction.parameters()).add_(.1)
        raise InterruptedError('synthetic interrupted proposal')
    except InterruptedError:v.load(initial)
    assert v.hashes()=={k:digest(s) for k,s in initial.items()}
    # Actual spawned native-process class, using post-synthetic-update weights.
    x=torch.cat(tests);state=clones[1][0].actor.state_dict();ctx=mp.get_context('spawn');a,b=ctx.Pipe()
    shared_counts=ctx.Array('q',4)  # Keep SemLock alive until the spawned child exits.
    os.environ['STAGE_PROCESS_CPU']='1';proc=ctx.Process(target=worker,args=(b,0,p,shared_counts,True));proc.start();b.close()
    try:
        a.send(dict(states=x.numpy(),policy_np={k:v.numpy() for k,v in state.items()}));assert a.poll(120);out=a.recv()
        with torch.no_grad():expected=clones[1][0].actor.step(x)[0].numpy()
        assert out['policy_hash']==digest(state) and np.array_equal(expected,out['actions'])
    finally:proc.join(10);assert not proc.is_alive();a.close();os.environ['STAGE_PROCESS_CPU']='0'
    folder=OUT/'preflight';folder.mkdir();configs=[]
    with (folder/'factory.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        native,create=K.S.native_setup(folder/'native')
        for j in [p['old_jobs'][0],p['original_jobs'][0]]:
            env,obs,_=K.R.task_env(create,j)
            try:
                obs=K.C.perturb_anchor(env,obs,j);assert not env._collision() and float(env.max_wheel)==5.
                configs.append(dict(id=j['id'],physics_steps=0,initial120_sha256=digest(K.T.L.packet(env,obs,0,j['case']['step_cap'],np.zeros(2,np.float32),0)),
                    max_wheel=float(env.max_wheel),force_limited=env.model.actuator_forcelimited[[env.id_lr_wheel,env.id_rr_wheel]].astype(int).tolist(),
                    force_range=env.model.actuator_forcerange[[env.id_lr_wheel,env.id_rr_wheel]].tolist()))
            finally:env.close()
    meta=dict(state='PASS',physics_steps=0,geometry_commands=2,disposable_Adam_this_attempt=4,disposable_Adam_including_failed_preflight=8,old_rows=28629,new_rows=len(groups[-1]['x']),
        real_worker_actions_exact=True,native_worker_training_forbidden=True,supervised_batch_vs_actual_max_error=maxerr,
        supervised_density_not_claimed=True,initial_teacher_on_own_states_exact=True,clone_trace_sha256=digest(clones[1][1]),
        synthetic_shadow_and_reject_interrupt_Actor_Adam_exact=True,geometry=configs,protocol_sha256=K.C.sha(HERE/'protocol.json'))
    save(OUT/'preflight.json',meta);print(json.dumps(meta))

if __name__=='__main__':main()
