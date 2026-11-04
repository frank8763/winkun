from t349_core import *
import multiprocessing as mp, contextlib


def main():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.time();deadline=start+1200
    def guard():assert time.time()<deadline
    obj=Learner(p,'reset');data=dataset(p,obj.actor,guard)
    assert len(data[0])==211 and len(data[1])==1
    counts=mp.get_context('spawn').Array('q',1);tests={};calls=0;initial=digest(obj.actor.learned())
    for arm in p['arms']:
        live=Learner(p,arm);shadow=Learner(p,arm);assert digest(live.actor.learned())==initial
        for g in [data[0][0],data[1][0]]:
            with torch.no_grad():assert np.array_equal(live.actor.sequence(g['x'],g['b']).clamp(-1,1).float().numpy(),g['b'].float().numpy())
        before=live.state();source_hash=digest(before);shadow.load(before)
        a=fit(p,live,data,guard,lambda _r:None,steps=2);calls+=2
        b=fit(p,shadow,data,guard,lambda _r:None,steps=2);calls+=2
        assert a==b and live.hashes()==shadow.hashes() and digest(before)==source_hash
        assert sum(float(v.grad.abs().sum()) for v in live.actor.core.parameters())>0
        mature=live.state();unchanged=digest(mature);other=shadow.hashes();clone=Learner(p,arm);clone.load(mature)
        with torch.no_grad():next(v for st in clone.opt.state.values() for v in st.values() if isinstance(v,torch.Tensor)).add_(1.)
        assert digest(mature)==unchanged and shadow.hashes()==other
        clone.load(mature);assert clone.hashes()==live.hashes()
        with torch.no_grad():next(clone.actor.core.parameters()).add_(1.)
        clone.load(mature);assert clone.hashes()==live.hashes() and digest(mature)==unchanged
        x=data[1][0]['x'];base=data[1][0]['b'];h=None;locations=[]
        with torch.no_grad():
            for row in x:
                _,loc,new,_=live.actor.step(row[None],h);locations.append(loc[0]);h=new if live.actor.memory_on else None
            serial=torch.stack(locations);sequence=live.actor.sequence(x,base)
            maximum=float((serial-sequence).abs().max());assert maximum<1e-12
            assert float((live.actor.sequence(x[:20],base[:20])-sequence[:20]).abs().max())<1e-12
            zero=live.actor.step(x[-1:])[1];changed=live.actor.step(x[-1:],torch.ones((1,1,16),dtype=torch.float64))[1]
            influence=float((zero-changed).abs().max())
            if arm=='memory':assert influence>1e-12
            else:assert influence==0
        ctx=mp.get_context('spawn');aa,bb=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']='1'
        child=ctx.Process(target=worker,args=(bb,0,p,counts,True));child.start();bb.close()
        aa.send(dict(arm=arm,learned=live.actor.learned(),states=x[:20].numpy()))
        assert aa.poll(60);result=aa.recv();child.join(10);assert not child.is_alive() and child.exitcode==0
        assert result['policy_hash']==live.actor.phash() and np.array_equal(result['locations'],serial[:20].numpy())
        aa.close();os.environ['STAGE_PROCESS_CPU']='0'
        tests[arm]=dict(sequence_step_max_location_error=maximum,actual_worker_serial_exact=True,
            mature_snapshot_shadow_reject_interrupt_exact=True,changed_hidden_location_difference=influence,worker_pid=result['pid'])
    quota=quota_contract();folder=OUT/'preflight';folder.mkdir();geometries={}
    with (folder/'factory.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        m,native,create,w,reward=B.setup(folder/'native',p)
        try:
            for job in [*p['course_dev_jobs'],*p['original_jobs']]:
                env,obs,_=K.R.task_env(create,job)
                try:
                    obs=K.C.perturb_anchor(env,obs,job);g=B.geometry(env,obs,w,job['case']['step_cap'])
                    assert env.step_count==0 and env.max_wheel==5.;geometries[job['id']]=g
                finally:w.events.pop(id(env),None);env.close()
        finally:w.close()
    save(OUT/'geometry.json',geometries)
    n=400;integration=np.asarray(next(iter(geometries.values()))['initial_native_integration'])
    cap=dict(states=np.zeros((n,120),np.float32),actions=np.zeros((n,2),np.float32),native_counts=np.ones(n,np.uint8)*100,
             next_states=np.zeros((n,120),np.float32),poses=np.zeros((n,3)),next_poses=np.zeros((n,3)),end_flags=np.zeros((n,3),np.uint8),rewards=np.zeros(n),
             history_poses=np.zeros((n+1,4,3)),history_goals=np.zeros((n+1,4,2)),history_times=np.zeros((n+1,4)),environment_rewards=np.zeros(n),
             hidden_before=np.zeros((n,16)),last_hidden_after=np.zeros(16),initial_native_integration=integration,final_native_integration=integration)
    buff=io.BytesIO();np.savez(buff,**packed(cap));assert len(buff.getvalue())+1024<p['full_bytes']
    assert calls==p['preflight_discarded_Adam']==8 and not any(counts[:])
    save(OUT/'preflight.json',dict(state='PASS',geometry_commands=7,physics_steps=0,discarded_Adam=calls,
        trainable_parameters=sum(v.numel() for v in obj.actor.trainable()),full_uncompressed_400_decision_archive_bytes=len(buff.getvalue()),
        identical_initial_learned_parameter_sha256=initial,arms=tests,actual_old_episode_boundaries=211,
        quota_contract=quota,full_sequence_rebuilt_from_zero_no_stale_hidden=True,RL_updates=0,elapsed_s=time.time()-start,
        source_protocol_sha256=K.C.sha(HERE/'protocol.json')))
    print(json.dumps(dict(state='PREFLIGHT_PASS',geometry=7,physics=0,discarded_Adam=8,tests=tests)))

if __name__=='__main__':main()
