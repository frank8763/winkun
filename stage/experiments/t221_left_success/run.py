"""Verified own-success consolidation, exact reconstruction and final old8."""
from left_support import *
import multiprocessing as mp,fcntl,shutil
class Stop(Exception):pass
def main():
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'status.json').exists(),'No duplicate T221'
    plan=json.loads((HERE/'protocol.json').read_text());start=time.monotonic();deadline=start+plan['wall_seconds'];wall=time.time()+plan['wall_seconds']
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    ctx=mp.get_context('spawn');workers=[];pipes=[];physics=commands=steps=shadow_steps=0;result=[];final=[];reason=None;reconstruction=False;checkpoints=[];shadow_checkpoints=[]
    def status(state):C.save(OUT/'status.json',dict(state=state,physics_steps=physics,commands=commands,supervised_steps=steps,shadow_supervised_steps=shadow_steps,RL_updates=0,elapsed_s=time.monotonic()-start))
    def check(pending=0,reserve=0):
        if time.monotonic()+reserve>=deadline:raise Stop('WALL_RESERVE')
        if used(OUT)+pending>=plan['final_bytes'] or used(ROOT/'stage/runs')+pending>=plan['global_bytes'] or shutil.disk_usage(ROOT).free<40*1024**3:raise Stop('STORAGE_RESERVE')
    def receive(pipe):
        while not pipe.poll(.5):
            if time.monotonic()>deadline:raise Stop('RESPONSE_DEADLINE')
        r=pipe.recv();assert 'error' not in r,r;return r
    def launch(n,temporal):
        for i in range(n):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(i+1);w=ctx.Process(target=worker,args=(b,i,wall,temporal));w.start();b.close();pipes.append(a);workers.append(w)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(pipe) for pipe in pipes];assert all(r.get('ready') for r in ready)
        C.save(OUT/('success_worker_ready.json' if temporal else 'eval_worker_ready.json'),ready)
    def stop():
        for pipe in pipes:
            try:pipe.send(None)
            except OSError:pass
        for w in workers:
            w.join(3)
            if w.is_alive():w.kill();w.join(3)
        assert all(not w.is_alive() for w in workers)
        workers.clear();pipes.clear()
    try:
        parent=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['policy'];p=S.policy(parent);ps=state_np(p);ph=digest(ps)
        with np.load(plan['retention_targets']) as z:
            x=torch.tensor(z['states']);y=torch.tensor(z['actions'])
        with torch.no_grad():initial_error=float((p(x)-y).abs().max())
        assert initial_error==0;C.save(OUT/'contracts.json',dict(state='PASS',parent_function_error=initial_error,physics_steps=0,optimizer_calls=0,protocol_sha256=C.sha(HERE/'protocol.json')))
        launch(2,True);source=plan['source_success'];assert source['policy_hash']==ph
        for i,pipe in enumerate(pipes):
            job={k:source[k] for k in ('arm','round','role','seed','mode','case','exact_snapshot')};job.update(id=f'verify_success_{i}',policy_np=ps,policy_hash=ph);pipe.send(('rollout',job))
        replay=[receive(pipe) for pipe in pipes];commands+=2;physics+=sum(r['row']['physics_steps'] for r in replay)
        for item in replay:
            row=item['row'];assert row['complete'] and row['success'] and not row['collision']
            assert row['array_hashes']==source['array_hashes'] and row['integration_sha256']==source['integration_sha256']
        assert all(np.array_equal(v,replay[1]['arrays'][k]) for k,v in replay[0]['arrays'].items())
        C.save(OUT/'success_verification.json',dict(state='PASS',source=source['id'],all10arrays_exact=True,final_integration_exact=True,physics_steps=physics,simulation_seconds=source['physics_steps']*.002,rows=[r['row'] for r in replay]))
        check(pending=120000,reserve=plan['final_reserve_seconds']);d=replay[0]['arrays'];np.savez_compressed(OUT/'verified_success.npz',states=d['states'],actions=d['actions'])
        stop();del replay,d;parent,g=groups(plan)
        def report(n,p_hash,o_hash,loss):
            nonlocal steps
            steps=n;checkpoints.append(dict(steps=n,policy_sha256=p_hash,optimizer_sha256=o_hash,loss=loss));status('SUPERVISED_CONSOLIDATION')
        live,live_opt,count=train_recipe(plan,g,plan['updates'],lambda:check(reserve=plan['final_reserve_seconds']),report)
        def shadow_report(n,p_hash,o_hash,loss):
            nonlocal shadow_steps
            shadow_steps=n;record=dict(steps=n,policy_sha256=p_hash,optimizer_sha256=o_hash,loss=loss)
            assert record==checkpoints[n//500-1];shadow_checkpoints.append(record);status('INDEPENDENT_RECONSTRUCTION')
        shadow,shadow_opt,count=train_recipe(plan,g,plan['updates'],lambda:check(reserve=plan['final_reserve_seconds']),shadow_report)
        assert digest(shadow.state_dict())==digest(live.state_dict()) and digest(shadow_opt.state_dict())==digest(live_opt.state_dict());reconstruction=True
        with torch.no_grad():fit=[float((shadow(x)-y.clamp(-.999999,.999999)).square().mean()) for x,y in g]
        recipe=dict(completed_supervised_steps=steps,checkpoint_parent=plan['checkpoint'],data_sha256=C.sha(OUT/'verified_success.npz'),final_policy_sha256=digest(shadow.state_dict()),final_optimizer_sha256=digest(shadow_opt.state_dict()),checkpoints=checkpoints,shadow_checkpoints=shadow_checkpoints,independent_reconstruction_exact=True,group_deployed_MSE=fit,software=dict(Python=sys.version,Torch=torch.__version__,NumPy=np.__version__))
        C.save(OUT/'recipe.json',recipe)
        launch(4,False);ps=state_np(shadow);ph=digest(ps)
        for off in range(0,8,4):
            check(pending=4*32*1024+16384)
            for pipe,i in zip(pipes,range(off,off+4)):
                pipe.send(('rollout',dict(id=f'final_reconstructed_{i}',arm='consolidated',round=-1,role='evaluation',seed=0,mode='deterministic',case=plan['primary_cases'][i],exact_snapshot=plan['task_snapshots'][i],policy_np=ps,policy_hash=ph)))
            batch=[receive(pipe) for pipe in pipes];commands+=len(batch);physics+=sum(r['row']['physics_steps'] for r in batch)
            final.extend([r['row'] for r in batch]);C.save(OUT/'final_old8.json',final);status('FINAL_RECONSTRUCTED_OLD8')
        reason='BUDGET_COMPLETE'
    except Stop as e:reason=str(e)
    except BaseException:reason='FAILED';C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physics_steps=physics,commands=commands))
    finally:stop()
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    summary=dict(state='FAILED' if reason=='FAILED' else 'COMPLETE',stop_reason=reason,physics_steps=physics,commands=commands,supervised_updates=steps,shadow_supervised_updates=shadow_steps,RL_updates=0,reconstruction_verified=reconstruction,final_reconstructed_old8=final,old8_safeGOAL=sum(bool(r['complete'] and r['success'] and not r['collision']) for r in final),old8_measured=len(final),automatic_promotion=False,stable_5m_gate_passed=False,all_workers_stopped=True,elapsed_s=time.monotonic()-start,own_bytes=used(OUT),all_runs_MiB=used(ROOT/'stage/runs')/1024**2,model_storage='Matched-version deterministic reconstruction from hashed parent + verified_success.npz + source recipe; no binary final checkpoint')
    C.save(OUT/'summary.json',summary);C.save(OUT/'status.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='final_reconstructed_old8'}),flush=True)
    assert physics<=plan['max_physics'] and commands<=plan['max_commands'] and used(OUT)<plan['final_bytes'] and used(ROOT/'stage/runs')<plan['global_bytes']
if __name__=='__main__':main()
