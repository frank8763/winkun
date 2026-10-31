from induced_support import *
import multiprocessing as mp,fcntl,shutil
class Stop(Exception):pass
def main():
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'status.json').exists(),'No duplicateT243'
    plan=json.loads((HERE/'protocol.json').read_text());start=time.monotonic();deadline=start+plan['wall_seconds'];wall=time.time()+plan['wall_seconds'];physics=commands=updates=shadow_updates=0;workers=[];pipes=[];rows=[];hold=[];validity=[];checks=[];shadow_checks=[];export=None;reason=None;reconstruction=False;hjobs=[]
    safe=lambda r:r.get('valid_start',True) and r['complete'] and r['success'] and not r['collision']
    def status(state):C.save(OUT/'status.json',dict(state=state,physics_steps=physics,commands=commands,supervised_updates=updates,shadow_updates=shadow_updates,RL_updates=0,development_cases=len(rows),heldout_cases=len(hold),heldout_validity_cases=len(validity),elapsed_s=time.monotonic()-start))
    def guard(n=0,count=0,pending=0,reserve=0):
        if time.monotonic()+reserve>=deadline:raise Stop('WALL_RESERVE')
        if commands+n>plan['max_commands'] or physics+count>plan['max_physics']:raise Stop('COUNT_RESERVE')
        if used(OUT)+pending>=plan['final_bytes'] or used(ROOT/'stage/runs')+pending>=plan['global_bytes'] or shutil.disk_usage(ROOT).free<40*1024**3:raise Stop('STORAGE_RESERVE')
    def receive(pipe):
        while not pipe.poll(.5):
            if time.monotonic()>deadline:raise Stop('RESPONSE_DEADLINE')
        r=pipe.recv();assert 'error' not in r,r;return r
    def append(name,r):
        with (OUT/name).open('a') as f:f.write(json.dumps(r)+'\n')
    def evaluate(jobs,name):
        nonlocal physics,commands
        for off in range(0,len(jobs),4):
            batch=jobs[off:off+4];guard(len(batch),sum(j['case']['step_cap'] for j in batch),16384)
            for pipe,j in zip(pipes,batch):pipe.send(('evaluate',dict(j,policy_np=ps,policy_hash=ph,arm='frozenT243',seed=0,mode='deterministic')))
            replies=[receive(p) for p in pipes[:len(batch)]];commands+=len(replies);physics+=sum(i['row']['physics_steps'] for i in replies)
            for j,item in zip(batch,replies):
                r=item['row'];r.update(valid_start=True,case=j['case'],delta_x_m=j['delta_x_m'],delta_y_m=j['delta_y_m'],delta_yaw_deg=j['delta_yaw_deg']);append(name,r)
                (rows if name=='development.jsonl' else hold).append(r)
            status('FROZEN_'+name)
            if not all(i['row']['complete'] for i in replies):raise Stop('CENSORED_CASE')
    try:
        for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);p=initial(plan);assert digest(p.state_dict())==plan['initial_policy_hash'];g=groups(plan);assert len(g)==11
        C.save(OUT/'contracts.json',dict(state='PASS',initial_function_hash=plan['initial_policy_hash'],initial_function_exact=True,groups=[dict(n=len(x),states_hash=digest(x),targets_hash=digest(y)) for x,y in g],physics_steps=0,optimizer_calls=0))
        ctx=mp.get_context('spawn')
        for i in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(i+1);w=ctx.Process(target=worker,args=(b,i,wall));w.start();b.close();pipes.append(a);workers.append(w)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(p) for p in pipes];assert all(r.get('ready') for r in ready);C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[w.pid for w in workers]))
        # Holdout labels/rollouts are never consulted in fitting; initial geometry
        # validity is policy-independent and locked before optimizer calls.
        for off in range(0,64,4):
            batch=plan['heldout_jobs'][off:off+4];guard(len(batch),0,8192,reserve=plan['final_reserve_seconds'])
            for pipe,j in zip(pipes,batch):pipe.send(('validate',j))
            rr=[receive(p) for p in pipes[:len(batch)]];commands+=len(rr)
            for j,r in zip(batch,rr):
                validity.append(r);append('heldout_validity.jsonl',r)
                if r['valid_start']:hjobs.append(j)
        def report(n,ph,oh,loss):
            nonlocal updates
            updates=n;checks.append(dict(step=n,policy_sha256=ph,optimizer_sha256=oh,loss=loss));status('INDUCED_STATE_SELF_REPAIR')
        live,opt=train(plan,g,report,lambda:guard(pending=2500000,reserve=plan['final_reserve_seconds']))
        def shadow_report(n,ph,oh,loss):
            nonlocal shadow_updates
            shadow_updates=n;r=dict(step=n,policy_sha256=ph,optimizer_sha256=oh,loss=loss);assert r==checks[n//500-1];shadow_checks.append(r);status('INDEPENDENT_RECONSTRUCTION')
        shadow,sopt=train(plan,g,shadow_report,lambda:guard(pending=2500000,reserve=plan['final_reserve_seconds']))
        assert digest(live.state_dict())==digest(shadow.state_dict()) and digest(opt.state_dict())==digest(sopt.state_dict());reconstruction=True
        export=S.save_gzip(OUT/'final_Actor.pt.gz',dict(policy=shadow.state_dict(),source='T243currentinducedstateretentionandownsuccess24',protocol_sha256=C.sha(HERE/'protocol.json'),RL_updates=0),lambda n:guard(pending=n))
        state=torch.load(io.BytesIO(gzip.decompress(Path(export['path']).read_bytes())),map_location='cpu',weights_only=False)['policy'];assert digest(state)==digest(shadow.state_dict());ps={k:v.numpy().copy() for k,v in state.items()};ph=digest(state)
        C.save(OUT/'recipe.json',dict(checkpoints=checks,shadow_checkpoints=shadow_checks,independent_reconstruction_exact=True,final_policy_sha256=ph,final_optimizer_sha256=digest(sopt.state_dict()),export=export,software=dict(Python=sys.version,Torch=torch.__version__,NumPy=np.__version__)))
        fit=[]
        with torch.no_grad():
            for i,(gx,gy) in enumerate(g):
                err=C.ParentPolicy.forward(shadow,gx)-gy
                fit.append(dict(group=i,states=len(gx),preclip_RMSE=float(err.square().mean().sqrt()),preclip_max=float(err.abs().max())))
        C.save(OUT/'fit_diagnostics.json',dict(groups=fit,not_a_navigation_gate=True))
        evaluate(plan['development_jobs'],'development.jsonl')
        if len(rows)==len(plan['development_jobs']) and all(safe(r) for r in rows):evaluate(hjobs,'heldout.jsonl');reason='DEVELOPMENT_AND_VALID_HELDOUT_COMPLETE'
        else:reason='DEVELOPMENT_GATE_FAILED_HELDOUT_NOT_RUN'
    except Stop as e:reason=str(e)
    except BaseException:reason='FAILED';C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physics_steps=physics,commands=commands,inflight_new_physics_upper_bound=4*23334))
    finally:
        for p in pipes:
            try:p.send(None)
            except OSError:pass
        for w in workers:
            w.join(3)
            if w.is_alive():w.kill();w.join(3)
        assert all(not w.is_alive() for w in workers)
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    scores={}
    for i in range(8):
        rr=[r for r in rows if r['group']==f'task{i}'];hh=[r for r in hold if r['group']==f'task{i}'];scores[f'task{i}']=dict(development_n=len(rr),development_safeGOAL=sum(safe(r) for r in rr),CONTACT=sum(bool(r['collision']) for r in rr),timeout=sum(bool(r['timeout']) for r in rr),heldout_n=len(hh),heldout_safeGOAL=sum(safe(r) for r in hh))
    centers=[r for r in rows if r['delta_x_m']==r['delta_y_m']==r['delta_yaw_deg']==0.];gate=reconstruction and len(rows)==len(plan['development_jobs']) and all(safe(r) for r in rows) and len(hold)==len(hjobs)>0 and all(safe(r) for r in hold) and len(centers)==8 and all(safe(r) for r in centers)
    summary=dict(state='FAILED' if reason=='FAILED' else 'COMPLETE',stop_reason=reason,scores=scores,original_centers_n=len(centers),original_centers_safeGOAL=sum(safe(r) for r in centers),development_n=len(rows),development_safeGOAL=sum(safe(r) for r in rows),development_invalid_starts_predeclared=5,heldout_requested=64,heldout_valid=len(hjobs),heldout_invalid_ids=[r['id'] for r in validity if not r['valid_start']],heldout_n=len(hold),heldout_safeGOAL=sum(safe(r) for r in hold),supervised_updates=updates,shadow_updates=shadow_updates,RL_updates=0,reconstruction_verified=reconstruction,export=export,physics_steps=physics,commands=commands,stable_valid_local_3m_4m_5m_gate=gate,original_T228_strict_gate=False,multimap_generalization_claim=False,automatic_promotion=False,all_workers_stopped=True,elapsed_s=time.monotonic()-start,own_bytes=used(OUT),runs_MiB=used(ROOT/'stage/runs')/1024**2)
    C.save(OUT/'summary.json',summary);C.save(OUT/'status.json',summary);print(json.dumps(summary),flush=True)
    assert physics<=plan['max_physics'] and commands<=plan['max_commands'] and used(OUT)<plan['final_bytes'] and used(ROOT/'stage/runs')<plan['global_bytes']
if __name__=='__main__':main()
