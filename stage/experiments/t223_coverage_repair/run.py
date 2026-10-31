from repair_support import *
import multiprocessing as mp,fcntl,shutil
sys.path.insert(0,str(ROOT/'stage/experiments/t222_stability'))
import acceptance_support as A
class Stop(Exception):pass
def eval_worker(pipe,index,deadline):
    A.OUT=OUT;A.worker(pipe,index,deadline)
def main():
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'status.json').exists(),'No duplicate T223'
    plan=json.loads((HERE/'protocol.json').read_text());start=time.monotonic();deadline=start+plan['wall_seconds'];wall=time.time()+plan['wall_seconds'];physics=commands=steps=shadow_steps=0;rows=[];sources=[];workers=[];pipes=[];reason=None;reconstruction=False;checks=[];sc=[]
    def status(state):C.save(OUT/'status.json',dict(state=state,physics_steps=physics,commands=commands,supervised_updates=steps,shadow_updates=shadow_steps,RL_updates=0,eval_cases=len(rows),elapsed_s=time.monotonic()-start))
    def guard(n=0,count=0,pending=0,reserve=0):
        if time.monotonic()+reserve>=deadline:raise Stop('WALL_RESERVE')
        if commands+n>plan['max_commands'] or physics+count>plan['max_physics']:raise Stop('COUNT_RESERVE')
        if used(OUT)+pending>=plan['final_bytes'] or used(ROOT/'stage/runs')+pending>=plan['global_bytes'] or shutil.disk_usage(ROOT).free<40*1024**3:raise Stop('STORAGE_RESERVE')
    def receive(pipe):
        while not pipe.poll(.5):
            if time.monotonic()>deadline:raise Stop('RESPONSE_DEADLINE')
        r=pipe.recv();assert 'error' not in r,r;return r
    def launch(target):
        ctx=mp.get_context('spawn')
        for i in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(i+1);w=ctx.Process(target=target,args=(b,i,wall));w.start();b.close();pipes.append(a);workers.append(w)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(p) for p in pipes];assert all(r.get('ready') for r in ready)
        C.save(OUT/('eval_workers.json' if target is eval_worker else 'source_workers.json'),dict(parent=os.getpid(),workers=[w.pid for w in workers]))
    def stop():
        for p in pipes:
            try:p.send(None)
            except OSError:pass
        for w in workers:
            w.join(3)
            if w.is_alive():w.kill();w.join(3)
        assert all(not w.is_alive() for w in workers);workers.clear();pipes.clear()
    try:
        for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        parent=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['policy'];p=S.policy(parent);ps=state_np(p);ph=digest(ps)
        with np.load(plan['retention_targets']) as z:
            with torch.no_grad():assert float((p(torch.tensor(z['states']))-torch.tensor(z['actions'])).abs().max())==0
        C.save(OUT/'contracts.json',dict(state='PASS',parent_function_error=0,physics_steps=0,optimizer_calls=0))
        launch(worker);sets=[]
        for pass_index in range(2):
            items=[]
            for off in range(0,6,4):
                jobs=plan['jobs'][off:off+4];guard(len(jobs),sum(j['case']['step_cap'] for j in jobs),8192,reserve=240)
                for pipe,j in zip(pipes,jobs):pipe.send(dict(j,policy_np=ps,policy_hash=ph))
                batch=[receive(pipe) for pipe in pipes[:len(jobs)]];commands+=len(batch);physics+=sum(r['row']['physics_steps'] for r in batch);items.extend(batch)
            sets.append(items);C.save(OUT/f'source_pass_{pass_index}.json',[r['row'] for r in items]);status('SOURCE_VERIFICATION')
        assert all(i['row']==j['row'] and all(np.array_equal(v,j['arrays'][k]) for k,v in i['arrays'].items()) for i,j in zip(*sets))
        sources=[i['row'] for i in sets[0]]
        safe=lambda r:r['complete'] and r['success'] and not r['collision']
        if sum(safe(r) for r in sources if r['arm']=='RIGHT')!=4 or sum(safe(r) for r in sources if r['arm']=='LEFT')<1:raise Stop('SOURCE_SUCCESS_GATE_FAILED')
        stop();g=training_groups(plan,sets[0]);sg=training_groups(plan,sets[1]);assert all(torch.equal(x,xx) and torch.equal(y,yy) for (x,y),(xx,yy) in zip(g,sg));del sets
        def report(n,ph,oh,loss):
            nonlocal steps
            steps=n;checks.append(dict(steps=n,policy_sha256=ph,optimizer_sha256=oh,loss=loss));status('SUPERVISED_COVERAGE_REPAIR')
        live,opt,count=L.train_recipe(plan,g,plan['updates'],lambda:guard(reserve=200),report)
        def sr(n,ph,oh,loss):
            nonlocal shadow_steps
            shadow_steps=n;r=dict(steps=n,policy_sha256=ph,optimizer_sha256=oh,loss=loss);assert r==checks[n//500-1];sc.append(r);status('INDEPENDENT_RECONSTRUCTION')
        shadow,sopt,count=L.train_recipe(plan,sg,plan['updates'],lambda:guard(reserve=160),sr)
        assert digest(live.state_dict())==digest(shadow.state_dict()) and digest(opt.state_dict())==digest(sopt.state_dict());reconstruction=True
        recipe=dict(independent_reconstruction_exact=True,source_rows=sources,final_policy_sha256=digest(shadow.state_dict()),final_optimizer_sha256=digest(sopt.state_dict()),checkpoints=checks,shadow_checkpoints=sc,completed_supervised_steps=steps,model_storage='Recreate6sourceepisodes with frozenT194/reset/seeds andverifyhashes, then3000fixedAdamsteps; no binary checkpoint')
        C.save(OUT/'recipe.json',recipe);launch(eval_worker);ps=state_np(shadow);ph=digest(ps);jobs=json.loads(Path(plan['evaluation_protocol']).read_text())['jobs']
        for off in range(0,len(jobs),4):
            batch=jobs[off:off+4];guard(len(batch),sum(j['case']['step_cap'] for j in batch),8192)
            for pipe,j in zip(pipes,batch):pipe.send(dict(j,policy_np=ps,policy_hash=ph))
            reply=[receive(pipe) for pipe in pipes[:len(batch)]];commands+=len(reply);physics+=sum(r['physics_steps'] for r in reply)
            for full in reply:
                row={k:full[k] for k in ('id','group','valid_start','complete','physics_steps','success','collision','timeout','remaining_m','nearest_m','states_sha256','actions_sha256','integration_sha256')};rows.append(row)
                with (OUT/'cases.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            status('FROZEN_DEVELOPMENT_GRID')
            if not all(r['valid_start'] and r['complete'] for r in reply):raise Stop('INVALID_OR_CENSORED_CASE')
        reason='BUDGET_COMPLETE'
    except Stop as e:reason=str(e)
    except BaseException:reason='FAILED';C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physics_steps=physics,commands=commands))
    finally:stop()
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    scores={}
    for side in ('old6','LEFT','RIGHT'):
        r=[r for r in rows if r['group']==side];scores[side]=dict(n=len(r),safeGOAL=sum(bool(t['complete'] and t['success'] and not t['collision']) for t in r),CONTACT=sum(bool(t['collision']) for t in r),timeout=sum(bool(t['timeout']) for t in r))
    gate=reason=='BUDGET_COMPLETE' and reconstruction and all(scores[g]['n']==scores[g]['safeGOAL']==n for g,n in [('old6',6),('LEFT',27),('RIGHT',27)])
    summary=dict(state='FAILED' if reason=='FAILED' else 'COMPLETE',stop_reason=reason,scores=scores,source_scores={side:dict(n=sum(r['arm']==side for r in sources),safeGOAL=sum(r['arm']==side and r['complete'] and r['success'] and not r['collision'] for r in sources)) for side in ('LEFT','RIGHT')},supervised_updates=steps,shadow_updates=shadow_steps,RL_updates=0,reconstruction_verified=reconstruction,physics_steps=physics,commands=commands,development_grid_gate_passed=gate,stable_independent_heldout_claim=False,automatic_promotion=False,all_workers_stopped=True,elapsed_s=time.monotonic()-start,own_bytes=used(OUT),all_runs_MiB=used(ROOT/'stage/runs')/1024**2)
    C.save(OUT/'summary.json',summary);C.save(OUT/'status.json',summary);print(json.dumps(summary),flush=True)
    assert physics<=plan['max_physics'] and commands<=plan['max_commands'] and used(OUT)<plan['final_bytes'] and used(ROOT/'stage/runs')<plan['global_bytes']
if __name__=='__main__':main()
