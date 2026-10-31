from acceptance_support import *
import multiprocessing as mp,fcntl,shutil
class Stop(Exception):pass
def main():
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'status.json').exists(),'No duplicate T222'
    plan=json.loads((HERE/'protocol.json').read_text());start=time.monotonic();deadline=start+plan['wall_seconds'];wall=time.time()+plan['wall_seconds'];physics=commands=0;rows=[];workers=[];pipes=[];reason=None;rebuilt=False
    def status(state):C.save(OUT/'status.json',dict(state=state,commands=commands,physics_steps=physics,cases=len(rows),navigation_training_updates=0,recipe_reconstruction_optimizer_calls=3000 if rebuilt else None,elapsed_s=time.monotonic()-start))
    def guard(n=0,steps=0,pending=0):
        if time.monotonic()+5>=deadline:raise Stop('WALL_RESERVE')
        if commands+n>plan['max_commands'] or physics+steps>plan['max_physics']:raise Stop('COUNT_RESERVE')
        if used(OUT)+pending>=plan['final_bytes'] or used(ROOT/'stage/runs')+pending>=plan['global_bytes'] or shutil.disk_usage(ROOT).free<40*1024**3:raise Stop('STORAGE_RESERVE')
    def receive(pipe):
        while not pipe.poll(.5):
            if time.monotonic()>deadline:raise Stop('RESPONSE_DEADLINE')
        r=pipe.recv();assert 'error' not in r,r;return r
    try:
        for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
        status('RECONSTRUCTING_FROM_DISK');torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        p=L.reconstruct();ph=digest(p.state_dict());assert ph==plan['final_policy_sha256'];rebuilt=True;ps=S.state_np(p)
        C.save(OUT/'reconstruction_gate.json',dict(state='PASS',policy_sha256=ph,recipe_optimizer_calls=3000,navigation_training_updates=0,source='T221 hashedparent+persistedverifiedsuccessdata+fixedrecipe; freshprocess'))
        ctx=mp.get_context('spawn')
        for i in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(i+1);w=ctx.Process(target=worker,args=(b,i,wall));w.start();b.close();pipes.append(a);workers.append(w)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(pipe) for pipe in pipes];assert all(r.get('ready') for r in ready);C.save(OUT/'workers.json',{'parent':os.getpid(),'workers':[w.pid for w in workers]})
        for off in range(0,len(plan['jobs']),4):
            batch=plan['jobs'][off:off+4];guard(len(batch),sum(j['case']['step_cap'] for j in batch),16384)
            for pipe,j in zip(pipes,batch):pipe.send(dict(j,policy_np=ps,policy_hash=ph))
            replies=[receive(pipe) for pipe in pipes[:len(batch)]];commands+=len(batch);physics+=sum(r['physics_steps'] for r in replies)
            for row in replies:
                rows.append(row)
                with (OUT/'cases.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            status('FROZEN_ACCEPTANCE')
            if not all(r['valid_start'] and r['complete'] for r in replies):raise Stop('INVALID_OR_CENSORED_CASE')
        reason='CASE_BUDGET_COMPLETE'
    except Stop as e:reason=str(e)
    except BaseException:reason='FAILED';C.save(OUT/'failure.json',{'traceback':traceback.format_exc(),'physics_steps':physics,'commands':commands})
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except OSError:pass
        for w in workers:
            w.join(3)
            if w.is_alive():w.kill();w.join(3)
        assert all(not w.is_alive() for w in workers)
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    source=json.loads((ROOT/'stage/runs/t221_left_success/summary.json').read_text())['final_reconstructed_old8'];centers={}
    for side,i in [('LEFT',6),('RIGHT',7)]:
        r=next((r for r in rows if r['group']==side and r['delta_x_m']==r['delta_y_m']==r['delta_yaw_deg']==0.),None)
        centers[side]=bool(r and r['complete'] and r['physics_steps']==source[i]['physics_steps'] and r['integration_sha256']==source[i]['integration_sha256'])
    scores={}
    for group in ('old6','LEFT','RIGHT'):
        r=[r for r in rows if r['group']==group];scores[group]=dict(n=len(r),safeGOAL=sum(bool(t['valid_start'] and t['complete'] and t['success'] and not t['collision']) for t in r),CONTACT=sum(bool(t['collision']) for t in r),timeout=sum(bool(t['timeout']) for t in r))
    gate=(reason=='CASE_BUDGET_COMPLETE' and all(scores[g]['n']==scores[g]['safeGOAL']==n for g,n in [('old6',6),('LEFT',27),('RIGHT',27)]) and all(centers.values()) and rebuilt)
    summary=dict(state='FAILED' if reason=='FAILED' else 'COMPLETE',stop_reason=reason,scores=scores,centers_exact_to_T221=centers,reconstructed_policy_sha256=plan['final_policy_sha256'],reconstruction_verified=rebuilt,
        navigation_training_updates=0,recipe_reconstruction_optimizer_calls=3000 if rebuilt else None,physics_steps=physics,commands=commands,stable_5m_local_grid_gate_passed=gate,independent_generalization_claim=False,automatic_promotion=False,all_workers_stopped=True,elapsed_s=time.monotonic()-start,own_bytes=used(OUT),all_runs_MiB=used(ROOT/'stage/runs')/1024**2)
    C.save(OUT/'summary.json',summary);C.save(OUT/'status.json',summary);print(json.dumps(summary),flush=True)
    assert physics<=plan['max_physics'] and commands<=plan['max_commands'] and used(OUT)<plan['final_bytes'] and used(ROOT/'stage/runs')<plan['global_bytes']
if __name__=='__main__':main()
