"""Eight serialized-policy original-scene closed-loop checks, no learning."""
from support import *
import fcntl,multiprocessing as mp,shutil
def main():
    assert not OUT.exists(),'No duplicate T215'
    source=json.loads((ROOT/'stage/runs/t214_preconditioned_curvature/summary.json').read_text())
    assert source['state']=='COMPLETE' and source['contracts']['all_checks_pass'] and source['export']
    assert source['export']['selection']['id'].startswith(('CG_','PCG_'))
    old=json.loads((ROOT/'stage/experiments/t211_guarded_progress/protocol.json').read_text());weights=source['export']['path']
    plan={'experiment':'T215','wall_seconds':180,'workers':4,'max_episodes':8,'max_physics':150000,
          'final_bytes':3*1024**2,'peak_bytes':4*1024**2,'weights':weights,'weights_sha256':C.sha(weights),
          'cases':old['primary_cases'],'snapshots':old['task_snapshots'],
          'scope':'Exactly8fixedoriginalold8deterministicrollouts, savedT214offlinecandidate; zerotraining/optimizer. No scenechange, planner,mapinput,waypoint,teacher,automaticpromotion or follow-ontraining; fixedtrainingtasksnotgeneralization.'}
    files=[Path(__file__),HERE/'support.py',Path(weights),ROOT/'stage/runs/t214_preconditioned_curvature/summary.json']+[Path(x) for x in plan['snapshots']]+[ROOT/'stage/experiments/t203_native_exploration/runtime.py',ROOT/'stage/experiments/t200_residual_placement/common.py']
    plan['input_hashes']={str(p):C.sha(p) for p in files}
    assert used(ROOT/'stage/runs')+plan['peak_bytes']<1024**3 and shutil.disk_usage(ROOT).free>40*1024**3
    OUT.mkdir();C.save(HERE/'protocol.json',plan);C.save(OUT/'protocol.json',plan)
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();deadline=time.time()+plan['wall_seconds'];pipes=[];workers=[];rows=[];physics=0;reason='COMPLETE'
    C.save(OUT/'status.json',{'state':'RUNNING','pid':os.getpid(),'optimizer_updates':0})
    try:
        ctx=mp.get_context('spawn')
        for i in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(i+1);w=ctx.Process(target=worker,args=(b,i,deadline));w.start();b.close();pipes.append(a);workers.append(w)
        for off in (0,4):
            assert physics+sum(c['step_cap'] for c in plan['cases'][off:off+4])<=plan['max_physics']
            for i,pipe in enumerate(pipes,off):
                pipe.send(('rollout',{'id':f'original_old{i}','arm':'offline_candidate','round':0,'role':'evaluation','mode':'deterministic','seed':0,
                    'weights':weights,'weights_sha256':plan['weights_sha256'],'case':plan['cases'][i],'exact_snapshot':plan['snapshots'][i]}))
            for pipe in pipes:
                assert pipe.poll(max(.1,deadline-time.time()+1));r=pipe.recv();assert 'error' not in r,r;rows.append(r);physics+=r['physics_steps']
                with (OUT/'rollouts.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
            assert all(r['complete'] for r in rows)
            C.save(OUT/'status.json',{'state':'RUNNING','episodes':len(rows),'physics_steps':physics,'optimizer_updates':0})
    except BaseException:reason='FAILED';C.save(OUT/'failure.json',{'traceback':traceback.format_exc(),'physics_steps':physics})
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except OSError:pass
        for w in workers:
            w.join(3)
            if w.is_alive():w.terminate();w.join(3)
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    safe=lambda r:r['complete'] and r['success'] and not r['collision']
    summary={'state':reason,'episodes':len(rows),'physics_steps':physics,'training_physics_steps':0,'optimizer_updates':0,
             'old7_safe_GOAL':sum(safe(r) for i,r in enumerate(rows) if i!=6),'LEFT_safe_GOAL':safe(rows[6]) if len(rows)>6 else None,
             'old8_safe_GOAL':sum(safe(r) for r in rows),'results':rows,'all_workers_stopped':True,'automatic_promotion':False,
             'stable_5m_gate_passed':False,'elapsed_s':time.monotonic()-start,'own_bytes':used(OUT)}
    assert used(OUT)<plan['final_bytes'] and used(ROOT/'stage/runs')<1024**3
    C.save(OUT/'summary.json',summary);C.save(OUT/'status.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='results'}))
if __name__=='__main__':main()
