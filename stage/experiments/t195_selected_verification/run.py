"""Fixed native-throttle network-bias grid and all-task acceptance check."""
import os
CPU_POOL=[int(v) for v in os.environ.get('STAGE_CPU_POOL',','.join(map(str,sorted(os.sched_getaffinity(0))))).split(',')]
os.environ['STAGE_CPU_POOL']=','.join(map(str,CPU_POOL));os.sched_setaffinity(0,{int(os.environ.get('STAGE_PROCESS_CPU',str(CPU_POOL[0])))})
import sys,time,fcntl,shutil,traceback,multiprocessing as mp
from multiprocessing.connection import wait
from common import *

def worker(pipe,index,deadline):
    folder=OUT/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True);log=(folder/'worker.log').open('a',buffering=1);sys.stdout=sys.stderr=log;env=None
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);plan=json.loads((HERE/'protocol.json').read_text())
        parent=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['policy'];policy=Policy().eval().requires_grad_(False);native,create=native_setup(folder)
        while True:
            job=pipe.recv()
            if job is None:break
            if env is not None:env.close();env=None
            policy.load_state_dict(parent)
            expected={k:v.clone() for k,v in parent.items()}
            assert all(torch.equal(expected[k],v) for k,v in policy.state_dict().items())
            case=job['case'];env,obs,actual=create(case);env.max_steps=case['step_cap']
            if any(job['perturbation']):
                import mujoco,math
                dx,dy,dyaw=job['perturbation'];yaw=env._pose()[2]+math.radians(dyaw);env.data.qpos[:2]+=[dx,dy];env.data.qpos[3:7]=[math.cos(yaw/2),0.,0.,math.sin(yaw/2)]
                mujoco.mj_forward(env.model,env.data)
                # Preserve factory history/previous action; update current sensors only.
                env.prev_goal_dist=float(env._goal_state()[1]);obs=np.asarray(env._observation(update_history=False),dtype=np.float32)
            assert not env._collision()
            goal=env.goal_xy.copy();model=model_hash(env);steps=0;term=trunc=False;nearest=float(env._goal_state()[1]);trace=[]
            while not(term or trunc):
                assert time.monotonic()<deadline
                state=observation(obs,1-steps/case['step_cap'])
                with torch.no_grad():action=action_clip(policy(torch.tensor(state.reshape(1,-1))).numpy()[0],'full_range')
                trace.append([steps,*env._pose(),*action])
                for _ in range(min(100,case['step_cap']-steps)):
                    obs,_,term,trunc,info=native.step_checked(env,action);obs=np.asarray(obs,dtype=np.float32);steps+=1;nearest=min(nearest,float(env._goal_state()[1]))
                    if term or trunc:break
            assert model_hash(env)==model and np.array_equal(goal,env.goal_xy)
            assert all(torch.equal(expected[k],v) for k,v in policy.state_dict().items())
            path=OUT/(job['id']+'.npz');np.savez_compressed(path,trace=np.asarray(trace))
            row={'id':job['id'],'perturbation':job['perturbation'],'group':job['group'],'case':case['id'],'distance_m':case['distance_m'],'physics_steps':steps,'success':bool(info['reach_goal']),'collision':bool(info['collision']),'timeout':bool(trunc and not term),'nearest_m':nearest,'remaining_m':float(env._goal_state()[1]),'trace_sha256':sha(path),'actual_seed':int(actual),'selected_actor_unchanged':True}
            pipe.send(row)
    except(EOFError,SystemExit):pass
    except BaseException:
        try:pipe.send({'error':traceback.format_exc()})
        except OSError:pass
        raise
    finally:
        if env is not None:env.close()
        pipe.close();log.close()

def main():
    lock=(OUT/'process.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'status.json').exists()
    plan=json.loads((HERE/'protocol.json').read_text())
    for p,h in plan['input_hashes'].items():assert sha(p)==h,p
    assert shutil.disk_usage(OUT).free>40*1024**3 and sum(p.stat().st_size for p in (REPO/'stage/runs').rglob('*') if p.is_file())+plan['max_bytes']<1024**3
    start=time.monotonic();deadline=start+plan['wall_seconds'];ctx=mp.get_context('spawn');pipes=[];workers=[];jobs=plan['jobs'].copy();rows=[]
    try:
        for i in range(8):
            a,b=ctx.Pipe();p=ctx.Process(target=worker,args=(b,i,deadline));os.environ['STAGE_PROCESS_CPU']=str(CPU_POOL[(i+1)%len(CPU_POOL)]);p.start();os.environ['STAGE_PROCESS_CPU']=str(CPU_POOL[0]);b.close();pipes.append(a);workers.append(p)
        save(OUT/'workers.json',{'parent':os.getpid(),'pids':[p.pid for p in workers]});active=set(pipes)
        for pipe in pipes:pipe.send(jobs.pop(0))
        while active:
            assert time.monotonic()<deadline
            for pipe in wait(list(active),timeout=1):
                row=pipe.recv();assert 'error' not in row,row;rows.append(row)
                with (OUT/'evaluation.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                assert sum(r['physics_steps'] for r in rows)<=plan['max_physics']
                save(OUT/'status.json',{'state':'EVALUATING','cases':len(rows),'physics_steps':sum(r['physics_steps'] for r in rows),'optimizer_updates':0})
                if jobs:pipe.send(jobs.pop(0))
                else:pipe.send(None);active.remove(pipe)
        primary=[r for r in rows if r['group']=='primary']
        right=next(r for r in primary if r['case']=='5m_2029654802_RIGHT')
        source=json.loads(Path(plan['selection_summary']).read_text())['selected']
        original_rows=[json.loads(s) for s in Path(plan['selection_cases']).read_text().splitlines()]
        for r in primary:
            old=next(v for v in original_rows if v['delta']==source['delta'] and v['case']==r['case'])
            for key in ['physics_steps','success','collision','timeout','nearest_m','remaining_m','trace_sha256']:assert r[key]==old[key],key
        grid=[right]+[r for r in rows if r['group']=='right_perturbation']
        scores={'old6':{'success':sum(r['success'] and not r['collision'] for r in primary if r['distance_m']<5),'n':6},'original5m':{'success':sum(r['success'] and not r['collision'] for r in primary if r['distance_m']==5),'n':2},'right_reset_grid':{'success':sum(r['success'] and not r['collision'] for r in grid),'n':len(grid),'collision':sum(r['collision'] for r in grid),'timeout':sum(r['timeout'] for r in grid)}}
        result={'state':'COMPLETE','scores':scores,'primary_reproduction_exact':True,'checkpoint_sha256':sha(plan['checkpoint']),'physics_steps':sum(r['physics_steps'] for r in rows),'optimizer_updates':0,'automatic_promotion':False,'incumbent':'T159','scope':plan['scope']}
        save(OUT/'summary.json',result);save(OUT/'status.json',result);print(json.dumps(result),flush=True)
    except BaseException:
        save(OUT/'failure.json',{'state':'FAILED','traceback':traceback.format_exc(),'recorded_physics_steps':sum(r['physics_steps'] for r in rows),'unrecorded_inflight_upper_bound':8*23334});raise
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except(OSError,EOFError):pass
        for p in workers:
            p.join(5)
            if p.is_alive():p.terminate();p.join(5)
        for pipe in pipes:pipe.close()

if __name__=='__main__':main()
