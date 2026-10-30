"""Frozen Stage scene-component controls and native actuator probes."""
from pathlib import Path
import hashlib
import importlib.util
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
import sys
import time
import traceback
import fcntl
import math
import numpy as np

CPU_POOL=[int(x) for x in os.environ.get('STAGE_CPU_POOL',','.join(map(str,sorted(os.sched_getaffinity(0))))).split(',')]
os.environ['STAGE_CPU_POOL']=','.join(map(str,CPU_POOL))
os.sched_setaffinity(0,{int(os.environ.get('STAGE_PROCESS_CPU',str(CPU_POOL[0])))})
import torch

REPO=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=REPO/'stage/runs/t179_mechanism_audit/contact_recheck'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    q=p.with_suffix('.partial');q.write_text(json.dumps(x,indent=2)+'\n');os.replace(q,p)
def load(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def worker(pipe,index,deadline):
    folder=OUT/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True)
    log=(folder/'worker.log').open('a',buffering=1);sys.stdout=sys.stderr=log
    os.environ['C98_JOB']=os.environ['TD3_JOB']=str(folder);os.environ['C22_SCREEN_ONLY']='0'
    env=renderer=None
    try:
        import mujoco
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        plan=json.loads((HERE/'contact_recheck_protocol.json').read_text())
        core=load(REPO/'stage/current/outputs/stage_t148/td3_core_goal_contrast.py','scene_control_core')
        actor=core.Actor().eval().requires_grad_(False)
        ck=torch.load(plan['actor_checkpoint'],map_location='cpu',weights_only=False);actor.load_state_dict(ck['actor']);del ck
        before={k:v.clone() for k,v in actor.state_dict().items()}
        t=load(REPO/'stage/current/outputs/stage_t148/train.py','scene_control_source')
        helper=t.load_module(t.HELPER,'scene_control_helper');legacy=t.load_module(t.STAGE,'scene_control_legacy')
        for p,h in legacy.EXPECTED.items():assert sha(p)==h,p
        runner=legacy.module(legacy.SRC,'scene_control_runner');_,v,mod,prep,_=runner.load_native();native=prep[3]
        make=legacy.make_short_task(runner.make_task)
        while True:
            job=pipe.recv()
            if job is None:break
            if env is not None:env.close()
            case=job['case'];env,renderer,obs,actual=helper.make_goal_task(make,v,mod,native,case['seed'],case['angle_deg'],case['distance_m'])
            if renderer is not None:renderer.close();renderer=None
            env.max_steps=job['step_cap'];goal=env.goal_xy.copy()
            reset_hash=hashlib.sha256(env.data.qpos.tobytes()+env.data.qvel.tobytes()+goal.tobytes()).hexdigest()
            gid=mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_GEOM,'real_circle_obs_9')
            assert gid>=0 and int(env.model.geom_type[gid])==int(mujoco.mjtGeom.mjGEOM_CYLINDER)
            assert int(env.model.geom_bodyid[gid])==0 and np.allclose(env.data.geom_xpos[gid],env.model.geom_pos[gid])
            scale=job['scale'];assert 0<=scale<=1
            original_size=env.model.geom_size.copy();original_pos=env.model.geom_pos.copy();original_masks=env.model.geom_contype.copy(),env.model.geom_conaffinity.copy()
            original_aabb=env.model.geom_aabb.copy();original_rbound=env.model.geom_rbound.copy()
            # Only shrink. Existing compiled broad-phase bounds remain conservative supersets.
            # Keep centers fixed at positive scales; zero uses the already tested removal convention.
            if scale==0:
                env.model.geom_pos[gid,:2]+=100
                env.model.geom_contype[gid]=0;env.model.geom_conaffinity[gid]=0
            else:
                env.model.geom_size[gid,0]*=scale
                assert env.model.geom_size[gid,0]<=original_size[gid,0]
            other=np.arange(env.model.ngeom)!=gid
            assert np.array_equal(env.model.geom_size[other],original_size[other])
            assert np.array_equal(env.model.geom_pos[other],original_pos[other])
            assert np.array_equal(env.model.geom_contype[other],original_masks[0][other])
            assert np.array_equal(env.model.geom_conaffinity[other],original_masks[1][other])
            assert np.array_equal(env.model.geom_aabb,original_aabb) and np.array_equal(env.model.geom_rbound,original_rbound)
            mujoco.mj_forward(env.model,env.data)
            # Probe actual mj_ray narrow-phase radius using only the target geom's group.
            groups=env.model.geom_group.copy();env.model.geom_group[:]=0;env.model.geom_group[gid]=5
            raymask=np.zeros(6,dtype=np.uint8);raymask[5]=1;hit=np.array([-1],dtype=np.int32)
            origin=original_pos[gid].copy();origin[0]+=1.;origin[2]=.36
            ray=float(mujoco.mj_ray(env.model,env.data,origin,np.array([-1.,0.,0.]),raymask,1,-1,hit))
            expected=1.-original_size[gid,0]*scale if scale>0 else -1.
            assert abs(ray-expected)<1e-6,(ray,expected)
            env.model.geom_group[:]=groups
            if job['history']=='refill':
                env.history.clear();env.prev_action=np.zeros(2,dtype=np.float32)
            obs=env._observation(update_history=False)
            assert not env._collision() and abs(float(env.model.opt.timestep)-.002)<1e-10
            initial_obs_sha=hashlib.sha256(np.asarray(obs).tobytes()).hexdigest()
            empty_min=float(obs[:48].min()*6)
            geom_hash=hashlib.sha256(env.model.geom_pos.tobytes()+env.model.geom_size.tobytes()+env.model.geom_contype.tobytes()+env.model.geom_conaffinity.tobytes()).hexdigest()
            removed=['real_circle_obs_9'] if scale==0 else []
            samples=[];travel=0.;previous=np.asarray(env._pose()[:2]);terminal=trunc=False
            speeds=[];yaw_rates=[];steps=0;start_time=float(env.data.time)
            while not (terminal or trunc):
                assert time.monotonic()<deadline
                if steps%job['hold']==0:
                    if job['kind']=='actuator_probe':raw=np.asarray(job['raw_action'],dtype=np.float32)
                    else:
                        with torch.no_grad():raw=actor(torch.tensor(np.asarray(obs,dtype=np.float32).reshape(1,-1))).numpy()[0]
                    samples.append((steps,obs.copy(),raw.copy(),np.asarray(env._pose())))
                mapped=np.array([(raw[0]+1)/2,raw[1]],dtype=np.float32);time0=float(env.data.time)
                obs,_,terminal,trunc,info=native.step_checked(env,mapped);obs=np.asarray(obs,dtype=np.float32)
                assert abs(float(env.data.time)-time0-.002)<1e-7
                pose=np.asarray(env._pose());travel+=float(np.linalg.norm(pose[:2]-previous));previous=pose[:2].copy()
                speeds.append(float(np.hypot(pose[3],pose[4])));yaw_rates.append(float(pose[5]));steps+=1
            assert np.array_equal(goal,env.goal_xy)
            assert all(torch.equal(before[k],v) for k,v in actor.state_dict().items())
            np.savez_compressed(OUT/(job['id']+'.npz'),steps=np.asarray([s[0] for s in samples]),states=np.asarray([s[1] for s in samples]),actions=np.asarray([s[2] for s in samples]),poses=np.asarray([s[3] for s in samples]))
            last=max(1,min(len(speeds),1000));speed=float(np.mean(speeds[-last:]));yaw_rate=float(np.mean(yaw_rates[-last:]))
            row={'id':job['id'],'kind':job['kind'],'case':case,'scene':job['scene'],'control_hold_steps':job['hold'],
                 'scale':scale,'history':job['history'],'radius_m':float(original_size[gid,0]*scale),'ray_probe_m':ray,'expected_ray_probe_m':expected,'initial_obs_sha256':initial_obs_sha,'reset_hash':reset_hash,'scene_hash':geom_hash,'actual_seed':int(actual),'removed_obstacles':removed,
                 'initial_lidar_min_m':empty_min,'steps':steps,'simulation_s':float(env.data.time)-start_time,
                 'success':bool(info['reach_goal']),'collision':bool(info['collision']),
                 'safe_success':bool(info['reach_goal'] and not info['collision']),'remaining_m':float(env._goal_state()[1]),
                 'goal_xy':goal.tolist(),'travel_m':travel,'steady_speed_m_s':speed,'steady_yaw_rate_rad_s':yaw_rate,
                 'steady_turn_radius_m':speed/abs(yaw_rate) if abs(yaw_rate)>1e-5 else None,
                 'final_pose':np.asarray(env._pose()).tolist(),'actor_unchanged':True}
            if job['kind']=='actuator_probe':row['raw_action']=job['raw_action']
            contacts=[]
            for contact in env.data.contact[:env.data.ncon]:
                g1,g2=int(contact.geom1),int(contact.geom2)
                robot1=int(env.model.geom_bodyid[g1]) in env.robot_body_ids;robot2=int(env.model.geom_bodyid[g2]) in env.robot_body_ids
                if robot1!=robot2 and float(contact.dist)<=.002:
                    contacts.append({'geom1':mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,g1),'geom2':mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,g2),'dist_m':float(contact.dist)})
            row['terminal_contacts']=contacts
            pipe.send(row)
    except (EOFError,SystemExit):pass
    except BaseException:
        try:pipe.send({'error':traceback.format_exc()})
        except OSError:pass
        raise
    finally:
        if renderer is not None:renderer.close()
        if env is not None:env.close()
        pipe.close();log.close()

def main():
    lock=(OUT/'process.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'status.json').exists(),'No duplicate/restart'
    plan=json.loads((HERE/'contact_recheck_protocol.json').read_text())
    for p,h in plan['input_hashes'].items():assert sha(p)==h,p
    assert sum(p.stat().st_size for p in (REPO/'stage/runs').rglob('*') if p.is_file())+64*1024**2<1024**3
    ctx=mp.get_context('spawn');pipes=[];processes=[];jobs=list(plan['jobs']);rows=[];start=time.monotonic();deadline=start+plan['wall_seconds']
    try:
        for i in range(min(8,len(jobs))):
            a,b=ctx.Pipe();p=ctx.Process(target=worker,args=(b,i,deadline));os.environ['STAGE_PROCESS_CPU']=str(CPU_POOL[(i+1)%len(CPU_POOL)]);p.start();os.environ['STAGE_PROCESS_CPU']=str(CPU_POOL[0]);b.close();pipes.append(a);processes.append(p)
        save(OUT/'workers.json',{'pids':[p.pid for p in processes],'parent':os.getpid()})
        active=set(pipes)
        for pipe in pipes:pipe.send(jobs.pop(0))
        while active:
            assert time.monotonic()<deadline
            for pipe in wait(list(active),timeout=1):
                row=pipe.recv();assert 'error' not in row,row;rows.append(row)
                with (OUT/'cases.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                frames=sum(r['steps'] for r in rows);assert frames<=plan['physics_budget']
                save(OUT/'status.json',{'state':'RUNNING','completed_cases':len(rows),'physics_steps':frames,'actor_updates':0})
                if jobs:pipe.send(jobs.pop(0))
                else:pipe.send(None);active.remove(pipe)
        pairs={}
        for row in rows:pairs.setdefault(row['case']['id'],set()).add(row['reset_hash'])
        assert all(len(s)==1 for s in pairs.values()),'Matched starts/goals changed'
        groups={}
        for row in rows:
            if row['kind']!='actuator_probe':
                key=str(row['scale'])+'/'+row['history']+'/'+str(row['control_hold_steps'])
                groups.setdefault(key,[]).append(row)
        scores={k:{'safe_success':sum(r['safe_success'] for r in v),'n':len(v),'collisions':sum(r['collision'] for r in v)} for k,v in groups.items()}
        save(OUT/'summary.json',{'state':'COMPLETE','scores':scores,'cases':rows,'paired_reset_hashes':'PASS','actor_unchanged':True,
                               'physics_steps':sum(r['steps'] for r in rows),'automatic_promotion':False,
                               'limits':'Post-hoc3case contact provenance recheck selected after radius diagnostic. Exact reruns are not additional independent evidence or training.'})
        save(OUT/'status.json',{'state':'COMPLETE','completed_cases':len(rows),'physics_steps':sum(r['steps'] for r in rows),'actor_updates':0,'elapsed_s':time.monotonic()-start})
        print(json.dumps({'state':'COMPLETE','scores':scores}),flush=True)
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except (OSError,EOFError):pass
        for p in processes:
            p.join(5)
            if p.is_alive():p.terminate();p.join(5)
        for pipe in pipes:pipe.close()

if __name__=='__main__':main()
