"""Frozen native collection and exact goal/history relabel contract audit.

Only instrumentation is attached in memory; original source files are read-only.
No original trajectory is reclassified as success at the original final goal.
"""
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import math,time,traceback,fcntl
from common import *

def goal_fields(pose,goal):
    x,y,yaw=map(float,pose[:3]);dx,dy=map(float,np.asarray(goal)-[x,y])
    bx=math.cos(yaw)*dx+math.sin(yaw)*dy;by=-math.sin(yaw)*dx+math.cos(yaw)*dy
    d=math.hypot(dx,dy);a=math.atan2(by,bx)
    return np.array([np.clip(d/10,0,2),math.sin(a),math.cos(a),np.clip(bx/10,-2,2),np.clip(by/10,-2,2)],dtype=np.float32),d,a

def rebuild(obs,pose,history_poses,goal,history_goals=None):
    result=obs.copy();result[48:53]=goal_fields(pose,goal)[0]
    for i,hp in enumerate(history_poses):
        hg=goal if history_goals is None else history_goals[i]
        _,d,a=goal_fields(hp,hg);result[64+12*i+8]=np.clip(d/10,0,2);result[64+12*i+9]=np.clip(a/math.pi,-1,1)
    return result

def main():
    lock=(OUT/'process.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'status.json').exists(),'No duplicate or implicit resume'
    plan=json.loads((HERE/'protocol.json').read_text())
    for p,h in plan['input_hashes'].items():assert sha(p)==h,p
    start=time.monotonic();deadline=start+plan['wall_seconds'];spent=0
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    policy=Policy().eval().requires_grad_(False);state=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['policy'];policy.load_state_dict(state)
    native,create=native_setup(OUT/'native')
    from training.e2e_phase10_lidar_v1.phase10_scene_e2e_env import Phase10SceneE2ELidarEnv as Base
    original_frame=Base._history_frame;events={};synthetic_seeds={}
    def frame_hook(env,lidar,distance,angle):
        frame=original_frame(env,lidar,distance,angle)
        entry={'frame':frame.copy(),'pose':np.asarray(env._pose(),dtype=np.float64),'goal':env.goal_xy.copy(),'time':float(env.data.time)}
        events.setdefault(id(env),deque(maxlen=16)).append(entry)
        return frame
    Base._history_frame=frame_hook
    def metadata(env):
        history=list(env.history);available=list(events.get(id(env),[]));records=[]
        for frame in history:
            found=next((r for r in reversed(available) if np.array_equal(r['frame'],frame)),None)
            assert found is not None,'History frame lacks actual generation witness'
            records.append(found)
        assert len(records)==4
        return np.asarray([r['pose'] for r in records]),np.asarray([r['goal'] for r in records]),np.asarray([r['time'] for r in records])
    results=[];max_error=0.;relabel_rows=[];rng=np.random.default_rng(plan['seed']);env=None
    try:
        save(OUT/'status.json',{'state':'COLLECTING','physics_steps':0,'optimizer_updates':0})
        for job in plan['jobs']:
            if env is not None:env.close();env=None
            env,obs,actual=create(job['case']);cap=job['case']['step_cap'];env.max_steps=cap
            if job.get('snapshot_side'):
                side=job['snapshot_side'];source=plan['anchor_sources'][side];obs=restore(env,read_snapshot(source['snapshot_path']));env.max_steps=cap
                # Historical reset frames have not been generated in this process.
                # Seed only after verifying the known T186 reset pose/goal contract.
                events[id(env)]=deque(maxlen=16)
                for f in list(env.history):
                    _,d,a=goal_fields(env._pose(),env.goal_xy)
                    assert abs(float(f[8])-d/10)<1e-6 and abs(float(f[9])-a/math.pi)<1e-6
                    events[id(env)].append({'frame':f.copy(),'pose':np.asarray(env._pose()),'goal':env.goal_xy.copy(),'time':float(env.data.time)})
                synthetic_seeds[job['id']]='T186 reset source pose/goal verified against stored fields; no claim of new generation witness before collection'
            model=model_hash(env);goal=env.goal_xy.copy();term=trunc=False;steps=0;decisions=[];physical=[];initial_history_goals=None
            while not(term or trunc):
                assert time.monotonic()<deadline
                hp,hg,ht=metadata(env);pose=np.asarray(env._pose(),dtype=np.float64)
                # Rebuild using each frame's actually recorded goal, retaining stale factory history.
                rebuilt=rebuild(obs,pose,hp,goal,hg);err=float(np.abs(rebuilt-obs).max());max_error=max(max_error,err);assert err<1e-6,err
                if initial_history_goals is None:initial_history_goals=hg.tolist()
                state113=observation(obs,1-steps/cap)
                with torch.no_grad():action=action_clip(policy(torch.tensor(state113.reshape(1,-1))).numpy()[0],'full_range')
                before=steps
                for _ in range(min(plan['hold_steps'],cap-steps)):
                    t0=float(env.data.time);obs,_,term,trunc,info=native.step_checked(env,action);obs=np.asarray(obs,dtype=np.float32);steps+=1;spent+=1
                    assert abs(float(env.data.time)-t0-.002)<1e-7
                    physical.append([steps,*env._pose(),float(info['collision']),float(info['reach_goal']),float(trunc)])
                    if term or trunc:break
                nhp,nhg,nht=metadata(env)
                next113=observation(obs,1-steps/cap)
                err=float(np.abs(rebuild(next113,env._pose(),nhp,goal,nhg)-next113).max());max_error=max(max_error,err);assert err<1e-6
                decisions.append((before,state113,action.copy(),steps-before,next113,pose,hp,hg,ht,np.asarray(env._pose()),nhp,nhg,nht))
            assert model_hash(env)==model and np.array_equal(goal,env.goal_xy)
            arrays={k:np.asarray([r[i] for r in decisions]) for i,k in enumerate(['steps','states','actions','native_counts','next_states','poses','history_poses','history_goals','history_times','next_poses','next_history_poses','next_history_goals','next_history_times'])}
            arrays['physical']=np.asarray(physical,dtype=np.float64);arrays['goal_xy']=goal
            path=OUT/(job['id']+'.npz');np.savez_compressed(path,**arrays)
            row={'id':job['id'],'actual_seed':int(actual),'physics_steps':steps,'decisions':len(decisions),'success':bool(info['reach_goal']),'collision':bool(info['collision']),'timeout':bool(trunc and not term),'remaining_m':float(env._goal_state()[1]),'goal_xy':goal.tolist(),'initial_history_goals':initial_history_goals,'initial_history_goal_differs_from_final':bool(np.max(np.abs(np.asarray(initial_history_goals)-goal))>1e-6),'snapshot_history_witness':synthetic_seeds.get(job['id'],'actual generation hook installed before native factory'),'path':str(path),'sha256':sha(path)}
            results.append(row);save(OUT/'status.json',{'state':'AUDITING','cases':len(results),'physics_steps':spent,'optimizer_updates':0})
            # Eight future achieved positions per trajectory, distinct decision endpoints.
            chosen=np.sort(rng.choice(np.arange(max(2,len(decisions)//4),len(decisions)),size=8,replace=False))
            physical=np.asarray(physical)
            for index in chosen:
                future_step=int(arrays['steps'][index]+arrays['native_counts'][index]);new_goal=physical[future_step-1,1:3].copy()
                if bool(physical[future_step-1,7]) or np.linalg.norm(arrays['poses'][0,:2]-new_goal)<=.20:
                    relabel_rows.append({'trajectory':job['id'],'future_step':future_step,'goal_xy':new_goal.tolist(),'eligible':False,'reason':'future point in collision or initial pose already inside radius; no replacement sampling'})
                    continue
                distances=np.linalg.norm(physical[:,1:3]-new_goal,axis=1)
                collision=physical[:,7].astype(bool);hits=np.flatnonzero((distances<=.20)&~collision)
                assert len(hits)>0
                first=int(hits[0]+1)
                # Keep collision priority and truncate the relabeled episode at first success.
                earlier_collision=np.flatnonzero(collision[:first])
                assert not len(earlier_collision),'A physically non-existent post-contact success cannot be used'
                transition=int(np.searchsorted(arrays['steps']+arrays['native_counts'],first,side='left'))
                begin=int(arrays['steps'][transition]);assert begin<first<=begin+arrays['native_counts'][transition]
                rr=rebuild(arrays['states'][transition],arrays['poses'][transition],arrays['history_poses'][transition],new_goal)
                next_rr=rebuild(arrays['next_states'][transition],arrays['next_poses'][transition],arrays['next_history_poses'][transition],new_goal)
                mask=np.ones(113,dtype=bool);mask[48:53]=False
                for j in range(4):mask[72+12*j:74+12*j]=False
                assert np.array_equal(rr[mask],arrays['states'][transition][mask]);assert np.array_equal(next_rr[mask],arrays['next_states'][transition][mask])
                assert rr[112]==arrays['states'][transition,112]
                if first<future_step:assert first!=future_step
                relabel_rows.append({'trajectory':job['id'],'future_step':future_step,'goal_xy':new_goal.tolist(),'eligible':True,'first_success_step':first,'terminal_transition':transition,'terminal_transition_physics':first-begin,'success_before_chosen_future':first<future_step,'within_action_hold':first<begin+int(arrays['native_counts'][transition]),'bootstrap_discount':0.,'endpoint_next_state_is_exact_terminal':first==begin+int(arrays['native_counts'][transition]),'next_state_usage':'For interior first-hit terminal, recorded later hold endpoint must not be presented as true first-hit observation; no bootstrap and no next-state losses. This audit does not emit a ready training replay.'})
            assert spent<=plan['max_physics']
            assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<plan['max_bytes']
        assert all(torch.equal(state[k],v) for k,v in policy.state_dict().items())
        result={'state':'COMPLETE','gate':'PASS_OBSERVATION_AND_TERMINATION_CONTRACT','cases':results,'native_physics_steps':spent,'optimizer_updates':0,'relabel_cases':len(relabel_rows),'eligible_relabel_cases':sum(r['eligible'] for r in relabel_rows),'max_original_observation_rebuild_error':max_error,'interior_hold_first_hits':sum(r.get('within_action_hold',False) for r in relabel_rows),'first_hits_before_selected_future':sum(r.get('success_before_chosen_future',False) for r in relabel_rows),'original5m_safe_success':sum(r['success'] and not r['collision'] for r in results if r['id'].startswith('original')),'ready_training_replay':False,'actor_unchanged':True,'automatic_promotion':False,'incumbent':'T159','scope':'Data contract audit only; future achieved goals are offline labels, never runtime waypoints. Goal-conditioned off-policy training must be declared separately. Initial factory stale-goal history preserved, with actual generation witnesses. Snapshot initial frame metadata separately qualified. No new original5m success.'}
        save(OUT/'relabel_cases.json',relabel_rows);save(OUT/'summary.json',result);save(OUT/'status.json',result);print(json.dumps(result,indent=2))
    except BaseException:
        save(OUT/'failure.json',{'state':'FAILED','traceback':traceback.format_exc(),'physics_steps':spent});raise
    finally:
        Base._history_frame=original_frame
        if env is not None:env.close()

if __name__=='__main__':main()
