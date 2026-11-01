"""Frozen one-hold control counterfactuals; never training data or deployment."""
from pathlib import Path
import hashlib
import io
import json
import os
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t269_same_state_controls'
sys.path.insert(0,str(ROOT/'stage/experiments/t259_closed_loop_recovery'))
from closed_loop_learning import L,S,C,R,np,torch,policy
import mujoco


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contacts(env):
    rows=[]
    for con in env.data.contact[:env.data.ncon]:
        a,b=int(con.geom1),int(con.geom2)
        ra=int(env.model.geom_bodyid[a]) in env.robot_body_ids
        rb=int(env.model.geom_bodyid[b]) in env.robot_body_ids
        if ra==rb or float(con.dist)>.002:continue
        other=b if ra else a;robot=a if ra else b
        name=mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,other) or ''
        body=mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_BODY,int(env.model.geom_bodyid[other])) or ''
        if any(x in (name+' '+body).lower() for x in ['floor','ground','plane']):continue
        rows.append(dict(robot_geom=mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,robot),other_geom=name,other_body=body,distance_m=float(con.dist)))
    return rows


def command(raw,delta):
    action=raw.copy();action[1]=np.clip(action[1]+np.float32(delta),-1.,1.)
    return action


def preflight(plan):
    assert not (OUT/'preflight.json').exists()
    for path,h in plan['input_hashes'].items():assert sha(path)==h,path
    compile((HERE/'diagnose.py').read_text(),str(HERE/'diagnose.py'),'exec')
    p=policy(plan).eval().requires_grad_(False);assert S.digest(p.state_dict())==plan['policy_hash']
    for a in [np.array([.9,.99],dtype=np.float32),np.array([.9,-.99],dtype=np.float32)]:
        for delta in [0.,-.03,.03]:
            b=command(a,delta);assert b[0]==a[0] and -1<=b[1]<=1
    with np.load(plan['reference']['path']) as z:
        assert int(z['native_counts'].astype(np.int64).sum())==plan['reference']['physics_steps']
        assert set(z.files)==set(plan['reference']['array_hashes'])-{'states','next_states'}
        assert all(S.digest(z[k])==plan['reference']['array_hashes'][k] for k in z.files)
    bound=8*plan['episode_bytes']+2*plan['snapshot_bytes']+plan['metadata_bytes']
    assert bound<=plan['final_bytes']
    assert S.used(ROOT/'stage/runs')-S.used(OUT)+plan['final_bytes']<plan['global_bytes']
    C.save(OUT/'preflight.json',dict(state='PASS',physics_steps=0,optimizer_calls=0,
        policy_hash=plan['policy_hash'],original_compact_payload_arrays_verified=True,
        one_hold_command_clip_contract=True,full_worst_case_bytes=bound,
        protocol_sha256=sha(HERE/'protocol.json'),native_state_restore_not_yet_tested=True))


def main():
    plan=json.loads((HERE/'protocol.json').read_text())
    if '--preflight' in sys.argv:return preflight(plan)
    assert not (OUT/'status.json').exists(),'No duplicate/restart or intervention grid expansion'
    assert json.loads((OUT/'preflight.json').read_text())['protocol_sha256']==sha(HERE/'protocol.json')
    for path,h in plan['input_hashes'].items():assert sha(path)==h,path
    start=time.time();deadline=start+plan['wall_seconds'];physics=commands=0;rows=[];env=None;reason='COMPLETE';contexts={}
    def guard():
        assert time.time()<deadline,'WALL'
        assert physics<=plan['max_physics'] and commands<=plan['max_commands'],'COUNTS'
        assert S.used(OUT)<plan['final_bytes'] and S.used(ROOT/'stage/runs')<plan['global_bytes'],'STORAGE'
        assert shutil.disk_usage(ROOT).free>=plan['min_free_bytes']
    def status(label):C.save(OUT/'status.json',dict(state=label,pid=os.getpid(),physics_steps=physics,commands=commands,cases=len(rows),optimizer_calls=0,updated_epoch=time.time(),elapsed_s=time.time()-start))
    def simulate(label,begin=0,delta=0.,capture=False):
        nonlocal env,physics,commands
        guard();cap=plan['job']['case']['step_cap']
        assert physics+cap-begin<=plan['max_physics'] and commands+1<=plan['max_commands']
        commands+=1
        env,obs,_=R.task_env(create,plan['job']);obs=C.perturb_anchor(env,obs,plan['job'])
        previous=np.zeros(2,dtype=np.float32);previous_count=0
        if begin:
            saved=C.read_snapshot(OUT/f'anchor_{begin}.npz');obs=C.restore(env,saved)
            context=json.loads((OUT/f'context_{begin}.json').read_text())
            previous=np.asarray(context['previous_action'],dtype=np.float32);previous_count=context['previous_count']
            assert np.array_equal(R.integration(env),saved['physics']) and env.step_count==begin
            packet=L.packet(env,obs,begin,cap,previous,previous_count)
            assert S.digest(packet)==context['packet120_sha256'],'Incomplete causal Actor context'
        scene=C.model_hash(env);goal=env.goal_xy.copy();steps=begin;term=trunc=False;info={};records=[];intervention=None
        while not(term or trunc) and steps<cap:
            guard()
            x=L.packet(env,obs,steps,cap,previous,previous_count)
            if capture and steps in plan['snapshot_steps']:
                dest=OUT/f'anchor_{steps}.npz';assert not dest.exists()
                C.write_snapshot(dest,C.capture(env,obs));assert dest.stat().st_size<=plan['snapshot_bytes']
                context=dict(absolute_step=steps,previous_action=previous.tolist(),previous_count=previous_count,
                    packet120_sha256=S.digest(x),native_integration_sha256=R.arrays_hash(R.integration(env)),
                    initial_pose=list(env._pose()[:3]),goal_xy=goal.tolist())
                C.save(OUT/f'context_{steps}.json',context);contexts[str(steps)]=context
            with torch.no_grad():action=p.step(torch.tensor(x[None]))[0][0].numpy()
            if steps==begin and delta!=0:
                raw=action.copy();action=command(action,delta)
                intervention=dict(absolute_step=steps,original_action=raw.tolist(),delivered_action=action.tolist(),steering_delta=delta,maximum_hold_steps=100,diagnostic_only=True)
            pose=np.asarray(env._pose()[:3]);d0=float(env._goal_state()[1]);before=steps
            for _ in range(min(100,cap-steps)):
                obs,_,term,trunc,info=native.step_checked(env,action);steps+=1;physics+=1
                if term or trunc:break
            n=steps-before;d1=float(env._goal_state()[1]);disc=R.GAMMA**n;terminal=term or trunc
            event=float(info['reach_goal'])-float(info['collision'])-.5*float(trunc)
            reward=-.000002*(1-disc)/(1-R.GAMMA)+.03*d0-disc*.03*d1*(not terminal)+(R.GAMMA**(n-1)*event if terminal else 0)
            nxt=L.packet(env,obs,steps,cap,action,n)
            records.append((x,action,np.zeros(2,dtype=np.float32),0.,n,nxt,pose,np.asarray(env._pose()[:3]),[info['collision'],info['reach_goal'],trunc],reward))
            previous=action.copy();previous_count=n
        names=['states','actions','means','logp','native_counts','next_states','poses','next_poses','end_flags','rewards']
        arrays={key:np.asarray([r[i] for r in records]) for i,key in enumerate(names)}
        arrays['native_counts']=arrays['native_counts'].astype(np.uint8);arrays['end_flags']=arrays['end_flags'].astype(np.uint8)
        assert C.model_hash(env)==scene and np.array_equal(goal,env.goal_xy)
        witness=contacts(env) if info.get('collision') else []
        if info.get('collision'):assert witness,'Native collision lacks recorded contact witness'
        buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);b=buffer.getvalue();assert len(b)<=plan['episode_bytes']
        dest=OUT/(label+'.npz');assert not dest.exists();dest.write_bytes(b)
        row=dict(id=label,start_step=begin,absolute_end_step=steps,physics_steps=steps-begin,
            complete=bool(term or trunc),success=bool(info.get('reach_goal')),collision=bool(info.get('collision')),timeout=bool(trunc),
            remaining_m=d1,array_hashes={k:S.digest(v) for k,v in arrays.items()},integration_sha256=R.arrays_hash(R.integration(env)),
            contact_witness=witness,terminal_pose=list(env._pose()[:3]),
            measured_actuator_context=arrays['next_states'][-1,113:117].tolist(),
            terminal_qvel=env.data.qvel.tolist(),intervention=intervention,
            path=str(dest),sha256=sha(dest),policy_hash=plan['policy_hash'],not_on_policy_training_data=True)
        env.close();env=None;rows.append(row)
        with (OUT/'cases.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        status(label);return row,arrays
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        p=policy(plan).eval().requires_grad_(False);assert S.digest(p.state_dict())==plan['policy_hash']
        native,create=S.native_setup(OUT/'native');status('STARTING')
        C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[],single_process_native=True))
        first,source=simulate('source_first',capture=True)
        ref=plan['reference'];assert first['array_hashes']==ref['array_hashes'] and first['integration_sha256']==ref['integration_sha256']
        second,again=simulate('source_exact_repeat')
        assert first['array_hashes']==second['array_hashes'] and first['integration_sha256']==second['integration_sha256']
        assert all(np.array_equal(v,again[k]) for k,v in source.items())
        for begin in plan['snapshot_steps']:
            control,a=simulate(f'control_{begin}',begin)
            assert control['integration_sha256']==first['integration_sha256']
            assert all(np.array_equal(v,source[k][begin//100:]) for k,v in a.items())
        C.save(OUT/'restoration_gate.json',dict(state='PASS',both_complete_suffixes_all10arrays_and_integration_exact=True,
            both120_contexts_exact=True,no_intervention_simulated_before_gate=True,snapshots=contexts))
        for begin in plan['snapshot_steps']:
            for delta in plan['steering_deltas']:
                simulate(f'branch_{begin}_{"plus" if delta>0 else "minus"}',begin,delta)
        assert len(rows)==8 and S.digest(p.state_dict())==plan['policy_hash']
    except BaseException:
        reason='FAILED';C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physics_steps=physics,commands=commands))
    finally:
        if env is not None:env.close();env=None
    result=dict(state=reason,physics_steps=physics,commands=commands,optimizer_calls=0,
        cases=len(rows),branches=[r for r in rows if r['intervention']],
        restoration_gate=(OUT/'restoration_gate.json').exists(),all_native_envs_closed=True,all_workers_stopped=True,
        policy_hash=plan['policy_hash'],checkpoint_sha256=sha(plan['checkpoint']),elapsed_s=time.time()-start,
        own_bytes=S.used(OUT),global_bytes=S.used(ROOT/'stage/runs'),automatic_next_training=False,
        interpretation='One-hold same-state interventions are diagnostic, not deployable routing, an improved Actor, or independent training seeds.')
    C.save(OUT/'summary.json',result);C.save(OUT/'status.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
