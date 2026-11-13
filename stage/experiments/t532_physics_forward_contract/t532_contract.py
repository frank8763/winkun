"""First, zero-native gate of the physical-surrogate proposal. No navigation fit."""
from pathlib import Path
import fcntl, hashlib, json, os, shutil, sys, time, traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t532_physics_forward_contract';DOC=ROOT/'stage/docs/day_20261113'
sys.path.insert(0,str(HERE));import t532_model as model


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);tmp=Path(str(p)+'.partial');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def used(folder):return sum(f.stat().st_size for f in Path(folder).rglob('*') if f.is_file())
def backend():
    sys.path.insert(0,str(ROOT/'stage/experiments/t501_joint_command_repair'))
    import t501_learning as A
    A.torch.set_num_threads(1);A.torch.use_deterministic_algorithms(True)
    # A hard guard: this experiment may deserialize an MJB, never step it.
    import mujoco
    def forbidden(*args,**kwargs):raise AssertionError('T532_ZERO_NATIVE_CONTRACT')
    mujoco.mj_step=forbidden;mujoco.mj_step1=forbidden;mujoco.mj_step2=forbidden
    return A
def verify(p):
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert time.time()<p['absolute_all_stop']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and set(os.sched_getaffinity(0))<={14,15}
    assert used(OUT)<=p['max_bytes'] and used(ROOT/'stage/runs')-used(OUT)+p['max_bytes']<=2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30
    for f,h in p['inputs'].items():assert sha(f)==h,f


def static_geometry():
    import mujoco,numpy as np
    base=ROOT/'stage/runtime/mk-mini/training/e2e_phase10_lidar_v1/assets'
    m=mujoco.MjModel.from_binary_path(str(base/'phase10_original_scene.mjb'))
    # Static relative body transforms only: no mj_forward or physics calls.
    poses={0:(np.zeros(3),np.eye(3))}
    for b in range(1,m.nbody):
        pp,pr=poses[int(m.body_parentid[b])];r=np.empty(9);mujoco.mju_quat2Mat(r,m.body_quat[b]);poses[b]=(pp+pr@m.body_pos[b],pr@r.reshape(3,3))
    wheel={};radii=[];root=None
    for name in ['left_front','right_front','left_rear','right_rear']:
        j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name+'_wheel_joint');assert j>=0
        b=int(m.jnt_bodyid[j]);rr=int(m.body_rootid[b]);root=rr if root is None else root;assert rr==root
        origin,rotation=poses[root];wheel[name]=rotation.T@(poses[b][0]-origin)
        if 'rear' in name:
            gids=[g for g in range(int(m.body_geomadr[b]),int(m.body_geomadr[b]+m.body_geomnum[b])) if (m.geom_contype[g] or m.geom_conaffinity[g]) and m.geom_type[g] in [mujoco.mjtGeom.mjGEOM_CYLINDER,mujoco.mjtGeom.mjGEOM_CAPSULE]]
            assert len(gids)==1;radii.append(float(m.geom_size[gids[0],0]))
    rear=(wheel['left_rear']+wheel['right_rear'])/2.;front=(wheel['left_front']+wheel['right_front'])/2.
    length=float(np.linalg.norm(front[:2]-rear[:2]));assert abs(length-.60)<1e-10 and radii[0]==radii[1]
    with np.load(base/'phase10_original_scene_state.npz') as s:maxsteer=float(s['max_steer'][0]);logging_gain=float(s['bicycle_speed_gain'][0])
    return dict(wheelbase_m=length,rear_center_body_xy=rear[:2].tolist(),rear_radius_m=radii[0],max_wheel_rad_s=5.,max_steer_rad=maxsteer,logging_speed_gain_NOT_used=logging_gain,wheel_body_positions={k:v.tolist() for k,v in wheel.items()},mujoco_version=mujoco.__version__,native_physics=0,approximations=['instantaneous tire rolling without slip','instantaneous steering and wheel speed','planar rigid chassis','recorded control sequence, NOT feedback observation rollout'])


def calculate(p):
    A=backend();np,torch=A.np,A.torch;source=read(p['actor_configuration']);result=[];payload={}
    for index,window in enumerate(p['windows']):
        verify(p);r=window['row'];a=A.G.E.N.S.arrays(r['path'])
        assert all(A.digest(v)==r['array_hashes'][k] for k,v in a.items())
        q=A.frozen(source,r['request']['model']);before=q.phash();assert before==r['policy_hash']
        start=window['first'];stop=start+p['holds'];assert len(a['states'])>=stop and a['states'].shape[1]==120
        assert not a['end_flags'][:stop].any() and not a['stochastic'][:stop].any()
        assert np.all(a['native_counts'][:stop]==100)
        # Regenerate the actual full causal prefix, including recurrent state.
        hidden=torch.zeros((1,1,64),dtype=torch.float64);maxloc=0.;maxhidden=0.
        with torch.no_grad():
            for k,x in enumerate(a['states'][:stop]):
                maxhidden=max(maxhidden,float(np.max(np.abs(hidden.numpy().reshape(64)-a['hidden_before'][k].reshape(64)))))
                n=q.normalized(torch.tensor(x)[None]);h,hidden=q.core(q.encoder(n)[None],hidden);loc=q.location(h[0],n)[0]
                maxloc=max(maxloc,float(np.max(np.abs(loc.numpy()-a['locations'][k]))))
                assert np.array_equal(loc.clamp(-1,1).float().numpy(),a['actions'][k]),(window['id'],k)
        assert maxloc<=1e-10 and maxhidden<=1e-10 and q.phash()==before
        dt=a['native_counts'][start:stop].astype(float)*.002;u=a['actions'][start:stop];pose=a['poses'][start];target=a['next_poses'][start:stop]
        pred=model.rollout(pose,u,dt,p['geometry'])
        # Velocity channels are from the current, past-generated observation;
        # no next pose or future velocity initializes either candidate.
        body_v=a['states'][start,53:55].astype(float);rate=float(a['states'][start,55])*2.
        baseline=model.constant_twist(pose,body_v,rate,dt)
        one=np.stack([model.advance(s,c,t,p['geometry']) for s,c,t in zip(a['poses'][start:stop],u,dt)])
        err=model.errors(pred,target);baseerr=model.errors(baseline,target);oneerr=model.errors(one,target)
        qualified=err['max_xy_m']<=p['max_xy_error_m'] and err['max_yaw_deg']<=p['max_yaw_error_deg']
        result.append(dict(id=window['id'],source_path=r['path'],source_sha256=r['sha256'],source_policy_hash=before,start_seconds=float(a['native_counts'][:start].sum())*.002,seconds=float(dt.sum()),source_outcome=dict(success=r['success'],collision=r['collision'],timeout=r['timeout']),full_prefix_action_exact=True,hidden_max_error=maxhidden,location_max_error=maxloc,primary_quasistatic=err,causal_constant_twist_control=baseerr,one_step_teacher_forced_DIAGNOSTIC_ONLY=oneerr,forward_gate=qualified,initial_body_v=body_v.tolist(),initial_yaw_rate=rate,source_midpoint_full_integration_saved=False))
        for name,v in dict(predicted=pred,actual=target,control=baseline,one_step=one,actions=u,dt=dt,start_pose=pose,initial_velocity=np.r_[body_v,rate]).items():payload[f'w{index}_{name}']=v
    forward=all(x['forward_gate'] for x in result)
    return dict(windows=result,forward_gate=forward,direction_gate=None,Actor_learning_eligible=False,closed_loop_surrogate_NOT_implemented=True,new_native=0,new_Actor_Adam=0,new_Value_Adam=0,new_model_fit=0,new_policy_gradient_updates=0,reason='FORWARD_GATE_PASS_LOCAL_DERIVATIVE_STILL_UNTESTED' if forward else 'UNFITTED_QUASISTATIC_MODEL_REJECTED_NO_ACTOR_LEARNING',not_a_test_of_all_differentiable_models=True),payload


def prepare():
    assert not OUT.exists();A=backend();prior=ROOT/'stage/runs/t531_joint_success_distillation'
    seal=read(prior/'SEALED.json');assert all(not Path('/proc',str(i)).exists() for i in seal['scientific_pids']+[read(prior/'launch.json')['pid']])
    assert read(prior/'pilot/audit.json')['state']=='PASS'
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T531.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    d=read(DOC/'T530_DIVERGENCE.json');newparent=read(ROOT/'stage/runs/t526_early_feedback_qualification/protocol.json')['baseline_row']
    windows=[dict(id='old_success',row=d['parent_row'],first=24),dict(id='old_failure',row=d['old_row'],first=24),dict(id='new_parent_failure',row=newparent,first=321),dict(id='new_candidate_success',row=d['new_success_row'],first=321)]
    end=1794562200.;assert time.strftime('%Y-%m-%d %H:%M',time.localtime(end))=='2026-11-13 17:30'
    assert time.time()+1200<end
    inputs={str(f):sha(f) for f in HERE.glob('*.py')};source=prior/'protocol.json';old=read(source)
    inputs.update({f:h for f,h in old['inputs'].items() if Path(f).suffix=='.py'})
    for w in windows:
        r=w['row'];inputs[r['path']]=r['sha256'];inputs[r['request']['model']]=r['loaded_model_sha256']
    base=ROOT/'stage/runtime/mk-mini/training/e2e_phase10_lidar_v1/assets'
    for f in [source,base/'phase10_original_scene.mjb',base/'phase10_original_scene_state.npz',DOC/'T532_PLAN.md']:inputs[str(f)]=sha(f)
    p=dict(id='T532_PHYSICS_FORWARD_ADMISSIBILITY',actor_configuration=str(source),windows=windows,holds=5,max_xy_error_m=.02,max_yaw_error_deg=1.,geometry=static_geometry(),inputs=inputs,max_bytes=1024*1024,main_seconds=300,audit_seconds=300,transport_seconds=300,source_seconds=180,absolute_all_stop=end,absolute_native_stop=end-900,absolute_learning_stop=end-1800,max_native_physics=0,max_native_commands=0,max_Actor_Adam=0,max_Value_Adam=0,max_model_fit=0,stage_cpus=[14,15],GPU=False,CPUQuota='100%',scope='Existing-data first contract gate ONLY. Recorded commands drive unfitted physical surrogate, no changing-sensor/GRU forward loop. Neither native parameter directions nor new navigation evaluated. No model selection or threshold sweep. All four windows must pass to justify a separately implemented directional diagnostic; no automatic Actor training.',authorization='User accepted revised small mechanism-first experiment. Stage only. No native training restart or extension of 17:30 window.')
    assert used(ROOT/'stage/runs')+p['max_bytes']+65536<2**30
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);verify(p)
    save(DOC/'T532_BUDGET.json',dict(own_max_bytes=p['max_bytes'],global_reserved_bytes=used(ROOT/'stage/runs')-used(OUT)+p['max_bytes'],free_bytes=shutil.disk_usage(ROOT).free,new_native=0,ActorAdam=0,ValueAdam=0,model_fit=0,main_seconds=300,audit_seconds=300,backup_seconds=300,source_seconds=180,all_stop=end))
    print('T532_REGISTERED_NOT_LAUNCHED',flush=True)


def preflight():
    p=read(OUT/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists()
    checks=model.fixtures();assert static_geometry()==p['geometry']
    # No scientific prediction errors are observed before protocol locking.
    save(OUT/'preflight.json',dict(**checks,pid=os.getpid(),static_geometry_deserialization=True,no_scene_created=True,real_trajectory_metrics_NOT_evaluated=True))
    print('T532_PREFLIGHT_PASS',flush=True)


def main():
    p=read(OUT/'protocol.json');verify(p);assert not (OUT/'summary.json').exists();start=time.time()
    with (ROOT/'stage/runs/.stage_writer.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result,payload=calculate(p);import numpy as np
        np.savez_compressed(OUT/'predictions.npz',**payload);assert (OUT/'predictions.npz').stat().st_size<32768
        save(OUT/'summary.json',dict(state='COMPLETE',pid=os.getpid(),elapsed_s=time.time()-start,result=result,payload_sha256=sha(OUT/'predictions.npz')));verify(p)
    print(result['reason'],flush=True)


def audit():
    p=read(OUT/'protocol.json');verify(p);s=read(OUT/'summary.json');assert not Path('/proc',str(s['pid'])).exists() and not (OUT/'audit.json').exists()
    result,payload=calculate(p);assert result==s['result'];import numpy as np
    with np.load(OUT/'predictions.npz') as z:
        assert set(z.files)==set(payload)
        for k,v in payload.items():assert np.array_equal(z[k],v),k
    assert sha(OUT/'predictions.npz')==s['payload_sha256']
    save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,new_optimizer=0,all_source_arrays_hidden_actions_and_prediction_metrics_recomputed=True,scientific_forward_gate=result['forward_gate'],local_direction_gate=None,Actor_learning_eligible=False,source_models_unchanged=True));print('T532_AUDIT_PASS',flush=True)


if __name__=='__main__':globals()[sys.argv[1]]()
