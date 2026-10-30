"""Native episodes for a fixed policy batch; state writes occur only at reset."""
import os
import sys
import math
import traceback
import hashlib
import mujoco
import numpy as np
import torch
from common import *

GAMMA=math.exp(-.002/60.)

def run(pipe,index,deadline):
    folder=OUT/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True)
    log=(folder/'worker.log').open('a',buffering=1);sys.stdout=sys.stderr=log
    env=None
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        plan=json.loads((HERE/'protocol.json').read_text());prep={'selected_hold':100,'snapshots':[]}
        native,create=native_setup(folder);policy=Policy().eval().requires_grad_(False)
        snapshots={s['id']:read_snapshot(s['path']) for s in prep['snapshots']}
        sources={s['id']:s for s in prep['snapshots']};hold=prep['selected_hold']
        while True:
            command=pipe.recv()
            if command is None:break
            job,weights=command
            rho=math.exp(-hold*.002/job.get('noise_correlation_s',1.0))
            policy.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in weights.items()})
            rng=np.random.default_rng(job['seed'])
            if env is not None:env.close();env=None
            if 'snapshot_side' in job:
                side=job['snapshot_side'];env,_,actual=create(plan['anchor_sources'][side]['case']);obs=restore(env,read_snapshot(plan['anchor_sources'][side]['snapshot_path']))
                obs=perturb_anchor(env,obs,job);cap=plan['anchor_step_cap'];env.max_steps=cap
                assert not env._collision() and np.isfinite(obs).all(),'Invalid reset: stop, do not resample'
                full=np.zeros(mujoco.mj_stateSize(env.model,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(env.model,env.data,full,mujoco.mjtState.mjSTATE_INTEGRATION)
                signature=hashlib.sha256(full.tobytes()+obs.tobytes()+env.goal_xy.tobytes()).hexdigest()
            else:
                env,obs,actual=create(job['case']);cap=job['case']['step_cap'];env.max_steps=cap
            full=np.zeros(mujoco.mj_stateSize(env.model,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(env.model,env.data,full,mujoco.mjtState.mjSTATE_INTEGRATION)
            signature=hashlib.sha256(full.tobytes()+obs.tobytes()+env.goal_xy.tobytes()).hexdigest()
            start_distance=float(env._goal_state()[1])
            goal=env.goal_xy.copy();steps=0;terminal=trunc=False;prev_state=np.zeros(113,dtype=np.float32);prev_latent=np.zeros(2,dtype=np.float32);prev_mean=np.zeros(2,dtype=np.float32)
            data=[];clip_count=zero_access=0;start_time=float(env.data.time)
            nearest=start_distance;telemetry=[];zero_drive=0;low_drive=0
            while not(terminal or trunc):
                assert __import__('time').monotonic()<deadline
                state=observation(obs,1-steps/cap);mask=float(steps>0)
                with torch.no_grad():
                    mean=policy(torch.tensor(state.reshape(1,-1))).numpy()[0]
                    sigma=policy.logstd.clamp(-1.6,0.).exp().numpy()
                    conditional=mean+rho*(prev_latent-prev_mean)*mask
                    scale=sigma*(math.sqrt(1-rho*rho) if mask else 1.)
                    if job['stochastic']:
                        latent=(conditional+scale*rng.normal(size=2)).astype(np.float32)
                        logp=float(torch.distributions.Normal(torch.tensor(conditional),torch.tensor(scale)).log_prob(torch.tensor(latent)).sum())
                        action=action_clip(latent,job['arm'])
                    else:
                        latent=mean.copy();logp=0.;action=action_clip(mean,job['arm'])
                clip_count+=int(np.any(np.abs(action-latent)>1e-7));zero_access+=int(action[0]<0)
                steer,wheel,_,_=env._policy_to_commands(action)
                telemetry.append([steps*.002,*env._pose()[:3],*mean,*action,wheel,float(np.linalg.norm(env.data.qvel[:2])),float(env._goal_state()[1])])
                zero_drive+=int(wheel<1e-8);low_drive+=int(wheel<.1)
                total=0.;elapsed_before=steps
                for n in range(1,min(hold,cap-steps)+1):
                    distance=float(env._goal_state()[1]);t0=float(env.data.time)
                    obs,_,terminal,trunc,info=native.step_checked(env,action);obs=np.asarray(obs,dtype=np.float32);steps+=1
                    assert abs(float(env.data.time)-t0-.002)<1e-7
                    nearest=min(nearest,float(env._goal_state()[1]));finite_end=bool(terminal or trunc)
                    event=10.*bool(info['reach_goal'])-10.*bool(info['collision'])-5.*bool(trunc and not terminal)
                    phi_next=0. if finite_end else -.3*float(env._goal_state()[1])
                    total+=GAMMA**(n-1)*(-.00002+event+GAMMA*phi_next+.3*distance)
                    if finite_end:break
                if job.get('collect',False):data.append((state,prev_state.copy(),prev_latent.copy(),mask,latent.copy(),logp,total,GAMMA**n,n,GAMMA**elapsed_before))
                prev_state=state.copy();prev_latent=latent.copy();prev_mean=mean.copy()
            assert np.array_equal(goal,env.goal_xy)
            result={'id':job['id'],'arm':job['arm'],'source':job.get('source_id',job.get('snapshot_id',job.get('case',{}).get('id'))),
                    'noise_correlation_s':job.get('noise_correlation_s',1.0),'steps':steps,'success':bool(info['reach_goal']),'collision':bool(info['collision']),
                    'timeout':bool(trunc and not terminal),'start_distance_m':start_distance,'remaining_m':float(env._goal_state()[1]),
                    'elapsed_simulation_s':float(env.data.time)-start_time,'stochastic':job['stochastic'],
                    'clipped_decisions':clip_count,'negative_native_throttle_decisions':zero_access,'actual_seed':int(actual),'course_bin':job.get('course_bin'),'initial_reset_signature':signature,'reset_deltas':[job.get('delta_x_m',0.),job.get('delta_y_m',0.),job.get('delta_yaw_deg',0.)],'reset_group':job.get('reset_group','primary')}
            trace=np.asarray(telemetry,dtype=np.float64)
            contacts=[]
            for contact in env.data.contact[:env.data.ncon]:
                g1,g2=int(contact.geom1),int(contact.geom2)
                r1=int(env.model.geom_bodyid[g1]) in env.robot_body_ids;r2=int(env.model.geom_bodyid[g2]) in env.robot_body_ids
                if r1!=r2 and float(contact.dist)<=.002:
                    contacts.append({'geom1':mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,g1),'geom2':mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,g2),'distance_m':float(contact.dist)})
            result.update(nearest_m=nearest,decisions=len(trace),terminal_contacts=contacts,final_pose=list(env._pose()),
                          native_mean_min=trace[:,4:6].min(0).tolist(),native_mean_max=trace[:,4:6].max(0).tolist(),
                          native_action_min=trace[:,6:8].min(0).tolist(),native_action_max=trace[:,6:8].max(0).tolist(),
                          wheel_command_quantiles=np.quantile(trace[:,8],[0,.1,.5,.9,1]).tolist(),
                          observed_speed_quantiles=np.quantile(trace[:,9],[0,.1,.5,.9,1]).tolist(),zero_drive_decisions=zero_drive,low_drive_decisions=low_drive)
            if job.get('collect',False) or job.get('case',{}).get('distance_m')==5:
                directory=OUT/'telemetry';directory.mkdir(exist_ok=True)
                name=job.get('model_tag',job['arm'])+'_'+job['id']
                path=directory/(name+'.npz');assert not path.exists()
                np.savez_compressed(path,trace=trace,columns=np.asarray(['time_s','x','y','yaw','mean_throttle','mean_steer','throttle','steer','wheel_command','speed_m_s','distance_m']))
                result['telemetry_path']=str(path);result['telemetry_sha256']=sha(path)
            if data:
                values=[];value=0.
                for row in reversed(data):value=row[6]+row[7]*value;values.append(value)
                values.reverse()
                arrays={k:np.asarray([row[i] for row in data],dtype=np.float32) for i,k in enumerate(['states','previous_states','previous_latents','previous_mask','latents','old_logp','rewards','discounts','native_counts','policy_weights'])}
                arrays['returns']=np.asarray(values,dtype=np.float32);result['trajectory']=arrays;result['initial_return']=values[0]
                discount=GAMMA**steps;base=-.00002*(1-discount)/(1-GAMMA)+GAMMA**(steps-1)*(10 if result['success'] else -10 if result['collision'] else -5)+.3*start_distance
                assert abs(base-values[0])<1e-7
            pipe.send(result)
    except (EOFError,SystemExit):pass
    except BaseException:
        try:pipe.send({'error':traceback.format_exc()})
        except OSError:pass
        raise
    finally:
        if env is not None:env.close()
        pipe.close();log.close()
