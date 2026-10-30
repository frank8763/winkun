"""Native episodes for a fixed policy batch; state writes occur only at reset."""
import os
import sys
import math
import traceback
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
        plan=json.loads((HERE/'protocol.json').read_text());prep=json.loads((OUT/'preparation.json').read_text())
        native,create=native_setup(folder);policy=Policy().eval().requires_grad_(False)
        snapshots={s['id']:read_snapshot(s['path']) for s in prep['snapshots']}
        sources={s['id']:s for s in prep['snapshots']};hold=prep['selected_hold'];rho=math.exp(-hold*.002/plan['noise_correlation_s'])
        while True:
            command=pipe.recv()
            if command is None:break
            job,weights=command
            policy.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in weights.items()})
            rng=np.random.default_rng(job['seed'])
            if env is not None:env.close();env=None
            if 'snapshot_id' in job:
                source=sources[job['snapshot_id']];env,_,actual=create(source['case']);obs=restore(env,snapshots[job['snapshot_id']])
                cap=plan['frontier_step_cap'];env.max_steps=int(env.step_count)+cap;start_distance=float(env._goal_state()[1])
            else:
                env,obs,actual=create(job['case']);cap=job['case']['step_cap'];env.max_steps=cap;start_distance=float(env._goal_state()[1])
            goal=env.goal_xy.copy();steps=0;terminal=trunc=False;prev_state=np.zeros(113,dtype=np.float32);prev_latent=np.zeros(2,dtype=np.float32);prev_mean=np.zeros(2,dtype=np.float32)
            data=[];clip_count=zero_access=0;start_time=float(env.data.time)
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
                total=0.;elapsed_before=steps
                for n in range(1,min(hold,cap-steps)+1):
                    distance=float(env._goal_state()[1]);t0=float(env.data.time)
                    obs,_,terminal,trunc,info=native.step_checked(env,action);obs=np.asarray(obs,dtype=np.float32);steps+=1
                    assert abs(float(env.data.time)-t0-.002)<1e-7
                    finite_end=bool(terminal or trunc)
                    event=10.*bool(info['reach_goal'])-10.*bool(info['collision'])-5.*bool(trunc and not terminal)
                    phi_next=0. if finite_end else -.3*float(env._goal_state()[1])
                    total+=GAMMA**(n-1)*(-.00002+event+GAMMA*phi_next+.3*distance)
                    if finite_end:break
                if job.get('collect',False):data.append((state,prev_state.copy(),prev_latent.copy(),mask,latent.copy(),logp,total,GAMMA**n,n,GAMMA**elapsed_before))
                prev_state=state.copy();prev_latent=latent.copy();prev_mean=mean.copy()
            assert np.array_equal(goal,env.goal_xy)
            result={'id':job['id'],'arm':job['arm'],'source':job.get('snapshot_id',job.get('case',{}).get('id')),
                    'steps':steps,'success':bool(info['reach_goal']),'collision':bool(info['collision']),
                    'timeout':bool(trunc and not terminal),'start_distance_m':start_distance,'remaining_m':float(env._goal_state()[1]),
                    'elapsed_simulation_s':float(env.data.time)-start_time,'stochastic':job['stochastic'],
                    'clipped_decisions':clip_count,'negative_native_throttle_decisions':zero_access,'actual_seed':int(actual)}
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
