"""Compare fresh-model reset with the original native prefix, before training."""
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import fcntl
import json
import numpy as np
import torch
from common import *

def main():
    lock=(OUT/'cross_environment.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'cross_environment_verification.json').exists()
    prep=json.loads((OUT/'preparation.json').read_text());assert prep['state']=='PASS'
    torch.set_num_threads(1);policy=Policy().eval().requires_grad_(False)
    policy.load_state_dict(torch.load(OUT/'initial_policy.pt',map_location='cpu',weights_only=False)['policy'])
    native,create=native_setup(OUT/'cross_preflight');hold=prep['selected_hold'];rows=[];frames=0
    for case_id in sorted({s['case']['id'] for s in prep['snapshots']}):
        sources=[s for s in prep['snapshots'] if s['case']['id']==case_id];case=sources[0]['case'];env=None
        try:
            env,obs,_=create(case);env.max_steps=case['step_cap'];steps=0
            while steps<max(s['prefix_steps']+200 for s in sources):
                if steps%hold==0:
                    with torch.no_grad():action=action_clip(policy(torch.tensor(observation(obs,1-steps/case['step_cap']).reshape(1,-1))).numpy()[0],'legacy_range')
                obs,_,term,trunc,_=native.step_checked(env,action);obs=np.asarray(obs,dtype=np.float32);steps+=1;frames+=1;assert not(term or trunc)
                for source in sources:
                    if steps!=source['prefix_steps']+200:continue
                    expected=capture(env,obs);fresh=None
                    try:
                        fresh,_,_=create(case);s=read_snapshot(source['path']);actual_obs=restore(fresh,s)
                        for j in range(200):
                            if j%hold==0:
                                with torch.no_grad():a=action_clip(policy(torch.tensor(observation(actual_obs,1-(source['prefix_steps']+j)/case['step_cap']).reshape(1,-1))).numpy()[0],'legacy_range')
                            actual_obs,_,te,tr,_=native.step_checked(fresh,a);actual_obs=np.asarray(actual_obs,dtype=np.float32);frames+=1;assert not(te or tr)
                        actual=capture(fresh,actual_obs);state_error=float(np.max(np.abs(actual['physics']-expected['physics'])));obs_error=float(np.max(np.abs(actual_obs-expected['obs'])))
                        assert state_error<1e-7 and obs_error<1e-7,(state_error,obs_error)
                        rows.append({'snapshot':source['id'],'physics_error':state_error,'obs_error':obs_error,'future_steps':200})
                    finally:
                        if fresh is not None:fresh.close()
        finally:
            if env is not None:env.close()
    assert len(rows)==4
    save(OUT/'cross_environment_verification.json',{'state':'PASS','cases':rows,'physics_steps':frames,'actor_updates':0})
    print(json.dumps({'state':'PASS','physics_steps':frames,'snapshots':4}),flush=True)

if __name__=='__main__':main()
