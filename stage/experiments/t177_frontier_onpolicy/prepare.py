"""Frozen native-command embedding gate and exact reset-state validation."""
from pathlib import Path
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import json
import time
import fcntl
import numpy as np
import torch
from common import *

def main():
    lock=(OUT/'prepare.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'preparation.json').exists()
    plan=json.loads((HERE/'prepare_protocol.json').read_text())
    for p,h in plan['input_hashes'].items():assert sha(p)==h,p
    torch.set_num_threads(1);policy=Policy().eval().requires_grad_(False)
    old=torch.load(plan['incumbent_checkpoint'],map_location='cpu',weights_only=False)['actor'];policy.initialize(old)
    legacy_actor=CORE.Actor().eval().requires_grad_(False);legacy_actor.load_state_dict(old)
    native,create=native_setup(OUT/'prepare');start=time.monotonic();rows=[];frames=0;max_action_error=0.;selected=None
    for hold in [100,25]:
        score=[]
        for case in plan['cases'][:6]:
            env=None
            try:
                env,obs,actual=create(case);env.max_steps=case['step_cap'];steps=0
                while steps<case['step_cap']:
                    assert time.monotonic()-start<900
                    with torch.no_grad():
                        mean=policy(torch.tensor(observation(obs,1-steps/case['step_cap']).reshape(1,-1))).numpy()[0]
                        teacher=legacy_actor(torch.tensor(obs.reshape(1,-1))).numpy()[0]
                    expected=np.array([.5*(teacher[0]+1),teacher[1]],dtype=np.float32)
                    max_action_error=max(max_action_error,float(np.abs(mean-expected).max()));assert max_action_error<1e-5
                    for _ in range(min(hold,case['step_cap']-steps)):
                        t0=float(env.data.time);obs,_,terminal,trunc,info=native.step_checked(env,action_clip(mean,'legacy_range'));obs=np.asarray(obs,dtype=np.float32)
                        assert abs(float(env.data.time)-t0-.002)<1e-7;steps+=1;frames+=1
                        if terminal or trunc:break
                    if terminal or trunc:break
                row={'case':case['id'],'hold':hold,'safe_success':bool(info['reach_goal'] and not info['collision']),'steps':steps,'remaining_m':float(env._goal_state()[1])};rows.append(row);score.append(row['safe_success'])
            finally:
                if env is not None:env.close()
        if all(score):selected=hold;break
    assert selected is not None,'Native-command embedding/cadence gate failed; no training'
    torch.save({'policy':policy.state_dict(),'incumbent_sha256':sha(plan['incumbent_checkpoint']),'hold':selected,'actor_training':False},OUT/'initial_policy.pt')
    snapshot_rows=[]
    for case in plan['cases'][6:]:
        env=None
        try:
            env,obs,actual=create(case);env.max_steps=case['step_cap'];steps=0;states={}
            while steps<case['step_cap']:
                assert time.monotonic()-start<900
                states[steps]=capture(env,obs)
                with torch.no_grad():mean=policy(torch.tensor(observation(obs,1-steps/case['step_cap']).reshape(1,-1))).numpy()[0]
                for _ in range(min(selected,case['step_cap']-steps)):
                    obs,_,terminal,trunc,info=native.step_checked(env,action_clip(mean,'legacy_range'));obs=np.asarray(obs,dtype=np.float32);steps+=1;frames+=1
                    if terminal or trunc:break
                if terminal or trunc:break
            assert info['collision'],'Expected source failure changed; reconsider curriculum'
            end=steps
            for seconds_before in [12,6]:
                prefix=((end-int(seconds_before/.002))//selected)*selected
                assert prefix>=0 and prefix+200 in states
                snap=states[prefix];obs2=restore(env,snap);expected=states[prefix+200]
                for step in range(200):
                    if step%selected==0:
                        with torch.no_grad():mean=policy(torch.tensor(observation(obs2,1-(prefix+step)/case['step_cap']).reshape(1,-1))).numpy()[0]
                    obs2,_,term,trunc2,inf=native.step_checked(env,action_clip(mean,'legacy_range'));obs2=np.asarray(obs2,dtype=np.float32);frames+=1
                    assert not(term or trunc2)
                actual_state=capture(env,obs2)
                error=float(np.max(np.abs(actual_state['physics']-expected['physics'])))
                obs_error=float(np.max(np.abs(obs2-expected['obs'])))
                assert error<1e-7 and obs_error<1e-7,('Reset fidelity failed',error,obs_error)
                path=OUT/(case['side']+f'_before{seconds_before}s.npz');write_snapshot(path,snap)
                snapshot_rows.append({'id':case['side']+f'_before{seconds_before}s','path':str(path),'sha256':sha(path),'case':case,
                                      'prefix_steps':prefix,'source_collision_steps':end,'time_before_collision_s':(end-prefix)*.002,
                                      'remaining_m':float(snap['obs'][48]*10),'physics_clone_max_error':error,'obs_clone_max_error':obs_error})
        finally:
            if env is not None:env.close()
    assert frames<=plan['physics_budget']
    save(OUT/'preparation.json',{'state':'PASS','selected_hold':selected,'old_cases':rows,'physical_embedding_max_action_error':max_action_error,
                               'snapshots':snapshot_rows,'initial_policy_sha256':sha(OUT/'initial_policy.pt'),'physics_steps':frames,
                               'time_feature':'113th value, remaining episode budget fraction; initialfirstlayerweightzero',
                               'no_qpos_qvel_writes_during_navigation':True,'elapsed_s':time.monotonic()-start})
    print(json.dumps({'state':'PASS','hold':selected,'snapshots':len(snapshot_rows),'physics_steps':frames}),flush=True)

if __name__=='__main__':main()
