"""One preselected historical neural Actor on the hard-right task, no noise."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent

def main():
    source=ROOT.parent/'stage_t148/train.py'
    spec=importlib.util.spec_from_file_location('t154_source',source)
    t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
    donor=Path('/home/linghu/mkmini_stage_td3_t146_rightturn_continuation_20261029_v1/checkpoints/episode_0724.pt')
    expected='7bbbcedd75ff4997015bc9878426ba76433fe7ad2156775d01877c14db5d1926'
    assert t.sha(donor)==expected
    assert not (ROOT/'result.json').exists()
    (ROOT/'plan.json').write_text(json.dumps({'experiment':'STAGE_T154_PRESELECTED_DONOR_ACTOR_TEST',
        'donor':str(donor),'sha256':expected,'selection_rule':'same most-recent successful training episode donor selected before the test',
        'seed':2029654802,'angle':18.8,'distance_m':3.0,'step_cap':14000,
        'exploration':False,'training':False,'checkpoint_sweep':False},indent=2)+'\n')
    os.environ['C98_JOB']=os.environ['TD3_JOB']=str(ROOT)
    os.environ['C22_SCREEN_ONLY']='0'
    torch.set_num_threads(1)
    core=t.load_module(t.CORE,'t154_core');actor=core.Actor(112,2)
    ck=torch.load(donor,map_location='cpu',weights_only=False)
    actor.load_state_dict(ck['actor']);actor.eval();actor.requires_grad_(False)
    helper=t.load_module(t.HELPER,'t154_helper');legacy=t.load_module(t.STAGE,'t154_legacy')
    for p,h in legacy.EXPECTED.items():assert t.sha(p)==h
    runner=legacy.module(legacy.SRC,'t154_runner')
    _,v,mod,prep,_=runner.load_native();N=prep[3]
    make=legacy.make_short_task(runner.make_task)
    env=renderer=None
    states=[];actions=[];next_states=[];dones=[]
    try:
        env,renderer,obs,seed=helper.make_goal_task(make,v,mod,N,2029654802,18.8,3.0)
        env.max_steps=14000
        with torch.no_grad():
            for step in range(1,14001):
                state=np.asarray(obs,dtype=np.float32).copy()
                raw=actor(torch.from_numpy(state).reshape(1,-1)).numpy()[0]
                action=np.array([(raw[0]+1)/2,raw[1]],dtype=np.float32)
                obs,_,term,trunc,info=N.step_checked(env,action)
                states.append(state);actions.append(raw.copy());next_states.append(np.asarray(obs).copy())
                dones.append(float(term or info['reach_goal'] or info['collision']))
                if term or trunc or info['reach_goal'] or info['collision']:break
        result={'experiment':'STAGE_T154_PRESELECTED_DONOR_ACTOR_TEST','donor_sha256':expected,
            'steps':step,'seconds':step*.002,'success':bool(info['reach_goal']),
            'collision':bool(info['collision']),'remaining_m':float(env._goal_state()[1]),
            'training':False,'neural_policy_only':True,'automatic_promotion':False}
        if result['success'] and not result['collision']:
            np.savez_compressed(ROOT/'verified_actor_right.npz',states=np.asarray(states),raw_actions=np.asarray(actions),
                next_states=np.asarray(next_states),dones=np.asarray(dones),episode_offsets=np.array([0,step]))
            result['archive_sha256']=t.sha(ROOT/'verified_actor_right.npz')
        (ROOT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print('T154_RESULT',json.dumps(result),flush=True)
    finally:
        if renderer is not None:renderer.close()
        if env is not None:env.close()

if __name__=='__main__':main()
