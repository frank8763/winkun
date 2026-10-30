"""Recover and physically verify one complete successful Stage training episode.

Replay is only a demonstration integrity check, never the final navigation policy.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
PARENT=Path('/home/linghu/mkmini_stage_td3_t146_rightturn_continuation_20261029_v1')
DONOR=PARENT/'checkpoints/episode_0724.pt'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert not (ROOT/'verified_right_success.npz').exists()
    checkpoints=[]
    for line in (PARENT/'train.log').read_text().splitlines():
        if line.startswith('TD3_T6_CHECKPOINT='):
            try: row=json.loads(line.split('=',1)[1])
            except json.JSONDecodeError: continue
            if row['checkpoint']==DONOR.name:checkpoints.append(row)
    assert len(checkpoints)==1
    donor_sha=sha(DONOR)
    assert donor_sha==checkpoints[0]['checkpoint_sha256']
    ck=torch.load(DONOR,map_location='cpu',weights_only=False)
    assert ck['episode_index']==724 and len(ck['episodes'])==724
    episode=ck['episodes'][-1]
    assert episode['success'] and not episode['collision'] and episode['end']=='GOAL'
    assert episode['seed']==2029654802 and episode['angle']==18.8
    n=int(episode['physics_frames'])
    replay=ck['replay'][-n:]
    assert n==11845 and len(replay)==n and all(r[5]=='RIGHT' for r in replay)
    states=np.asarray([r[0] for r in replay],dtype=np.float32)
    actions=np.asarray([r[1] for r in replay],dtype=np.float32)
    next_states=np.asarray([r[3] for r in replay],dtype=np.float32)
    dones=np.asarray([r[4] for r in replay],dtype=np.float32)
    assert np.flatnonzero(dones).tolist()==[n-1]
    assert np.max(np.abs(states[1:]-next_states[:-1]))<1e-6
    os.environ['C98_JOB']=os.environ['TD3_JOB']=str(ROOT)
    os.environ['C22_SCREEN_ONLY']='0'
    torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('t152_env',ROOT.parent/'stage_t148/train.py')
    t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
    helper=t.load_module(t.HELPER,'t152_helper')
    legacy=t.load_module(t.STAGE,'t152_legacy')
    for path,h in legacy.EXPECTED.items():assert sha(path)==h
    runner=legacy.module(legacy.SRC,'t152_runner')
    _,v,mod,prep,_=runner.load_native()
    native=prep[3];make=legacy.make_short_task(runner.make_task)
    env=renderer=None
    try:
        env,renderer,obs,seed=helper.make_goal_task(make,v,mod,native,2029654802,18.8,3.0)
        env.max_steps=14000
        assert seed==2029654802
        max_error=0.0
        for i,raw in enumerate(actions):
            error=float(np.max(np.abs(np.asarray(obs)-states[i])))
            max_error=max(error,max_error)
            assert error<2e-4, ('DEMONSTRATION_STATE_MISMATCH',i,error)
            action=np.asarray([(raw[0]+1)/2,raw[1]],dtype=np.float32)
            obs,_,term,trunc,info=native.step_checked(env,action)
            assert not info['collision']
            assert (not term and not trunc) or i==n-1
        assert info['reach_goal'] and not info['collision'] and term and not trunc
        np.savez_compressed(ROOT/'verified_right_success.npz',states=states,raw_actions=actions,
            next_states=next_states,dones=dones,episode_offsets=np.array([0,n],dtype=np.int64))
        report={'experiment':'STAGE_T152_HARD_RIGHT_DEMONSTRATION_VERIFICATION',
            'donor_checkpoint':str(DONOR),'donor_checkpoint_sha256':donor_sha,
            'selection_rule':'most recent full successful hard-right episode in completed T146 history',
            'original_episode':episode,'verified_frames':n,'max_observation_error':max_error,
            'simulation_seconds':float(env.data.time),'remaining_m':float(env._goal_state()[1]),
            'success':True,'collision':False,'training':False,'final_policy_test':False,
            'archive_sha256':sha(ROOT/'verified_right_success.npz'),
            'contains_exploration_actions':True,'automatic_promotion':False}
        (ROOT/'demonstration_verification.json').write_text(json.dumps(report,indent=2)+'\n')
        print('T152_DEMONSTRATION_VERIFIED',json.dumps(report),flush=True)
    finally:
        if renderer is not None:renderer.close()
        if env is not None:env.close()

if __name__=='__main__':main()
