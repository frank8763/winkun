"""Fresh successful T159 transitions at the validated 20Hz control cadence."""
from pathlib import Path
import importlib.util
import json
import os
import hashlib
import numpy as np
import torch
from reward_contract import reward,GAMMA,checks

REPO=Path(__file__).resolve().parents[3];OUT=REPO/'stage/runs/t169_value_seed'

def main():
    assert not (OUT/'demonstrations.npz').exists()
    checks();torch.set_num_threads(1)
    os.environ['C98_JOB']=os.environ['TD3_JOB']=str(OUT);os.environ['C22_SCREEN_ONLY']='0'
    path=REPO/'stage/current/outputs/stage_t148/train.py'
    spec=importlib.util.spec_from_file_location('value_seed_source',path);t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
    helper=t.load_module(t.HELPER,'value_seed_helper');legacy=t.load_module(t.STAGE,'value_seed_legacy')
    for p,h in legacy.EXPECTED.items():assert t.sha(p)==h
    runner=legacy.module(legacy.SRC,'value_seed_runner');_,v,mod,prep,_=runner.load_native();native=prep[3]
    agent=t.create_agent(t.load_module(t.CORE,'value_seed_core'))
    parent=REPO/'stage/current/outputs/stage_t159/consolidated_actor.pt'
    agent.actor.load_state_dict(torch.load(parent,map_location='cpu',weights_only=False)['actor']);agent.actor.eval().requires_grad_(False)
    before={k:x.clone() for k,x in agent.actor.state_dict().items()}
    cases=json.loads((REPO/'stage/current/outputs/stage_t164/frozen_eval/declaration.json').read_text())['cases'][:6]
    rows=[];offsets=[0];reports=[];returns=[]
    for case in cases:
        env=renderer=None;episode=[]
        try:
            env,renderer,obs,actual=helper.make_goal_task(legacy.make_short_task(runner.make_task),v,mod,native,case['seed'],case['angle_deg'],case['distance_m'])
            env.max_steps=case['step_cap'];step=0;terminal=trunc=False
            while step<case['step_cap']:
                old=np.asarray(obs,dtype=np.float32).copy();raw=agent.act(obs)
                action=np.array([(raw[0]+1)/2,raw[1]],dtype=np.float32);total=0.;count=0
                for _ in range(min(25,case['step_cap']-step)):
                    distance=float(env._goal_state()[1]);time0=float(env.data.time)
                    obs,_,terminal,trunc,info=native.step_checked(env,action);step+=1
                    assert abs(float(env.data.time)-time0-.002)<1e-7
                    total+=GAMMA**count*reward(distance,float(env._goal_state()[1]),info['reach_goal'],info['collision'],terminal);count+=1
                    if terminal or trunc:break
                episode.append((old,raw.copy(),total,np.asarray(obs,dtype=np.float32).copy(),float(terminal),GAMMA**count,count))
                if terminal or trunc:break
            assert info['reach_goal'] and not info['collision'],case['id']
            mc=[];value=0.
            for row in reversed(episode):value=row[2]+(1-row[4])*row[5]*value;mc.append(value)
            rows.extend(episode);returns.extend(reversed(mc));offsets.append(len(rows))
            reports.append({'id':case['id'],'steps':step,'success':True,'collision':False,'actual_seed':int(actual),'decisions':len(episode)})
            print('VALUE_DEMO',json.dumps(reports[-1]),flush=True)
        finally:
            if renderer is not None:renderer.close()
            if env is not None:env.close()
    assert all(torch.equal(before[k],x) for k,x in agent.actor.state_dict().items())
    arrays={name:np.asarray([r[i] for r in rows],dtype=np.float32) for i,name in enumerate(['states','actions','rewards','next_states','dones','discounts','native_counts'])}
    arrays.update(mc_returns=np.asarray(returns,dtype=np.float32),episode_offsets=np.asarray(offsets,dtype=np.int64))
    np.savez_compressed(OUT/'demonstrations.npz',**arrays)
    (OUT/'demonstration_verification.json').write_text(json.dumps({'state':'PASS','episodes':reports,'safe_success':6,'n':6,'training_updates':0,'actor_unchanged':True,'teacher':'Single frozen T159, training data only','sha256':t.sha(OUT/'demonstrations.npz'),'decisions':len(rows)},indent=2)+'\n')

if __name__=='__main__':main()
