"""Native Stage TD3 pilot with physical-time transitions and fixed-start final goals.

One learned Actor produces continuous controls; action holding is the declared
control frequency. No planner, map input, intermediate goal or runtime teacher.
"""
from pathlib import Path
from collections import deque
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import signal
import sys
import time
import shutil
import numpy as np
import torch
import torch.nn.functional as F

REPO=Path(__file__).resolve().parents[3]
ROOT=Path(os.environ['STAGE_RUN_DIR']).resolve()
assert ROOT.is_relative_to(REPO/'stage/runs')
sys.path.insert(0,str(REPO/'stage/experiments/t165_resume'))
from resume_support import rotate_checkpoint
from reward_contract import reward as shaped_native_reward, checks as reward_checks
from value_pretrain import prepare
MASK=[57,58,74,75,86,87,98,99,110,111]

def save(name,obj):
    p=ROOT/name;tmp=p.with_suffix('.partial')
    tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,p)

def guard():
    used=sum(p.stat().st_size for p in (REPO/'stage/runs').rglob('*') if p.is_file())
    assert used<1024**3,('ALL_NEW_RUNS_DISK_CAP',used)
    assert shutil.disk_usage(ROOT).free>40*1024**3

def aggregate(rewards,gamma,divisor):
    return sum(gamma**i*r for i,r in enumerate(rewards))/divisor,gamma**len(rewards)

def check_time_targets():
    r,g=aggregate([1.,2.],.9,25)
    assert abs(r-2.8/25)<1e-12 and abs(g-.81)<1e-12
    assert abs(math.exp(-.002/60)**25-math.exp(-.05/60))<1e-12
    # The same fixed divisor applies to a partial block; time discount uses its actual length.
    r,g=aggregate([3.],.9,25);assert r==3/25 and g==.9

def main():
    lock=(ROOT/'process.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'status.json').exists(),'No duplicate or implicit restart'
    plan=json.loads((ROOT/'plan.json').read_text());check_time_targets();reward_checks();guard()
    spec=importlib.util.spec_from_file_location('smdp_stage_source',REPO/'stage/current/outputs/stage_t148/train.py')
    t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
    for p,h in plan['input_hashes'].items():assert t.sha(p)==h,p
    os.environ['C98_JOB']=os.environ['TD3_JOB']=str(ROOT);os.environ['C22_SCREEN_ONLY']='0'
    torch.set_num_threads(1);torch.manual_seed(plan['seed'])
    rng=np.random.default_rng(plan['seed']);noise_rng=np.random.default_rng(plan['seed']+1)
    core=t.load_module(t.CORE,'smdp_core');agent=t.create_agent(core)
    parent=torch.load(plan['actor_parent'],map_location='cpu',weights_only=False)
    agent.actor.load_state_dict(parent['actor']);agent.actor_target.load_state_dict(parent['actor'])
    teacher=copy.deepcopy(agent.actor).eval().requires_grad_(False)
    agent.actor_optimizer=torch.optim.Adam(agent.actor.parameters(),lr=1e-5)
    agent.critic_optimizer=torch.optim.Adam(agent.critic.parameters(),lr=1e-4)
    for parameter in agent.critic.parameters():assert torch.isfinite(parameter).all()
    with np.load(plan['retention_states'],allow_pickle=False) as z:
        anchors=torch.from_numpy(z['states'].astype(np.float32));offsets=z['episode_offsets'].tolist()
    with torch.no_grad():targets=torch.cat([teacher(x) for x in anchors.split(512)])
    helper=t.load_module(t.HELPER,'smdp_helper');legacy=t.load_module(t.STAGE,'smdp_legacy')
    for p,h in legacy.EXPECTED.items():assert t.sha(p)==h
    runner=legacy.module(legacy.SRC,'smdp_runner');_,v,mod,prep,_=runner.load_native();native=prep[3]
    make=legacy.make_short_task(runner.make_task)
    replay=deque(maxlen=20000)
    demonstrations,calibration=prepare(agent,plan['demonstration_file'],plan['seed'])
    save('value_calibration.json',calibration)
    print('VALUE_CALIBRATED',json.dumps(calibration),flush=True)
    episodes=[];frames=decisions=updates=actor_updates=0
    hold=plan['control_hold_physics_steps'];gamma=math.exp(-.002/plan['discount_efold_seconds'])
    candidates=[x for x in plan['task_bank'] if not x['goal_pose_collision']]
    fixed=[x for x in candidates if x['distance_m']==5 and abs(x['goal_angle_deg']-x['template_angle_deg'])<1e-8]
    assert len(fixed)==2
    stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    started=time.monotonic();deadline=started+plan['wall_seconds']
    last={}
    def status(state):
        return {'state':state,'physics_frames':frames,'target_frames':plan['physics_budget'],
            'policy_decisions':decisions,'critic_updates':updates,'actor_updates':actor_updates,
            'episodes':len(episodes),'training_success':sum(e['success'] for e in episodes),
            'training_collisions':sum(e['collision'] for e in episodes),'elapsed_s':time.monotonic()-started,
            'control_hold_steps':hold,'control_hz':1/(hold*.002),'automatic_promotion':False,'last_losses':last}
    save('status.json',status('INITIALIZED'));path=None
    while frames<plan['physics_budget'] and time.monotonic()<deadline and not stop[0]:
        index=len(episodes)
        task=fixed[index%2]
        env=renderer=None
        try:
            env,renderer,obs,actual=helper.make_goal_task(make,v,mod,native,
                task['template_seed'],task['template_angle_deg'],3.)
            assert int(actual)==task['template_actual_seed']
            assert np.max(np.abs(np.array(env._pose()[:3])-np.array(task['start_pose'])))<1e-7
            env.goal_xy=np.asarray(task['goal_xy'],dtype=float).copy()
            env.prev_goal_dist=float(env._goal_state()[1]);env.history.clear()
            obs=env._observation(update_history=False)
            assert abs(env.prev_goal_dist-task['distance_m'])<1e-5
            env.max_steps=2*int(math.ceil(task['distance_m']/3*14000));ep_steps=0;success=collision=False
            reason='STEP_CAP';noise=np.zeros(2,dtype=np.float32);next_noise_at=frames
            while ep_steps<env.max_steps:
                if frames>=plan['physics_budget'] or time.monotonic()>=deadline:reason='EXPERIMENT_BUDGET';break
                if frames>=next_noise_at:
                    noise=noise_rng.normal(0,.15,2).astype(np.float32)
                    next_noise_at=frames+int(round(1./.002))
                raw=np.clip(agent.act(obs)+noise,-1,1).astype(np.float32)
                action=np.array([(raw[0]+1)/2,raw[1]],dtype=np.float32)
                before=np.asarray(obs,dtype=np.float32).copy();rewards=[];terminal=trunc=False
                for _ in range(min(hold,env.max_steps-ep_steps,plan['physics_budget']-frames)):
                    olddist=float(env._goal_state()[1]);time0=float(env.data.time)
                    obs,native_reward,terminal,trunc,info=native.step_checked(env,action)
                    assert abs(float(env.data.time)-time0-.002)<1e-7
                    rewards.append(shaped_native_reward(olddist,float(env._goal_state()[1]),
                        info['reach_goal'],info['collision'],terminal))
                    frames+=1;ep_steps+=1
                    success=bool(info['reach_goal']);collision=bool(info['collision'])
                    if terminal or trunc:break
                reward,discount=aggregate(rewards,gamma,1);decisions+=1
                replay.append((before,raw.copy(),reward,np.asarray(obs,dtype=np.float32).copy(),
                    float(terminal or success or collision),discount,len(rewards)))
                if len(replay)>=1:
                    batch=[replay[int(i)] for i in rng.choice(len(replay),32,replace=len(replay)<32)]
                    batch.extend(demonstrations[int(i)] for i in rng.choice(len(demonstrations),32,replace=False))
                    s,a,r,ns,d,g=[torch.from_numpy(np.asarray([b[k] for b in batch],dtype=np.float32)) for k in range(6)]
                    with torch.no_grad():
                        smooth=(torch.randn_like(a)*.15).clamp(-.4,.4)
                        na=(agent.actor_target(ns)+smooth).clamp(-1,1)
                        q1t,q2t=agent.critic_target(ns,na)
                        target=r[:,None]+(1-d[:,None])*g[:,None]*torch.minimum(q1t,q2t)
                    q1,q2=agent.critic(s,a);closs=F.mse_loss(q1,target)+F.mse_loss(q2,target)
                    assert torch.isfinite(closs)
                    agent.critic_optimizer.zero_grad(set_to_none=True);closs.backward()
                    torch.nn.utils.clip_grad_norm_(agent.critic.parameters(),10);agent.critic_optimizer.step();updates+=1
                    last={'critic':float(closs.detach())}
                    if decisions>=128 and updates%2==0:
                        for p in agent.critic.parameters():p.requires_grad_(False)
                        q=agent.critic.q1(torch.cat([s,agent.actor(s)],dim=-1))
                        ids=np.concatenate([rng.integers(offsets[i],offsets[i+1],64) for i in range(len(offsets)-1)])
                        retention=(agent.actor(anchors[ids])-targets[ids]).square().mean()
                        loss=-.5*q.mean()/q.detach().abs().mean().clamp(min=1.)+10*retention
                        assert torch.isfinite(loss)
                        agent.actor_optimizer.zero_grad(set_to_none=True);loss.backward()
                        agent.actor.net[0].weight.grad[:,MASK]=0
                        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),.5);agent.actor_optimizer.step();actor_updates+=1
                        for p in agent.critic.parameters():p.requires_grad_(True)
                        with torch.no_grad():
                            for target_net,net in [(agent.actor_target,agent.actor),(agent.critic_target,agent.critic)]:
                                for tp,p in zip(target_net.parameters(),net.parameters()):tp.mul_(.995).add_(p,alpha=.005)
                        last.update(actor=float(loss.detach()),retention=float(retention.detach()))
                if decisions%40==0:
                    guard();live=status('RUNNING');live.update(current_goal=task,current_remaining_m=float(env._goal_state()[1]))
                    save('status.json',live);print('SMDP_PROGRESS',json.dumps(live),flush=True)
                if terminal or trunc:
                    reason='COLLISION' if collision else 'GOAL' if success else 'TRUNCATED' if trunc else 'TERMINATED';break
            row={'task':task,'physics_steps':ep_steps,'success':success,'collision':collision,'end':reason,
                'remaining_m':float(env._goal_state()[1])};episodes.append(row)
            with (ROOT/'episodes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print('SMDP_EPISODE',json.dumps(row),flush=True)
            state={'algorithm':'TD3_SMDP_SUCCESS_VALUE_SEED','actor':agent.actor.state_dict(),
                'actor_target':agent.actor_target.state_dict(),'critic':agent.critic.state_dict(),
                'critic_target':agent.critic_target.state_dict(),'actor_optimizer':agent.actor_optimizer.state_dict(),
                'critic_optimizer':agent.critic_optimizer.state_dict(),'replay':list(replay),'episodes':episodes,
                'physics_frames':frames,'decisions':decisions,'updates':updates,'actor_updates':actor_updates,
                'numpy_rng':rng.bit_generator.state,'noise_rng':noise_rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
                'plan_sha256':t.sha(ROOT/'plan.json'),'episode_boundary':reason!='EXPERIMENT_BUDGET',
                'control_hold_steps':hold,'native_dt':.002,'discount_efold_seconds':60.,'t148_resume_compatible':False}
            guard();path=rotate_checkpoint(ROOT,state);live=status('RUNNING')
            live.update(checkpoint=str(path),checkpoint_sha256=t.sha(path));save('status.json',live)
        finally:
            if renderer is not None:renderer.close()
            if env is not None:env.close()
    final=status('COMPLETED' if frames>=plan['physics_budget'] else 'STOPPED_USER_SIGNAL' if stop[0] else 'PAUSED_WALL_BUDGET')
    final.update(checkpoint=str(path),checkpoint_sha256=t.sha(path) if path else None);save('status.json',final)
    print('SMDP_FINISHED',json.dumps(final),flush=True)

if __name__=='__main__':main()
