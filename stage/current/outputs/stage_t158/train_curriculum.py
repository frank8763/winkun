"""Bounded online TD3 pilot with frozen-policy retention on a 3m/4m curriculum.

Engineering pilot, not a controlled reproduction of offline TD3+BC.
Actor and target are initialized from T157; Critic and optimizer from T148_0748.
"""
from pathlib import Path
from collections import deque
import copy
import fcntl
import importlib.util
import json
import math
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent
MASK=[57,58,74,75,86,87,98,99,110,111]

def save(name,obj):
    path=ROOT/name;tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,path)

def main():
    lock=(ROOT/'process.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'status.json').exists(), 'Do not duplicate or implicitly restart this pilot'
    plan=json.loads((ROOT/'plan.json').read_text())
    spec=importlib.util.spec_from_file_location('t158_environment_source',ROOT.parent/'stage_t148/train.py')
    t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
    for path,h in plan['input_hashes'].items():assert t.sha(path)==h, path
    os.environ['C98_JOB']=os.environ['TD3_JOB']=str(ROOT)
    os.environ['C22_SCREEN_ONLY']='0'
    torch.set_num_threads(1);torch.manual_seed(plan['seed']);rng=np.random.default_rng(plan['seed'])
    core=t.load_module(t.CORE,'t158_core')
    agent=t.create_agent(core)
    initial=torch.load(plan['initial_actor'],map_location='cpu',weights_only=False)
    critic_parent=torch.load(plan['critic_parent'],map_location='cpu',weights_only=False)
    agent.actor.load_state_dict(initial['actor'],strict=True)
    agent.actor_target.load_state_dict(initial['actor'],strict=True)
    agent.critic.load_state_dict(critic_parent['critic'],strict=True)
    agent.critic_target.load_state_dict(critic_parent['critic_target'],strict=True)
    agent.critic_optimizer.load_state_dict(critic_parent['critic_optimizer'])
    agent.actor_optimizer=torch.optim.Adam(agent.actor.parameters(),lr=plan['actor_lr'])
    for group in agent.critic_optimizer.param_groups:group['lr']=plan['critic_lr']
    agent.total_updates=int(critic_parent['updates'])
    teacher=copy.deepcopy(agent.actor).eval().requires_grad_(False)
    with np.load(plan['retention_observations'],allow_pickle=False) as z:
        anchor_obs=torch.from_numpy(np.asarray(z['states'],dtype=np.float32).copy())
    with torch.no_grad():
        anchor_actions=torch.cat([teacher(x) for x in anchor_obs.split(512)])
    replay=deque(critic_parent['replay'],maxlen=plan['replay_capacity'])
    helper=t.load_module(t.HELPER,'t158_helper');legacy=t.load_module(t.STAGE,'t158_legacy')
    for path,h in legacy.EXPECTED.items():assert t.sha(path)==h
    runner=legacy.module(legacy.SRC,'t158_runner')
    _,v,mod,prep,_=runner.load_native();N=prep[3]
    make=legacy.make_short_task(runner.make_task)
    new_frames=0;actor_updates=0;critic_updates=0;episodes=[]
    deadline=time.monotonic()+plan['wall_seconds']
    start=time.monotonic()
    loss_values={}

    def status(state):
        return {'state':state,'new_physics_frames':new_frames,'target_new_frames':plan['new_frames'],
            'actor_updates':actor_updates,'critic_updates':critic_updates,'episodes':len(episodes),
            'training_success':sum(x['success'] for x in episodes),
            'training_collisions':sum(x['collision'] for x in episodes),
            'elapsed_seconds':time.monotonic()-start,'last_losses':loss_values,
            'automatic_promotion':False,'frozen_performance_confirmed':False}

    save('status.json',status('INITIALIZED'))
    print('T158_INITIALIZED',json.dumps(status('INITIALIZED')),flush=True)
    while new_frames<plan['new_frames'] and time.monotonic()<deadline:
        task=plan['tasks'][len(episodes)%len(plan['tasks'])]
        env=renderer=None
        try:
            env,renderer,obs,actual_seed=helper.make_goal_task(make,v,mod,N,
                task['seed'],task['angle_deg'],task['distance_m'])
            env.max_steps=task['step_cap']
            assert abs(float(env._goal_state()[1])-task['distance_m'])<1e-4
            success=collision=False;reason='STEP_CAP';episode_frames=0;noise=np.zeros(2,dtype=np.float32)
            for step in range(1,task['step_cap']+1):
                if new_frames>=plan['new_frames'] or time.monotonic()>=deadline:
                    reason='EXPERIMENT_BUDGET';break
                if (step-1)%25==0:noise=rng.normal(0,plan['exploration_std'],2).astype(np.float32)
                raw=np.clip(agent.act(obs)+noise,-1,1).astype(np.float32)
                action=np.array([(raw[0]+1)/2,raw[1]],dtype=np.float32)
                previous_distance=float(env._goal_state()[1]);time0=float(env.data.time)
                nxt,native_reward,terminated,truncated,info=N.step_checked(env,action)
                assert abs(float(env.data.time)-time0-.002)<1e-7
                penalty=t._t59_external_clearance_reward(env)
                progress=np.clip(previous_distance-float(env._goal_state()[1]),-.02,.02)
                reward=float(native_reward)-penalty*(.8+2.4*max(0,float(action[0]))**2)+16*float(progress)
                success=bool(info['reach_goal']);collision=bool(info['collision'])
                done=float(terminated or success or collision)
                replay.append((np.asarray(obs,dtype=np.float32).copy(),raw.copy(),reward,
                    np.asarray(nxt,dtype=np.float32).copy(),done,'LEFT' if task['angle_deg']<0 else 'RIGHT'))
                obs=nxt;new_frames+=1;episode_frames+=1
                if new_frames%plan['update_every_frames']==0:
                    ids=rng.choice(len(replay),64,replace=False)
                    samples=[replay[int(i)] for i in ids]
                    s,a,r,ns,d=[torch.from_numpy(np.asarray([x[k] for x in samples],dtype=np.float32)) for k in range(5)]
                    with torch.no_grad():
                        smooth=(torch.randn_like(a)*.15).clamp(-.4,.4)
                        na=(agent.actor_target(ns)+smooth).clamp(-1,1)
                        q1t,q2t=agent.critic_target(ns,na)
                        target=r[:,None]+(1-d[:,None])*agent.gamma*torch.minimum(q1t,q2t)
                    q1,q2=agent.critic(s,a)
                    closs=F.mse_loss(q1,target)+F.mse_loss(q2,target)
                    assert torch.isfinite(closs)
                    agent.critic_optimizer.zero_grad(set_to_none=True);closs.backward()
                    torch.nn.utils.clip_grad_norm_(agent.critic.parameters(),10)
                    agent.critic_optimizer.step();agent.total_updates+=1;critic_updates+=1
                    loss_values={'critic':float(closs.detach())}
                    if new_frames>=plan['critic_adaptation_frames'] and critic_updates%4==0:
                        for parameter in agent.critic.parameters():parameter.requires_grad_(False)
                        q=agent.critic.q1(torch.cat([s,agent.actor(s)],dim=-1))
                        scale=plan['rl_alpha']/q.detach().abs().mean().clamp(min=1.)
                        anchor_ids=np.concatenate([rng.integers(0,10612,128),
                            rng.integers(10612,20319,64),rng.integers(20319,len(anchor_obs),64)])
                        retention=(agent.actor(anchor_obs[anchor_ids])-anchor_actions[anchor_ids]).square().mean()
                        aloss=-scale*q.mean()+retention
                        assert torch.isfinite(aloss)
                        agent.actor_optimizer.zero_grad(set_to_none=True);aloss.backward()
                        agent.actor.net[0].weight.grad[:,MASK]=0
                        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),.5)
                        agent.actor_optimizer.step();actor_updates+=1
                        for parameter in agent.critic.parameters():parameter.requires_grad_(True)
                        with torch.no_grad():
                            for target_net,net in [(agent.actor_target,agent.actor),(agent.critic_target,agent.critic)]:
                                for tp,p in zip(target_net.parameters(),net.parameters()):tp.mul_(1-agent.tau).add_(p,alpha=agent.tau)
                        assert torch.count_nonzero(agent.actor.net[0].weight[:,MASK])==0
                        loss_values.update({'actor':float(aloss.detach()),'retention':float(retention.detach()),'q_scale':float(scale)})
                if new_frames%1000==0:
                    live=status('RUNNING');live['current_task']=task;live['current_remaining_m']=float(env._goal_state()[1])
                    save('status.json',live);print('T158_PROGRESS',json.dumps(live),flush=True)
                if success or collision or terminated or truncated:
                    reason='COLLISION' if collision else 'GOAL' if success else 'TRUNCATED' if truncated else 'TERMINATED';break
            row={**task,'actual_seed':int(actual_seed),'physics_frames':episode_frames,
                'success':success,'collision':collision,'end':reason,'remaining_m':float(env._goal_state()[1])}
            episodes.append(row)
            with (ROOT/'episodes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print('T158_EPISODE',json.dumps(row),flush=True)
            state={'algorithm':'ONLINE_TD3_FROZEN_POLICY_RETENTION','actor':agent.actor.state_dict(),
                'actor_target':agent.actor_target.state_dict(),'critic':agent.critic.state_dict(),
                'critic_target':agent.critic_target.state_dict(),'actor_optimizer':agent.actor_optimizer.state_dict(),
                'critic_optimizer':agent.critic_optimizer.state_dict(),'total_updates':agent.total_updates,
                'new_physics_frames':new_frames,'episodes':episodes,'replay':list(replay),
                'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
                'initialization':plan['initialization'],'plan_sha256':t.sha(ROOT/'plan.json'),
                't148_resume_compatible':False,'automatic_promotion':False}
            path=ROOT/f'episode_{len(episodes):03d}.pt';tmp=path.with_suffix('.pt.partial')
            torch.save(state,tmp);os.replace(tmp,path)
            report=status('RUNNING');report.update({'checkpoint':str(path),'checkpoint_sha256':t.sha(path)})
            save('status.json',report)
        finally:
            if renderer is not None:renderer.close()
            if env is not None:env.close()
    final=status('COMPLETED' if new_frames>=plan['new_frames'] else 'PAUSED_WALL_BUDGET')
    final.update({'checkpoint':str(path),'checkpoint_sha256':t.sha(path)})
    save('status.json',final);print('T158_TRAINING_FINISHED',json.dumps(final),flush=True)

if __name__=='__main__':main()
