"""Shared GPU TD3 learner with eight independent native Stage simulators."""
from pathlib import Path
from collections import deque
import copy
import fcntl
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import shutil
import signal
import time
import numpy as np
import torch
import torch.nn.functional as F
from worker import run

REPO=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=REPO/'stage/runs/t170_vector_td3'
MASK=[57,58,74,75,86,87,98,99,110,111]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):
    p=OUT/n;tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,p)
def used():return sum(p.stat().st_size for p in (REPO/'stage/runs').rglob('*') if p.is_file())
def recv(pipe):
    assert pipe.poll(60),'Native worker response timeout'
    row=pipe.recv();assert 'error' not in row,row
    return row
def main():
    lock=(OUT/'process.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'status.json').exists(),'No duplicate/restart'
    plan=json.loads((HERE/'protocol.json').read_text())
    for p,h in plan['input_hashes'].items():assert sha(p)==h,p
    assert json.loads((OUT/'preflight.json').read_text())['state']=='PASS'
    assert used()+128*1024**2<1024**3 and shutil.disk_usage(OUT).free>40*1024**3
    torch.set_num_threads(1);torch.manual_seed(plan['seed']);torch.cuda.manual_seed_all(plan['seed'])
    torch.cuda.set_per_process_memory_fraction(.35);torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(plan['seed']);noise_rng=[np.random.default_rng(plan['seed']+100+i) for i in range(8)]
    spec=importlib.util.spec_from_file_location('vector_td3_core',REPO/'stage/current/outputs/stage_t148/td3_core_goal_contrast.py')
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    agent=core.TD3(112,2,device='cuda',tau=.005,policy_noise=.15,noise_clip=.4,policy_delay=2)
    parent_actor=torch.load(plan['actor_checkpoint'],map_location='cpu',weights_only=False)
    parent_critic=torch.load(plan['critic_checkpoint'],map_location='cpu',weights_only=False)
    agent.actor.load_state_dict(parent_actor['actor']);agent.actor_target.load_state_dict(parent_actor['actor'])
    teacher=copy.deepcopy(agent.actor).eval().requires_grad_(False)
    assert parent_critic['control_hold_steps']==25 and parent_critic['discount_efold_seconds']==60.
    agent.critic.load_state_dict(parent_critic['critic']);agent.critic_target.load_state_dict(parent_critic['critic_target'])
    agent.critic_optimizer.load_state_dict(parent_critic['critic_optimizer'])
    for group in agent.critic_optimizer.param_groups:group['lr']=1e-4
    agent.actor_optimizer=torch.optim.Adam(agent.actor.parameters(),lr=1e-5)
    replay=deque(parent_critic['replay'],maxlen=20000)
    with np.load(plan['demonstrations'],allow_pickle=False) as z:
        names=['states','actions','rewards','next_states','dones','discounts','native_counts']
        arrays=[z[k].astype(np.float32).copy() for k in names];offsets=z['episode_offsets'].tolist()
    demos=[tuple(a[i].copy() if isinstance(a[i],np.ndarray) else float(a[i]) for a in arrays) for i in range(len(arrays[0]))]
    anchors=torch.from_numpy(arrays[0]).cuda()
    with torch.no_grad():anchor_targets=teacher(anchors)
    del parent_actor,parent_critic
    ctx=mp.get_context('spawn');pipes=[];processes=[];observations=[];tasks=[]
    noise=np.zeros((8,2),dtype=np.float32);noise_age=np.full(8,500)
    frames=updates=actor_updates=iterations=0;episodes=[];signatures=set();checked_gates=set();last={};path=None
    stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    start=time.monotonic();deadline=start+plan['wall_seconds'];next_checkpoint=250000
    def status(state):
        return {'state':state,'physics_frames':frames,'target_frames':1000000,'critic_updates':updates,'actor_updates':actor_updates,
            'vector_iterations':iterations,'n_envs':8,'training_episodes':len(episodes),'training_success':sum(r['success'] for r in episodes),
            'training_collisions':sum(r['collision'] for r in episodes),'unique_initial_tasks':len(signatures),'elapsed_s':time.monotonic()-start,
            'physics_frames_per_second':frames/max(.001,time.monotonic()-start),'gpu_peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2,
            'new_runs_total_bytes':used(),'automatic_promotion':False,'last_losses':last}
    def checkpoint():
        nonlocal path,last_checkpoint_frames
        assert used()+128*1024**2<1024**3,'Checkpoint headroom required within global1GiB cap'
        state={'algorithm':'VECTOR_GPU_TD3_VALUE_SEED','actor':agent.actor.state_dict(),'actor_target':agent.actor_target.state_dict(),
            'critic':agent.critic.state_dict(),'critic_target':agent.critic_target.state_dict(),'actor_optimizer':agent.actor_optimizer.state_dict(),
            'critic_optimizer':agent.critic_optimizer.state_dict(),'replay':list(replay),'physics_frames':frames,'updates':updates,
            'actor_updates':actor_updates,'episodes':episodes,'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all(),'noise_rng':[r.bit_generator.state for r in noise_rng],
            'protocol_sha256':sha(HERE/'protocol.json'),'exact_live_environment_resume':False,
            'resume_limit':'Eight live environment states are not serialized; future initialization is an explicit warm start, not exact resume.'}
        partial=OUT/'checkpoint_partial.pt';torch.save(state,partial);assert partial.stat().st_size<128*1024**2
        latest=OUT/'checkpoint_latest.pt';previous=OUT/'checkpoint_previous.pt'
        if previous.exists():previous.unlink()
        if latest.exists():os.replace(latest,previous)
        os.replace(partial,latest);path=latest;last_checkpoint_frames=frames
        live=status('RUNNING');live.update(checkpoint=str(latest),checkpoint_sha256=sha(latest));save('status.json',live)
    final_state='PAUSED_WALL_BUDGET'
    last_checkpoint_frames=-1
    try:
        for index in range(8):
            a,b=ctx.Pipe();p=ctx.Process(target=run,args=(b,index,str(OUT),plan['seed']));p.start();b.close();pipes.append(a);processes.append(p)
        save('workers.json',{'pids':[p.pid for p in processes],'parent_pid':os.getpid(),'n_envs':8})
        for pipe in pipes:pipe.send(('reset',0))
        for pipe in pipes:
            row=recv(pipe);observations.append(row['obs']);tasks.append(row['task']);signatures.add(row['task']['signature'])
        save('status.json',status('RUNNING'))
        while frames<1000000 and time.monotonic()<deadline and not stop[0]:
            if used()+128*1024**2>=1024**3 or shutil.disk_usage(OUT).free<=40*1024**3:
                final_state='STOPPED_RESOURCE_LIMIT';break
            x=torch.from_numpy(np.asarray(observations,dtype=np.float32)).cuda()
            with torch.no_grad():actions=agent.actor(x).cpu().numpy()
            for i in range(8):
                if noise_age[i]>=500:noise[i]=noise_rng[i].normal(0,.15,2);noise_age[i]=0
            actions=np.clip(actions+noise,-1,1).astype(np.float32)
            remaining=1000000-frames;active=[]
            for i,pipe in enumerate(pipes):
                quota=min(25,max(0,remaining-i*25))
                if quota:pipe.send(('step',actions[i],quota));active.append(i)
            for i in active:
                row=recv(pipes[i]);replay.append((observations[i].copy(),actions[i].copy(),row['reward'],row['obs'].copy(),row['done'],row['discount'],row['n']))
                observations[i]=row['obs'];frames+=row['n'];noise_age[i]+=row['n']
                if row['episode'] is not None:
                    episode={**row['episode'],'worker':i,'global_physics_frames':frames};episodes.append(episode)
                    with (OUT/'episodes.jsonl').open('a') as f:f.write(json.dumps(episode)+'\n')
                    phase=0 if frames<250000 else 1 if frames<500000 else 2
                    pipes[i].send(('reset',phase));row=recv(pipes[i]);observations[i]=row['obs'];tasks[i]=row['task'];signatures.add(row['task']['signature']);noise_age[i]=500
            batch=[replay[int(i)] for i in rng.choice(len(replay),192,replace=False)]
            batch.extend(demos[int(i)] for i in rng.choice(len(demos),64,replace=False))
            s,a,r,ns,d,g=[torch.from_numpy(np.asarray([b[k] for b in batch],dtype=np.float32)).cuda() for k in range(6)]
            with torch.no_grad():
                smooth=(torch.randn_like(a)*.15).clamp(-.4,.4);na=(agent.actor_target(ns)+smooth).clamp(-1,1)
                qt1,qt2=agent.critic_target(ns,na);target=r[:,None]+(1-d[:,None])*g[:,None]*torch.minimum(qt1,qt2)
            q1,q2=agent.critic(s,a);loss=F.mse_loss(q1,target)+F.mse_loss(q2,target);assert torch.isfinite(loss)
            agent.critic_optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(agent.critic.parameters(),10);agent.critic_optimizer.step();updates+=1
            last={'critic':float(loss.detach().cpu())}
            if updates%2==0:
                if frames>=250000:
                    for p in agent.critic.parameters():p.requires_grad_(False)
                    q=agent.critic.q1(torch.cat([s,agent.actor(s)],dim=-1))
                    ids=np.concatenate([rng.integers(offsets[j],offsets[j+1],32) for j in range(6)])
                    keep=(agent.actor(anchors[ids])-anchor_targets[ids]).square().mean()
                    aloss=-.5*q.mean()/q.detach().abs().mean().clamp(min=1.)+10*keep;assert torch.isfinite(aloss)
                    agent.actor_optimizer.zero_grad(set_to_none=True);aloss.backward();agent.actor.net[0].weight.grad[:,MASK]=0
                    torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),.5);agent.actor_optimizer.step();actor_updates+=1
                    for p in agent.critic.parameters():p.requires_grad_(True)
                    last.update(actor=float(aloss.detach().cpu()),retention=float(keep.detach().cpu()))
                with torch.no_grad():
                    for tn,n in [(agent.actor_target,agent.actor),(agent.critic_target,agent.critic)]:
                        for tp,p in zip(tn.parameters(),n.parameters()):tp.mul_(.995).add_(p,alpha=.005)
            iterations+=1
            if iterations%25==0:
                save('status.json',status('RUNNING'));print('VECTOR_PROGRESS',json.dumps(status('RUNNING')),flush=True)
            failed=False
            for gate,required in [(250000,8),(500000,20)]:
                if frames>=gate and gate not in checked_gates:
                    checked_gates.add(gate);success=sum(r['success'] for r in episodes)
                    passed=success>=required and len(signatures)>=16
                    with (OUT/'curriculum_gates.jsonl').open('a') as f:f.write(json.dumps({'gate_frames':gate,'observed_frames':frames,'success':success,'required':required,'unique_tasks':len(signatures),'passed':passed})+'\n')
                    if not passed:final_state='STOPPED_CURRICULUM_GATE';failed=True;break
            if frames>=next_checkpoint:checkpoint();next_checkpoint+=250000
            if failed:break
        if frames>=1000000:final_state='COMPLETED'
        elif stop[0]:final_state='STOPPED_USER_SIGNAL'
        if last_checkpoint_frames!=frames:checkpoint()
        final=status(final_state);final.update(checkpoint=str(path),checkpoint_sha256=sha(path));save('status.json',final)
        print('VECTOR_FINISHED',json.dumps(final),flush=True)
    finally:
        for pipe in pipes:
            try:pipe.send(('close',))
            except (OSError,EOFError):pass
        for p in processes:
            p.join(5)
            if p.is_alive():p.terminate();p.join(5)
        for pipe in pipes:pipe.close()
if __name__=='__main__':main()
