"""PPO clipped surrogate with complete MC/LOO advantages, no learned Critic."""
from pathlib import Path
import os
CPU_POOL=[int(x) for x in os.environ.get('STAGE_CPU_POOL',','.join(map(str,sorted(os.sched_getaffinity(0))))).split(',')]
os.environ['STAGE_CPU_POOL']=','.join(map(str,CPU_POOL))
os.sched_setaffinity(0,{int(os.environ.get('STAGE_PROCESS_CPU',str(CPU_POOL[0])))})
import fcntl
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import shutil
import time
import numpy as np
import torch
from common import *
from worker import run

def used():return sum(p.stat().st_size for p in (REPO/'stage/runs').rglob('*') if p.is_file())
def main():
    lock=(OUT/'training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'status.json').exists(),'No duplicate or implicit resume'
    plan=json.loads((HERE/'protocol.json').read_text());prep={'selected_hold':100,'snapshots':[]};assert json.loads(Path(plan['preflight_summary']).read_text())['training_gate']=='FAIL_ROBUSTNESS' # local repair only; no expanded-distance curriculum
    for p,h in plan['input_hashes'].items():assert sha(p)==h,p
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.35);torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    initial=torch.load(plan['initial_policy_checkpoint'],map_location='cpu',weights_only=False)['policy']
    with np.load(plan['retention_demonstrations'],allow_pickle=False) as z:
        a_states=z['states'].copy();a_actions=z['actions'].copy();offsets=z['episode_offsets'].copy();counts=z['native_counts'].copy()
    a_actions[:,0]=.5*(a_actions[:,0]+1)
    clocks=[]
    for j in range(6):
        lo,hi=offsets[j:j+2];elapsed=np.r_[0,np.cumsum(counts[lo:hi-1])];cap=plan['primary_cases'][j]['step_cap'];clocks.extend(1-elapsed/cap)
    anchors=torch.tensor(np.column_stack([a_states,np.asarray(clocks,dtype=np.float32)]),device='cuda');targets=torch.tensor(a_actions,device='cuda')
    ctx=mp.get_context('spawn');pipes=[];workers=[];start=time.monotonic();deadline=start+plan['wall_seconds'];all_results={};training_rows=[];max_logp_error=0.
    def payload(policy):return {k:v.detach().cpu().numpy().copy() for k,v in policy.state_dict().items()}
    def receive(pipe):
        assert pipe.poll(max(.1,min(180,deadline-time.monotonic()))),'Native episode timeout'
        row=pipe.recv();assert 'error' not in row,row;return row
    def conditional(policy,s,prev,latent,mask):
        rho=math.exp(-prep['selected_hold']*.002/current_tau)
        mean=policy(s)+rho*(latent-policy(prev))*mask[:,None]
        std=policy.logstd.clamp(-1.6,0.).exp()[None,:]*torch.where(mask[:,None]>0,math.sqrt(1-rho*rho),1.)
        return torch.distributions.Normal(mean,std)
    try:
        for i in range(8):
            a,b=ctx.Pipe();p=ctx.Process(target=run,args=(b,i,deadline));os.environ['STAGE_PROCESS_CPU']=str(CPU_POOL[(i+1)%len(CPU_POOL)]);p.start();os.environ['STAGE_PROCESS_CPU']=str(CPU_POOL[0]);b.close();pipes.append(a);workers.append(p)
        save(OUT/'workers.json',{'pids':[p.pid for p in workers],'parent':os.getpid()})
        for arm in ['fixed_source','randomized_reset']:
            current_tau=plan['arms'][arm]['noise_correlation_s']
            torch.manual_seed(plan['seed']);torch.cuda.manual_seed_all(plan['seed']);rng=np.random.default_rng(plan['seed'])
            policy=Policy().cuda();policy.load_state_dict(initial)
            with torch.no_grad():policy.logstd.copy_(torch.log(torch.tensor(plan['initial_std'],device='cuda')))
            optimizer=torch.optim.Adam([{'params':policy.body.parameters(),'lr':1e-5},{'params':policy.native_head.parameters(),'lr':1e-4},{'params':[policy.logstd],'lr':1e-3}])
            folder=OUT/arm;folder.mkdir();frames=updates=0;episodes=[];last={}
            for iteration in range(plan['batches_per_arm']):
                assert used()+64*1024**2<1024**3 and shutil.disk_usage(OUT).free>40*1024**3
                weights=payload(policy)
                course_bin='local_robustness'
                selected=plan['training_pairs'][iteration]
                for i,pipe in enumerate(pipes):
                    source=selected[i%4].copy()
                    if arm=='fixed_source':
                        source.update(delta_x_m=0.,delta_y_m=0.,delta_yaw_deg=0.)
                    job={**source,'id':f'{arm}_{iteration}_{i}','source_id':source['id'],'arm':arm,'reset_group':'train',
                         'noise_correlation_s':current_tau,'seed':plan['seed']+iteration*100+i,'stochastic':True,'collect':True}
                    pipe.send((job,weights))
                batch=[receive(pipe) for pipe in pipes]
                for i,row in enumerate(batch):
                    trajectory=row.pop('trajectory');assert row['source']==batch[(i+4)%8]['source'];assert row['initial_reset_signature']==batch[(i+4)%8]['initial_reset_signature']
                    # The partner arrays are kept separately below; no learned or bootstrapped value is used.
                    row['_trajectory']=trajectory
                arrays=[]
                for i,row in enumerate(batch):
                    trajectory=row['_trajectory'];partner=batch[(i+4)%8]['_trajectory']['returns'];baseline=np.zeros_like(trajectory['returns'])
                    n=min(len(baseline),len(partner));baseline[:n]=partner[:n]
                    trajectory['advantages']=(trajectory['returns']-baseline)/10.;arrays.append(trajectory)
                joined={k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]}
                np.savez_compressed(folder/f'batch_{iteration}.npz',**joined)
                tensors={k:torch.tensor(v,device='cuda') for k,v in joined.items()}
                with torch.no_grad():
                    old=conditional(policy,tensors['states'],tensors['previous_states'],tensors['previous_latents'],tensors['previous_mask']).log_prob(tensors['latents']).sum(-1)
                    error=float((old-tensors['old_logp']).abs().max());max_logp_error=max(max_logp_error,error);assert error<2e-4,('Behavior likelihood mismatch',error)
                for epoch in range(4):
                    permutation=rng.permutation(len(joined['states']));early=False
                    for begin in range(0,len(permutation),128):
                        ids=permutation[begin:begin+128];x={k:v[ids] for k,v in tensors.items()}
                        dist=conditional(policy,x['states'],x['previous_states'],x['previous_latents'],x['previous_mask'])
                        logratio=dist.log_prob(x['latents']).sum(-1)-x['old_logp'];ratio=logratio.exp();adv=x['advantages']
                        surrogate=-torch.minimum(ratio*adv,ratio.clamp(.9,1.1)*adv)
                        ids_keep=np.concatenate([rng.integers(offsets[j],offsets[j+1],32) for j in range(6)])
                        keep=(policy(anchors[ids_keep])-targets[ids_keep]).square().mean()
                        loss=(surrogate*x['policy_weights']).mean()+plan['arms'][arm]['retention_weight']*keep-.001*dist.entropy().sum(-1).mean();assert torch.isfinite(loss)
                        optimizer.zero_grad(set_to_none=True);loss.backward();policy.body.net[0].weight.grad[:,[57,58,74,75,86,87,98,99,110,111]]=0
                        torch.nn.utils.clip_grad_norm_(policy.parameters(),.5);optimizer.step();updates+=1
                        kl=float(((ratio-1)-logratio).detach().mean());last={'ppo':float(surrogate.detach().mean()),'retention':float(keep.detach()),'conditional_kl':kl}
                        if kl>.02:early=True;break
                    if early:break
                for row in batch:
                    row.pop('_trajectory');episodes.append(row);training_rows.append(row);frames+=row['steps']
                    with (OUT/'episodes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                assert frames<=plan['max_physics_per_arm']
                checkpoint=folder/'checkpoint_latest.pt';previous=folder/'checkpoint_previous.pt';partial=folder/'checkpoint_partial.pt'
                torch.save({'policy':policy.state_dict(),'optimizer':optimizer.state_dict(),'arm':arm,'batches':iteration+1,'physics_steps':frames,
                            'actor_updates':updates,'algorithm':'AR_GAUSSIAN_PPO_MC_LOO_NO_CRITIC','protocol_sha256':sha(HERE/'protocol.json'),
                            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':rng.bit_generator.state,
                            'exact_live_environment_resume':False,'native_command_domain':[-1,1],'noise_correlation_s':current_tau},partial)
                assert partial.stat().st_size<32*1024**2
                if previous.exists():previous.unlink()
                if checkpoint.exists():os.replace(checkpoint,previous)
                os.replace(partial,checkpoint)
                status={'state':'TRAINING','arm':arm,'batch':iteration+1,'physics_steps_this_arm':frames,'actor_updates_this_arm':updates,
                        'training_episodes_this_arm':len(episodes),'training_success_this_arm':sum(r['success'] for r in episodes),'course_bin':course_bin,'last_losses':last,'elapsed_s':time.monotonic()-start}
                save(OUT/'status.json',status);print(json.dumps(status),flush=True)
            all_results[arm]={'checkpoint':str(checkpoint),'sha256':sha(checkpoint),'physics_steps':frames,'actor_updates':updates,
                              'training_episodes':len(episodes),'training_success':sum(r['success'] for r in episodes),
                              'training_collisions':sum(r['collision'] for r in episodes)}
            del policy,optimizer
        save(OUT/'status.json',{'state':'FROZEN_EVALUATION','training':all_results})
        eval_rows=[];eval_physics=0
        model_specs={'initial_embedding':{'checkpoint':plan['initial_policy_checkpoint'],'arm':'full_range'},**{k:{'checkpoint':v['checkpoint'],'arm':k} for k,v in all_results.items()}}
        for name,spec in model_specs.items():
            ck=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False);weights={k:v.cpu().numpy().copy() for k,v in ck['policy'].items()};del ck
            jobs=[{'id':c['id'],'case':c,'stochastic':False,'collect':False,'arm':spec['arm'],'seed':0} for c in plan['primary_cases']]
            jobs.extend({**j,'stochastic':False,'collect':False,'arm':spec['arm'],'seed':0} for j in plan['evaluation_resets'])
            for offset in range(0,len(jobs),8):
                subset=jobs[offset:offset+8]
                for pipe,job in zip(pipes,subset):pipe.send((job,weights))
                for pipe,job in zip(pipes,subset):
                    row=receive(pipe);row['model']=name;row['group']=job['reset_group'] if 'snapshot_side' in job else 'original_5m' if job['case']['distance_m']==5 else 'old_3m_4m'
                    eval_rows.append(row);eval_physics+=row['steps'];assert eval_physics<=plan['max_evaluation_physics']
                    with (OUT/'evaluation.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        scores={}
        for name in model_specs:
            scores[name]={}
            for group in ['old_3m_4m','original_5m','development_grid','fresh_holdout']:
                subset=[r for r in eval_rows if r['model']==name and r['group']==group]
                scores[name][group]={'safe_success':sum(r['success'] and not r['collision'] for r in subset),'n':len(subset),'collisions':sum(r['collision'] for r in subset)}
        for arm in all_results:
            prev=OUT/arm/'checkpoint_previous.pt'
            if prev.exists():save(OUT/arm/'discarded_predecessor.json',{'sha256':sha(prev),'bytes':prev.stat().st_size,'reason':'Owned rolling predecessor, frozen final retained'});prev.unlink()
        summary={'state':'COMPLETE','training':all_results,'scores':scores,'max_cpu_gpu_behavior_logp_error':max_logp_error,
                 'automatic_promotion':False,'incumbent':'T159','scope':'One paired pilot seed, two full-state near-goal anchors. Only training reset perturbation differs; same parent and PPO objective. Local robustness is not original5m performance. Development grid already consumed in T186; fresh fixed holdout not used for training/selection. No automatic promotion or algorithm superiority claim.'}
        save(OUT/'summary.json',summary);save(OUT/'status.json',{'state':'COMPLETE','training':all_results,'scores':scores,'elapsed_s':time.monotonic()-start})
        print(json.dumps(summary),flush=True)
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except(OSError,EOFError):pass
        for p in workers:
            p.join(5)
            if p.is_alive():p.terminate();p.join(5)
        for pipe in pipes:pipe.close()

if __name__=='__main__':main()
