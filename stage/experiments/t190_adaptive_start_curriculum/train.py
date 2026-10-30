"""Bounded online MC/PPO with unordered, competence-selected training resets."""
import os
CPU_POOL=[int(v) for v in os.environ.get('STAGE_CPU_POOL',','.join(map(str,sorted(os.sched_getaffinity(0))))).split(',')]
os.environ['STAGE_CPU_POOL']=','.join(map(str,CPU_POOL))
os.sched_setaffinity(0,{int(os.environ.get('STAGE_PROCESS_CPU',str(CPU_POOL[0])))})
import fcntl,math,multiprocessing as mp,shutil,time,traceback
from common import *
from worker import run

def used():return sum(p.stat().st_size for p in (REPO/'stage/runs').rglob('*') if p.is_file())

def main():
    lock=(OUT/'training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'status.json').exists(),'No duplicate or implicit resume'
    plan=json.loads((HERE/'protocol.json').read_text())
    for p,h in plan['input_hashes'].items():assert sha(p)==h,p
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.35);torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    torch.manual_seed(plan['seed']);torch.cuda.manual_seed_all(plan['seed'])
    rng=np.random.default_rng(plan['seed']);proposals_rng=np.random.default_rng(plan['seed']+10000)
    policy=Policy().cuda();policy.load_state_dict(torch.load(plan['parent_checkpoint'],map_location='cpu',weights_only=False)['policy'])
    with torch.no_grad():policy.logstd.copy_(torch.log(torch.tensor(plan['initial_std'],device='cuda')))
    groups=[{'params':list(policy.body.parameters()),'lr':1e-5,'name':'body'},{'params':list(policy.native_head.parameters()),'lr':1e-4,'name':'native_head'},{'params':list(policy.direct_head.parameters()),'lr':1e-4,'name':'direct_head'},{'params':[policy.logstd],'lr':1e-3,'name':'logstd'}]
    optimizer=torch.optim.Adam(groups)
    save(OUT/'optimizer_groups.json',[{'name':g['name'],'lr':g['lr'],'parameters':sum(v.numel() for v in g['params'])} for g in groups])
    with np.load(plan['retention_demonstrations'],allow_pickle=False) as z:
        states=z['states'].copy();actions=z['actions'].copy();offsets=z['episode_offsets'].copy();counts=z['native_counts'].copy()
    actions[:,0]=.5*(actions[:,0]+1);clocks=[]
    for j in range(6):
        lo,hi=offsets[j:j+2];clocks.extend(1-np.r_[0,np.cumsum(counts[lo:hi-1])]/plan['primary_cases'][j]['step_cap'])
    anchors=torch.tensor(np.column_stack([states,np.asarray(clocks,dtype=np.float32)]),device='cuda');targets=torch.tensor(actions,device='cuda')
    sides=['LEFT','RIGHT'];root={side:{'id':'root_'+side,'snapshot_side':side,'delta_x_m':0.,'delta_y_m':0.,'delta_yaw_deg':0.,'reset_group':'near_goal'} for side in sides}
    pool={side:[root[side]] for side in sides}
    original_xy={'LEFT':np.array([-.75,.75]),'RIGHT':np.array([1.,2.25])}
    original={c['side']:{'id':c['id'],'case':c,'reset_group':'original_5m'} for c in plan['primary_cases'][-2:]}
    start=time.monotonic();deadline=start+plan['wall_seconds'];ctx=mp.get_context('spawn');pipes=[];workers=[]
    frames=decisions=updates=probe_frames=0;episodes=[];rounds=[];max_logp_error=0.;stop_reason='round_budget'
    def payload():return {k:v.detach().cpu().numpy().copy() for k,v in policy.state_dict().items()}
    def receive(pipe):
        assert pipe.poll(max(.1,min(180,deadline-time.monotonic()))),'Native receive timeout'
        row=pipe.recv();assert 'error' not in row,row;return row
    def evaluate(jobs,tag):
        weights=payload();rows=[]
        for off in range(0,len(jobs),8):
            subset=jobs[off:off+8]
            for pipe,source in zip(pipes,subset):
                job={**source,'source_id':source['id'],'stochastic':False,'collect':False,'proposal_probe':True,'save_trace':True,'arm':'curriculum','seed':0,'model_tag':tag};pipe.send((job,weights))
            for pipe in pipes[:len(subset)]:
                row=receive(pipe);row['evaluation_tag']=tag;rows.append(row)
                with (OUT/'evaluation.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        return rows
    def conditional(x):
        rho=math.exp(-.2)
        mean=policy(x['states'])+rho*(x['previous_latents']-policy(x['previous_states']))*x['previous_mask'][:,None]
        std=policy.logstd.clamp(-1.6,0.).exp()[None,:]*torch.where(x['previous_mask'][:,None]>0,math.sqrt(1-rho*rho),1.)
        return torch.distributions.Normal(mean,std)
    def xy(job):
        base=plan['anchor_sources'][job['snapshot_side']]['reset_task']['start_pose']
        return np.asarray(base[:2])+[job['delta_x_m'],job['delta_y_m']]
    def checkpoint(round_index):
        path=OUT/'checkpoint_latest.pt';prev=OUT/'checkpoint_previous.pt';temp=OUT/'checkpoint_partial.pt'
        torch.save({'policy':policy.state_dict(),'optimizer':optimizer.state_dict(),'physics_steps':frames,'actor_updates':updates,'training_decisions':decisions,'completed_round':round_index,'protocol_sha256':sha(HERE/'protocol.json'),'pool':pool,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'rng':rng.bit_generator.state,'proposal_rng':proposals_rng.bit_generator.state,'exact_live_environment_resume':False},temp)
        assert temp.stat().st_size<32*1024**2
        if prev.exists():prev.unlink()
        if path.exists():os.replace(path,prev)
        os.replace(temp,path);return path
    try:
        for i in range(8):
            a,b=ctx.Pipe();p=ctx.Process(target=run,args=(b,i,deadline));os.environ['STAGE_PROCESS_CPU']=str(CPU_POOL[(i+1)%len(CPU_POOL)]);p.start();os.environ['STAGE_PROCESS_CPU']=str(CPU_POOL[0]);b.close();pipes.append(a);workers.append(p)
        save(OUT/'workers.json',{'parent':os.getpid(),'pids':[p.pid for p in workers]})
        for round_index in range(plan['rounds']):
            assert used()+80*1024**2<1024**3 and shutil.disk_usage(OUT).free>40*1024**3
            save(OUT/'status.json',{'state':'PROBING','round':round_index,'training_physics':frames,'probe_physics':probe_frames})
            proposed=[]
            for side in sides:
                for j in range(plan['proposal_per_side']):
                    parent=pool[side][j%len(pool[side])];angle=float(proposals_rng.uniform(-math.pi,math.pi));radius=float(proposals_rng.uniform(*plan['proposal_radius_m']))
                    job={**parent,'id':f'round{round_index}_{side}_{j}','delta_x_m':parent['delta_x_m']+radius*math.cos(angle),'delta_y_m':parent['delta_y_m']+radius*math.sin(angle),'delta_yaw_deg':parent['delta_yaw_deg']+float(proposals_rng.uniform(-plan['proposal_yaw_deg'],plan['proposal_yaw_deg'])),'reset_group':'proposal','proposal_parent_id':parent['id']}
                    proposed.append(job)
            save(OUT/f'round_{round_index}_proposals.json',{'pool_before':pool,'jobs':proposed,'rng_after':proposals_rng.bit_generator.state})
            pre=evaluate(proposed,f'round{round_index}_before');probe_frames+=sum(r['steps'] for r in pre)
            valid={side:[j for j,r in zip(proposed,pre) if j['snapshot_side']==side and not r['invalid_reset']] for side in sides}
            failed={side:[j for j,r in zip(proposed,pre) if j['snapshot_side']==side and not r['invalid_reset'] and not r['success']] for side in sides}
            assert all(valid.values()),'No valid proposed reset for one side; stop without resampling'
            for b in range(plan['batches_per_round']):
                selected=[(failed[side] or valid[side])[b%len(failed[side] or valid[side])] for side in sides]
                support=sides[b%2];selected.extend([pool[support][(b//2)%len(pool[support])],original[sides[b%2]]])
                weights=payload()
                for i,pipe in enumerate(pipes):
                    source=selected[i%4];job={**source,'id':f'r{round_index}_b{b}_w{i}','source_id':source['id'],'arm':'curriculum','stochastic':True,'collect':True,'seed':plan['seed']+round_index*10000+b*100+i,'noise_correlation_s':1.}
                    pipe.send((job,weights))
                batch=[receive(pipe) for pipe in pipes]
                for i,row in enumerate(batch):
                    assert not row['invalid_reset'];assert row['source']==batch[(i+4)%8]['source'];assert row['initial_reset_signature']==batch[(i+4)%8]['initial_reset_signature']
                arrays=[]
                for i,row in enumerate(batch):
                    tr=row['trajectory'];partner=batch[(i+4)%8]['trajectory']['returns'];baseline=np.zeros_like(tr['returns']);n=min(len(baseline),len(partner));baseline[:n]=partner[:n];tr['advantages']=(tr['returns']-baseline)/10.;arrays.append(tr)
                joined={k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]};np.savez_compressed(OUT/f'round{round_index}_batch{b}.npz',**joined)
                tensors={k:torch.tensor(v,device='cuda') for k,v in joined.items()}
                with torch.no_grad():
                    err=float((conditional(tensors).log_prob(tensors['latents']).sum(-1)-tensors['old_logp']).abs().max());max_logp_error=max(max_logp_error,err);assert err<2e-4,err
                for epoch in range(4):
                    permutation=rng.permutation(len(joined['states']));early=False
                    for begin in range(0,len(permutation),128):
                        ids=permutation[begin:begin+128];x={k:v[ids] for k,v in tensors.items()};dist=conditional(x)
                        logratio=dist.log_prob(x['latents']).sum(-1)-x['old_logp'];ratio=logratio.exp();adv=x['advantages']
                        surrogate=-torch.minimum(ratio*adv,ratio.clamp(.9,1.1)*adv)
                        keep_ids=np.concatenate([rng.integers(offsets[j],offsets[j+1],32) for j in range(6)])
                        keep=(policy(anchors[keep_ids])-targets[keep_ids]).square().mean()
                        loss=(surrogate*x['policy_weights']).mean()+10.*keep-.001*dist.entropy().sum(-1).mean();assert torch.isfinite(loss)
                        optimizer.zero_grad(set_to_none=True);loss.backward();policy.body.net[0].weight.grad[:,[57,58,74,75,86,87,98,99,110,111]]=0
                        torch.nn.utils.clip_grad_norm_(policy.parameters(),.5);optimizer.step();updates+=1
                        kl=float(((ratio-1)-logratio).detach().mean())
                        if kl>.02:early=True;break
                    if early:break
                for row in batch:
                    row.pop('trajectory');row['round']=round_index;row['batch']=b;episodes.append(row);frames+=row['steps'];decisions+=row['decisions']
                    with (OUT/'episodes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                assert frames<=plan['max_training_physics']
                checkpoint(round_index)
                status={'state':'TRAINING','round':round_index,'batch':b+1,'training_physics':frames,'training_decisions':decisions,'updates':updates,'episodes':len(episodes),'success':sum(r['success'] for r in episodes),'original5m_success':sum(r['success'] for r in episodes if r['reset_group']=='original_5m'),'elapsed_s':time.monotonic()-start}
                save(OUT/'status.json',status);print(json.dumps(status),flush=True)
            post=evaluate(proposed,f'round{round_index}_after');probe_frames+=sum(r['steps'] for r in post)
            gates=evaluate([{'id':c['id'],'case':c} for c in plan['primary_cases']],f'round{round_index}_primary');probe_frames+=sum(r['steps'] for r in gates)
            assert probe_frames<=plan['max_probe_physics']
            for a,z in zip(pre,post):
                assert a['invalid_reset']==z['invalid_reset']
                if not a['invalid_reset']:assert a['initial_reset_signature']==z['initial_reset_signature']
            for side in sides:
                successful=[j for j,r in zip(proposed,post) if j['snapshot_side']==side and r['success'] and not r['collision']]
                # Previous non-root sources must be successful under the current policy to be reused.
                # Only freshly tested candidates plus root survive; root is separately tested below.
                successful.sort(key=lambda j:float(np.linalg.norm(xy(j)-original_xy[side])))
                chosen=[root[side]]+successful[:2]
                remainder=successful[2:]
                if remainder:chosen.append(max(remainder,key=lambda j:min(float(np.linalg.norm(xy(j)-xy(k))) for k in chosen)))
                pool[side]=chosen
            root_checks=evaluate(list(root.values()),f'round{round_index}_roots');probe_frames+=sum(r['steps'] for r in root_checks)
            assert probe_frames<=plan['max_probe_physics']
            for side,row in zip(sides,root_checks):
                if not row['success']:pool[side]=[j for j in pool[side] if j['id']!=root[side]['id']]
            score={'round':round_index,'proposals':len(pre),'invalid':sum(r['invalid_reset'] for r in pre),'before_success':sum(r['success'] for r in pre),'after_success':sum(r['success'] for r in post),'gained':[a['id'] for a,z in zip(pre,post) if not a['success'] and z['success']],'lost':[a['id'] for a,z in zip(pre,post) if a['success'] and not z['success']],'old6_success':sum(r['success'] and not r['collision'] for r in gates[:6]),'original5m_success':sum(r['success'] and not r['collision'] for r in gates[6:]),'pool':pool}
            rounds.append(score);save(OUT/f'round_{round_index}_result.json',score);print(json.dumps({'state':'ROUND_COMPLETE',**score}),flush=True)
            checkpoint(round_index)
            if score['old6_success']<6:stop_reason='old6_regression';break
            if not all(pool.values()):stop_reason='no_successful_curriculum_source';break
            if score['original5m_success']==2:stop_reason='primary_tasks_passed';break
        final_jobs=[{'id':c['id'],'case':c} for c in plan['primary_cases']]+plan['evaluation_resets']
        final=evaluate(final_jobs,'final');final_frames=sum(r['steps'] for r in final);assert final_frames<=plan['max_final_physics']
        scores={}
        for group in ['old_3m_4m','original_5m','development_grid','fresh_holdout']:
            subset=[r for j,r in zip(final_jobs,final) if (j.get('reset_group') or ('original_5m' if j['case']['distance_m']==5 else 'old_3m_4m'))==group]
            scores[group]={'safe_success':sum(r['success'] and not r['collision'] for r in subset),'n':len(subset),'collision':sum(r['collision'] for r in subset)}
        path=OUT/'checkpoint_latest.pt';prev=OUT/'checkpoint_previous.pt'
        if prev.exists():save(OUT/'discarded_predecessor.json',{'sha256':sha(prev),'bytes':prev.stat().st_size,'reason':'Owned rolling predecessor; evaluated final retained'});prev.unlink()
        result={'state':'COMPLETE','stop_reason':stop_reason,'rounds':rounds,'scores':scores,'training_physics':frames,'training_decisions':decisions,'optimizer_updates':updates,'training_episodes':len(episodes),'training_success':sum(r['success'] for r in episodes),'original5m_training_success':sum(r['success'] for r in episodes if r['reset_group']=='original_5m'),'original5m_training_n':sum(r['reset_group']=='original_5m' for r in episodes),'probe_physics':probe_frames,'final_evaluation_physics':final_frames,'max_cpu_gpu_logp_error':max_logp_error,'checkpoint_sha256':sha(path),'automatic_promotion':False,'incumbent':'T159','scope':plan['scope']}
        save(OUT/'summary.json',result);save(OUT/'status.json',{**result,'elapsed_s':time.monotonic()-start});print(json.dumps(result),flush=True)
    except BaseException:
        save(OUT/'failure.json',{'state':'FAILED','traceback':traceback.format_exc(),'recorded_training_physics':frames,'recorded_probe_physics':probe_frames,'unrecorded_inflight_upper_bound':8*23334});raise
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except(OSError,EOFError):pass
        for p in workers:
            p.join(5)
            if p.is_alive():p.terminate();p.join(5)
        for pipe in pipes:pipe.close()

if __name__=='__main__':main()
