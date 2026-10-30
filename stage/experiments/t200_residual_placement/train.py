"""Bounded paired SAC pilot: fresh critics, identical tasks, HER data ablation."""
import os
CPU_POOL=[int(v) for v in os.environ.get('STAGE_CPU_POOL',','.join(map(str,sorted(os.sched_getaffinity(0))))).split(',')]
os.environ['STAGE_CPU_POOL']=','.join(map(str,CPU_POOL))
os.sched_setaffinity(0,{int(os.environ.get('STAGE_PROCESS_CPU',str(CPU_POOL[0])))})
import copy,fcntl,multiprocessing as mp,shutil,time,traceback
from common import *
from collect import run
from replay import Replay

def used():return sum(p.stat().st_size for p in (REPO/'stage/runs').rglob('*') if p.is_file())

def main():
    lock=(OUT/'training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'status.json').exists(),'No duplicate or implicit resume'
    plan=json.loads((HERE/'protocol.json').read_text())
    for p,h in plan['input_hashes'].items():assert sha(p)==h,p
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.35);torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    assert json.loads((OUT/'preparation.json').read_text())['state']=='PASS'
    parent=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['policy']
    with np.load(plan['retention_targets'],allow_pickle=False) as z:
        anchors=torch.tensor(z['states'],device='cuda');targets=torch.tensor(z['actions'],device='cuda');offsets=z['offsets'].copy()
    start=time.monotonic();deadline=start+plan['wall_seconds'];ctx=mp.get_context('spawn');pipes=[];workers=[];training={};eval_rows=[]
    def payload(policy):return {k:v.detach().cpu().numpy().copy() for k,v in policy.state_dict().items()}
    def receive(pipe):
        assert pipe.poll(max(.1,min(180,deadline-time.monotonic()))),'Native episode timeout'
        row=pipe.recv();assert 'error' not in row,row;return row
    def evaluate(policy,name,jobs):
        weights=payload(policy);rows=[]
        for off in range(0,len(jobs),8):
            subset=jobs[off:off+8]
            for pipe,source in zip(pipes,subset):pipe.send(({**source,'id':name+'_'+source['id'],'arm':name,'stochastic':False,'collect':False,'seed':0,'placement':policy.placement},weights))
            for pipe,source in zip(pipes,subset):
                row=receive(pipe);row['model']=name;row['group']=source.get('group','near_goal' if source.get('snapshot_side') else 'original_5m' if source['case']['distance_m']==5 else 'old6');rows.append(row)
                with (OUT/'evaluation.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        return rows
    try:
        for i in range(8):
            a,b=ctx.Pipe();p=ctx.Process(target=run,args=(b,i,deadline));os.environ['STAGE_PROCESS_CPU']=str(CPU_POOL[(i+1)%len(CPU_POOL)]);p.start();os.environ['STAGE_PROCESS_CPU']=str(CPU_POOL[0]);b.close();pipes.append(a);workers.append(p)
        save(OUT/'workers.json',{'parent':os.getpid(),'pids':[p.pid for p in workers]})
        initial=Policy().cuda();initial.load_state_dict(parent)
        pre=evaluate(initial,'initial',[{'id':c['id'],'case':c} for c in plan['primary_cases']])
        assert sum(r['success'] and not r['collision'] for r in pre[:6])==6 and pre[7]['success'] and not pre[7]['collision'],'Initial seven-task gate failed'
        eval_rows.extend(pre);del initial
        for arm in ['native','latent']:
            torch.manual_seed(plan['seed']);torch.cuda.manual_seed_all(plan['seed']);rng=np.random.default_rng(plan['seed'])
            policy=Policy(placement=arm).cuda();policy.load_state_dict(parent)
            with torch.no_grad():policy.logstd.copy_(torch.log(torch.tensor(plan['initial_std'],device='cuda')))
            q=Critic().cuda();qt=copy.deepcopy(q).requires_grad_(False)
            actor_opt=torch.optim.Adam([{'params':list(policy.body.parameters()),'lr':1e-5},{'params':list(policy.native_head.parameters())+list(policy.direct_head.parameters())+list(policy.extra_head.parameters()),'lr':1e-4},{'params':[policy.logstd],'lr':1e-4}])
            critic_opt=torch.optim.Adam(q.parameters(),lr=3e-4)
            replay=Replay(plan['seed']);folder=OUT/arm;folder.mkdir();episodes=[];frames=decisions=actor_updates=critic_updates=0
            for p in plan['seed_episodes']:replay.add(p)
            replay.materialize();losses={}
            def update(update_actor):
                nonlocal actor_updates,critic_updates,losses
                x=replay.sample(256,'cuda');active=x['discounts'][:,0]>0
                with torch.no_grad():
                    target=x['rewards'].clone()
                    if active.any():
                        next_a,logp=policy.sample(x['next_states'][active]);a,b=qt(x['next_states'][active],next_a)
                        target[active]+=x['discounts'][active]*(torch.minimum(a,b)-plan['alpha']*logp)
                a,b=q(x['states'],x['actions']);loss_q=(a-target).square().mean()+(b-target).square().mean();assert torch.isfinite(loss_q)
                critic_opt.zero_grad(set_to_none=True);loss_q.backward();torch.nn.utils.clip_grad_norm_(q.parameters(),1.);critic_opt.step();critic_updates+=1
                with torch.no_grad():
                    for t,s in zip(qt.parameters(),q.parameters()):t.lerp_(s,.005)
                losses={'critic_loss':float(loss_q.detach()),'target_min':float(target.min()),'target_max':float(target.max())}
                if update_actor:
                    q.requires_grad_(False);a,lp=policy.sample(x['states']);qa,qb=q(x['states'],a)
                    ids=np.concatenate([rng.integers(offsets[j],offsets[j+1],32) for j in range(7)])
                    keep=(policy(anchors[ids])-targets[ids]).square().mean()
                    loss_a=(plan['alpha']*lp-torch.minimum(qa,qb)).mean()+10.*keep;assert torch.isfinite(loss_a)
                    actor_opt.zero_grad(set_to_none=True);loss_a.backward();policy.body.net[0].weight.grad[:,[57,58,74,75,86,87,98,99,110,111]]=0
                    torch.nn.utils.clip_grad_norm_(policy.parameters(),.5);actor_opt.step();q.requires_grad_(True);actor_updates+=1
                    with torch.no_grad():policy.logstd.clamp_(-3.,0.)
                    losses.update(actor_loss=float(loss_a.detach()),retention=float(keep.detach()))
            # Fresh critics receive fixed seed data before any policy gradient.
            save(OUT/'status.json',{'state':'CRITIC_WARMUP','arm':arm,'critic_updates':0})
            for i in range(plan['warmup_updates']):
                assert time.monotonic()<deadline;update(False)
            def terminal_metrics():
                result={}
                with torch.no_grad():
                    for name,group in [('original',replay.orig)]:
                        if group is None:continue
                        ids=np.flatnonzero(group['terminal'][:,0]);s=torch.tensor(group['states'][ids],device='cuda');a=torch.tensor(group['actions'][ids],device='cuda');y=torch.tensor(group['rewards'][ids],device='cuda');u,v=q(s,a)
                        result[name]={'n':len(ids),'q1_mae':float((u-y).abs().mean()),'q2_mae':float((v-y).abs().mean())}
                return result
            gate=terminal_metrics();save(folder/'critic_warmup.json',{'metrics':gate,'updates':critic_updates,'threshold':.15,'scope':'Observed terminal target sanity only, not unseen action-ranking validation or complete value calibration.'})
            assert all(max(r['q1_mae'],r['q2_mae'])<.15 for r in gate.values()),'Fresh critic terminal sanity gate failed; stop pilot without actor updates'
            for batch in range(plan['batches_per_arm']):
                assert used()+plan['pending_batch_bytes']<1024**3 and shutil.disk_usage(OUT).free>40*1024**3
                assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())+plan['pending_batch_bytes']<plan['peak_bytes']
                weights=payload(policy)
                for i,pipe in enumerate(pipes):
                    source=plan['training_jobs'][i];job={**source,'id':f'{arm}_b{batch}_w{i}','arm':arm,'stochastic':True,'collect':True,'seed':plan['seed']+batch*100+i,'placement':arm};pipe.send((job,weights))
                behavior=folder/f'behavior_batch_{batch}.npz';np.savez_compressed(behavior,**weights);assert behavior.stat().st_size<3*1024**2
                rows=[receive(pipe) for pipe in pipes]
                for row in rows:
                    assert Path(row['path']).stat().st_size<256*1024
                    replay.add(row['path']);episodes.append(row);frames+=row['physics_steps'];decisions+=row['decisions'];row['behavior_checkpoint']=str(behavior);row['behavior_sha256']=sha(behavior)
                    with (OUT/'episodes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                replay.materialize()
                for i in range(plan['updates_per_batch']):
                    assert time.monotonic()<deadline;update(True)
                assert frames<=plan['max_physics_per_arm']
                partial=folder/'checkpoint_partial.pt';path=folder/'checkpoint_latest.pt';prev=folder/'checkpoint_previous.pt'
                torch.save({'policy':policy.state_dict(),'critic':q.state_dict(),'critic_target':qt.state_dict(),'actor_optimizer':actor_opt.state_dict(),'critic_optimizer':critic_opt.state_dict(),'arm':arm,'physics_steps':frames,'actor_updates':actor_updates,'critic_updates':critic_updates,'protocol_sha256':sha(HERE/'protocol.json'),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'rng':rng.bit_generator.state,'replay_rng':replay.rng.bit_generator.state,'replay_sources':replay.receipts,'exact_live_environment_resume':False},partial)
                assert partial.stat().st_size<12*1024**2
                if prev.exists():prev.unlink()
                if path.exists():os.replace(path,prev)
                os.replace(partial,path)
                status={'state':'TRAINING','arm':arm,'batch':batch+1,'physics_steps':frames,'decisions':decisions,'actor_updates':actor_updates,'critic_updates':critic_updates,'episodes':len(episodes),'success':sum(r['success'] for r in episodes),'original5m_success':sum(r['success'] for r in episodes if not r.get('snapshot_side')),'losses':losses,'elapsed_s':time.monotonic()-start}
                save(OUT/'status.json',status);print(json.dumps(status),flush=True)
            save(folder/'replay_receipts.json',replay.receipts)
            training[arm]={'checkpoint':str(path),'sha256':sha(path),'physics_steps':frames,'decisions':decisions,'actor_updates':actor_updates,'critic_updates':critic_updates,'episodes':len(episodes),'success':sum(r['success'] for r in episodes),'original5m_success':sum(r['success'] for r in episodes if not r.get('snapshot_side')),'original5m_n':sum(not r.get('snapshot_side') for r in episodes),'terminal_metrics':terminal_metrics(),'original_replay_transitions':len(replay.orig['states']),'replay_transitions':len(replay.orig['states'])}
            final_jobs=[{'id':c['id'],'case':c} for c in plan['primary_cases']]+plan['evaluation_resets']
            eval_rows.extend(evaluate(policy,arm,final_jobs))
            prev=OUT/arm/'checkpoint_previous.pt'
            if prev.exists():save(OUT/arm/'discarded_predecessor.json',{'bytes':prev.stat().st_size,'sha256':sha(prev),'reason':'Owned predecessor; evaluated final retained'});prev.unlink()
            # Completed budget and final evaluation: preserve all model weights,
            # remove only optimizer payloads after exact three-network verification.
            assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())+5*1024**2<plan['peak_bytes']
            original=torch.load(path,map_location='cpu',weights_only=False)
            compact={k:v for k,v in original.items() if k not in ['actor_optimizer','critic_optimizer']}
            compact['complete_optimizer_resume_supported']=False
            compact['optimizer_states_removed_after_budget_completion']=True
            slim=folder/'checkpoint_compact_partial.pt';torch.save(compact,slim);check=torch.load(slim,map_location='cpu',weights_only=False)
            for role in ['policy','critic','critic_target']:
                assert all(torch.equal(v,check[role][k]) for k,v in original[role].items())
            receipt={'original_full_sha256':sha(path),'original_full_bytes':path.stat().st_size,'compact_sha256':sha(slim),'compact_bytes':slim.stat().st_size,'all_actor_and_twin_critic_weights_exact':True,'reason':'Budget complete, final policy evaluated. Only unneeded optimizer state removed; no claim of exact optimizer resume.'}
            os.replace(slim,path);save(folder/'optimizer_cleanup_receipt.json',receipt)
            training[arm]['sha256']=sha(path);training[arm]['complete_optimizer_resume_supported']=False
            del policy,q,qt,actor_opt,critic_opt,replay,original,compact,check

        scores={}
        for name in ['initial','native','latent']:
            scores[name]={}
            for group in ['old6','original_5m','left_fresh_reset']:
                selected=[r for r in eval_rows if r['model']==name and r['group']==group];scores[name][group]={'safe_success':sum(r['success'] and not r['collision'] for r in selected),'n':len(selected),'collision':sum(r['collision'] for r in selected),'timeout':sum(r['timeout'] for r in selected)}
        assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<plan['max_bytes']
        result={'state':'COMPLETE','training':training,'scores':scores,'evaluation_physics':sum(r['physics_steps'] for r in eval_rows),'automatic_promotion':False,'incumbent':'T159','scope':plan['scope'],'diagnostic_physics':0}
        save(OUT/'summary.json',result);save(OUT/'status.json',{**result,'elapsed_s':time.monotonic()-start});print(json.dumps(result),flush=True)
    except BaseException:
        save(OUT/'failure.json',{'state':'FAILED','traceback':traceback.format_exc(),'completed_arms':training,'unrecorded_inflight_physics_upper_bound':8*23334});raise
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except(OSError,EOFError):pass
        for p in workers:
            p.join(5)
            if p.is_alive():p.terminate();p.join(5)
        for pipe in pipes:pipe.close()

if __name__=='__main__':main()
