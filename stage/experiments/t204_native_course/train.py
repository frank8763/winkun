"""One bounded daytime pilot: true episodic policy gradients, matched retention ablation."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['CUDA_CACHE_DISABLE']='1';os.sched_setaffinity(0,{0})
from pathlib import Path
import sys,json,time,fcntl,multiprocessing as mp,shutil,copy,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t204_native_course'
sys.path.insert(0,str(ROOT/'stage/experiments/t203_native_exploration'))
from runtime import *

class BudgetStop(Exception):pass
def write(name,value):C.save(OUT/name,value)
def append(name,value):
    with (OUT/name).open('a') as f:f.write(json.dumps(value)+'\n')
def own():return sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
def allused():return sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file())

def main():
    lock=(OUT/'train.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'status.json').exists(),'No duplicate or implicitresume'
    plan=json.loads((HERE/'protocol.json').read_text())
    for path,h in plan['input_hashes'].items():assert C.sha(path)==h,path
    assert json.loads((OUT/'contracts.json').read_text())['state']=='PASS'
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True);torch.cuda.set_per_process_memory_fraction(.25)
    start=time.time();deadline=plan['deadline_unix'];ctx=mp.get_context('spawn');pipes=[];workers=[];policies={};accepted={};stats={'training_physics':0,'diagnostic_physics':0,'replay_physics':0,'training_episodes':0,'diagnostic_episodes':0,'partial_episodes':0,'actor_updates':0,'rounds_completed':0};pool=[];evaluations=[];reason=None;selected=None;inflight=0
    rng=np.random.default_rng(plan['seed']);arm_rng={a:np.random.default_rng(plan['seed']) for a in plan['arms']};round_index=-1;case=plan['primary_cases'][6]
    (OUT/'behaviors').mkdir();(OUT/'candidates').mkdir();(OUT/'exports').mkdir()
    parent=Policy().eval();parent.load_state_dict(torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['policy']);parent_file=OUT/'parent_heads.npz';save_heads(parent_file,parent);immutable={k:v.clone() for k,v in parent.state_dict().items() if k not in HEAD_KEYS}
    with np.load(plan['retention_targets']) as z:anchors=torch.tensor(z['states'],device='cuda');targets=torch.tensor(z['actions'],device='cuda');offsets=z['offsets'].copy()
    def total():return stats['training_physics']+stats['diagnostic_physics']+stats['replay_physics']
    def status(state,extra=None):write('status.json',{'state':state,**stats,'total_physics':total(),'round':round_index,'deadline_local':plan['deadline_local'],'elapsed_s':time.time()-start,'own_bytes':own(),'all_new_runs_bytes':allused(),'selected':selected,**(extra or {})})
    def guard(reserve=0,physics=0,final=False):
        if time.time()>=deadline:raise BudgetStop('WALL_DEADLINE')
        if total()+physics>plan['max_total_physics']:raise BudgetStop('PHYSICS_RESERVE')
        if allused()+reserve>=1024**3 or shutil.disk_usage(ROOT).free<=40*1024**3:raise BudgetStop('GLOBAL_STORAGE_RESERVE')
        if own()+reserve+plan['summary_reserve_bytes']+(0 if finalizing else plan['export_reserve_bytes'])>=plan['max_bytes']:raise BudgetStop('FINAL_STORAGE_RESERVE')
        if own()+reserve>=plan['peak_bytes']:raise BudgetStop('PEAK_STORAGE_RESERVE')
    def receive(pipe):
        if not pipe.poll(max(.1,min(180,deadline-time.time()+2))):raise BudgetStop('NATIVE_RESPONSE_DEADLINE')
        return pipe.recv()
    def tasks(spec):
        guard(reserve=16*1024);pipes[0].send(('make_task',spec));r=receive(pipes[0]);assert 'error' not in r,r;append('tasks.jsonl',{**spec,**r})
        return {**r,'case':spec['case'],'delta':spec.get('delta'),'original':not spec.get('near_snapshot')}
    def run_jobs(jobs,category='diagnostic'):
        nonlocal inflight
        result=[]
        for off in range(0,len(jobs),8):
            sub=jobs[off:off+8];guard(reserve=len(sub)*280*1024,physics=sum(j['case']['step_cap'] for j in sub));inflight=sum(j['case']['step_cap'] for j in sub)
            for pipe,j in zip(pipes,sub):pipe.send(('rollout',j))
            rr=[receive(pipe) for pipe in pipes[:len(sub)]];errors=[r for r in rr if 'error' in r]
            for r in rr:
                if 'error' in r:continue
                stats[category+'_physics']+=r['physics_steps'];stats[category+'_episodes']+=int(r['complete']);stats['partial_episodes']+=int(not r['complete']);append('rollouts.jsonl',r);result.append(r)
            inflight=0
            if errors:raise RuntimeError(json.dumps(errors))
            if not all(r['complete'] for r in result):raise BudgetStop('WALL_CENSORED_EPISODES')
            status('RUNNING')
        return result
    def verify_compact(rows):
        nonlocal inflight
        for off in range(0,len(rows),8):
            sub=[r for r in rows[off:off+8] if r.get('path')]
            if not sub:continue
            guard(reserve=len(sub)*40*1024,physics=sum(r['physics_steps'] for r in sub));inflight=sum(r['physics_steps'] for r in sub)
            for pipe,row in zip(pipes,sub):pipe.send(('replay',row))
            checks=[receive(pipe) for pipe in pipes[:len(sub)]]
            for r,v in zip(sub,checks):
                assert 'error' not in v,v;stats['replay_physics']+=v['physics_steps'];append('replay_checks.jsonl',v)
                if not v['verified']:raise BudgetStop('REPLAY_CENSORED_BY_WALL')
                compact(r,v,OUT)
            inflight=0
    def job(task,path,id,mode='deterministic',seed=0,arm=None,role='diagnostic',summary=True):
        return {'id':id,'task':task['id'],'case':task['case'],'exact_snapshot':task['snapshot'],'heads':str(path),'seed':int(seed),'mode':mode,'arm':arm,'round':round_index,'role':role,'summary_only':summary,'initial_pose':task['pose']}
    def eval_tasks(tasklist,path,prefix,arm=None,retain=False):
        rr=run_jobs([job(t,path,f'{prefix}_{i}',arm=arm,role='evaluation',summary=not retain) for i,t in enumerate(tasklist)])
        evaluations.extend(rr);verify_compact(rr);return rr
    def safe(r):return r['complete'] and r['success'] and not r['collision']
    def candidate(path,p):save_heads(path,p);return path
    def train_arm(arm,rows):
        p=policies[arm];parameters=p.heads_only();opt=torch.optim.Adam(parameters,lr=plan['head_lr']);groups={};arrays=[];returns=[];episode_adv={}
        for r in rows:groups.setdefault(r['task'],[]).append(r)
        rawadv=[]
        for rr in groups.values():
            assert len(rr)==3 and all(r['mode']=='physical_gaussian' and r['complete'] for r in rr)
            values=np.asarray([r['environment_return'] for r in rr]);adv=values-(values.sum()-values)/2
            for r,a in zip(rr,adv):episode_adv[r['id']]=float(a);rawadv.append(a)
        scale=max(plan['minimum_advantage_scale'],float(np.std(rawadv)));states=[];actions=[];oldlp=[];advantages=[]
        for r in rows:
            with np.load(r['path']) as z:
                states.append(z['states']);actions.append(z['actions']);oldlp.append(z['logp']);advantages.append(np.full(len(z['states']),episode_adv[r['id']]/scale))
        x=torch.tensor(np.concatenate(states),device='cuda');a=torch.tensor(np.concatenate(actions),device='cuda');old=torch.tensor(np.concatenate(oldlp),device='cuda');adv=torch.tensor(np.concatenate(advantages),device='cuda');n=len(x);factor=n/(len(rows)*234)
        with torch.no_grad():identity_error=float((p.physical_logp(x,a)-old).abs().max());assert identity_error<2e-5
        updates=0;last={};kl=0.
        for epoch in range(plan['epochs']):
            ids=arm_rng[arm].permutation(n)
            for off in range(0,n,plan['minibatch']):
                guard();ix=ids[off:off+plan['minibatch']];lp=p.physical_logp(x[ix],a[ix]);logratio=lp-old[ix];assert bool(torch.isfinite(logratio).all());ratio=logratio.exp();pg=-torch.minimum(ratio*adv[ix],ratio.clamp(1-plan['ppo_clip'],1+plan['ppo_clip'])*adv[ix]).mean()*factor
                keep_ids=np.concatenate([arm_rng[arm].integers(offsets[j],offsets[j+1],32) for j in range(7)]);errors=(p(anchors[keep_ids])-targets[keep_ids]).square().mean(1).reshape(7,32).mean(1)
                penalty=errors if arm=='keep10' else (errors-plan['retention_tolerance_rmse']**2).clamp_min(0)
                keep=penalty.mean()*plan['retention_weight'];loss=pg+keep;assert bool(torch.isfinite(loss));opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(parameters,.5);opt.step();updates+=1;stats['actor_updates']+=1;assert stats['actor_updates']<=plan['max_actor_updates']
                last={'policy_loss':float(pg.detach()),'weighted_retention_loss':float(keep.detach()),'retention_group_MSE':errors.detach().cpu().tolist()}
            with torch.no_grad():d=p.physical_logp(x,a)-old;kl=float((d.exp()-1-d).mean())
            if kl>plan['max_KL']:break
        for k,v in immutable.items():assert torch.equal(p.state_dict()[k].cpu(),v),k
        return {'arm':arm,'updates':updates,'epochs':epoch+1,'identity_log_density_error':identity_error,'empirical_KL':kl,'raw_episode_advantages':episode_adv,'advantage_scale':scale,'constant_trajectory_scale':factor,'critic_updates':0,**last}
    primary=[];cache={};pool_best=None;stalled=0;radius_index=0;expansion_streak=0;finalizing=False
    try:
        for i in range(8):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(i+1);w=ctx.Process(target=worker,args=(b,i,str(OUT),plan['checkpoint'],deadline));w.start();b.close();pipes.append(a);workers.append(w)
        os.environ['STAGE_PROCESS_CPU']='0';write('workers.json',{'parent':os.getpid(),'pids':[w.pid for w in workers]});status('INITIAL_GATES')
        for c in plan['primary_cases']:primary.append(tasks({'id':'primary_'+c['id'],'case':c}))
        initial=eval_tasks(primary,parent_file,'initial',retain=True);assert all(safe(initial[i]) for i in [0,1,2,3,4,5,7]),'Initial seven-task gate'
        root=tasks({'id':'source','case':case,'near_snapshot':plan['near_snapshot'],'delta':[0,0,0]});assert root['valid'];sr=eval_tasks([root],parent_file,'source')[0];assert safe(sr);pool=[root];original_xy=np.asarray(primary[6]['pose'][:2]);original_yaw=float(primary[6]['pose'][2]);pool_best=float(np.linalg.norm(np.asarray(root['pose'][:2])-original_xy));root_pose=np.asarray(root['pose'][:3]);cache[root['id']]=sr;cache[primary[6]['id']]=initial[6]
        for arm in plan['arms']:
            p=Policy().cuda().eval();p.load_state_dict(parent.state_dict());p.heads_only();policies[arm]=p;accepted[arm]=heads(p)
        for round_index in range(plan['max_rounds']):
            guard(reserve=plan['pending_round_bytes'],physics=plan['pending_round_physics']);active=[]
            ranked=sorted(pool,key=lambda t:float(np.linalg.norm(np.asarray(t['pose'][:2])-original_xy)))
            for slot in range(2):
                if slot==1 and (round_index+1)%4==0:active.append(primary[6]);continue
                radius=plan['course_radii_m'][radius_index]
                base=ranked[0] if slot==0 else ranked[int(rng.integers(min(8,len(ranked))))];basepose=np.asarray(base['pose'][:3]);direction=original_xy-basepose[:2];direction/=max(float(np.linalg.norm(direction)),1e-12);perp=np.asarray([-direction[1],direction[0]])
                found=None
                for attempt in range(4):
                    if slot==0:
                        heading=direction
                        if attempt>=2:
                            angle=math.radians(75 if attempt==2 else -75)
                            heading=math.cos(angle)*direction+math.sin(angle)*perp
                        sideways=np.asarray([-heading[1],heading[0]])
                        step=.8*radius*heading+float(rng.uniform(-.5,.5))*radius*sideways
                    else:
                        angle=float(rng.uniform(-math.pi,math.pi))
                        step=radius*float(rng.uniform(.5,1))*np.asarray([math.cos(angle),math.sin(angle)])
                    assert np.linalg.norm(step)<=radius+1e-12
                    yaw_error=(original_yaw-basepose[2]+math.pi)%(2*math.pi)-math.pi;yaw_step=float(np.clip(math.degrees(yaw_error)+rng.uniform(-1,1),-5,5));delta=[float(basepose[0]+step[0]-root_pose[0]),float(basepose[1]+step[1]-root_pose[1]),float(math.degrees(basepose[2]-root_pose[2])+yaw_step)]
                    t=tasks({'id':f'r{round_index}_s{slot}_a{attempt}','case':case,'near_snapshot':plan['near_snapshot'],'delta':delta,'radius_m':radius,'base_task':base['id'],'step_norm_m':float(np.linalg.norm(step))})
                    if t['valid']:found=t;break
                if found is None:raise BudgetStop('NO_VALID_MICRO_RESET')
                active.append(found)
            parent_rows={}
            for t in active:
                if t['id'] not in cache:cache[t['id']]=eval_tasks([t],parent_file,f'parent_r{round_index}_{t["id"]}')[0]
                parent_rows[t['id']]=cache[t['id']]
            collection={};round_metrics=[]
            for arm in plan['arms']:
                behavior=candidate(OUT/'behaviors'/f'{arm}_r{round_index}.npz',policies[arm]);jobs=[]
                for j,t in enumerate(active):
                    jobs.append(job(t,behavior,f'{arm}_r{round_index}_t{j}_det',arm=arm,role='course_control',summary=False))
                    for k in range(3):jobs.append(job(t,behavior,f'{arm}_r{round_index}_t{j}_n{k}',mode='physical_gaussian',seed=plan['seed']+round_index*100+j*10+k,arm=arm,role='training',summary=False))
                # Keep diagnostic controls and learning data separately accounted.
                controls=run_jobs([j for j in jobs if j['mode']=='deterministic']);stoch=run_jobs([j for j in jobs if j['mode']=='physical_gaussian'],'training');assert stats['training_episodes']<=plan['max_training_episodes'] and stats['training_physics']<=plan['max_training_physics'];collection[arm]=controls+stoch
                metric=train_arm(arm,stoch);proposal=heads(policies[arm]);gate_success=False;chosen=0.
                for factor in plan['interpolation_factors']:
                    blended={k:accepted[arm][k]+np.float32(factor)*(proposal[k]-accepted[arm][k]) for k in HEAD_KEYS};set_heads(policies[arm],blended);path=candidate(OUT/'candidates'/f'{arm}_r{round_index}_f{factor}.npz',policies[arm]);gate=eval_tasks([primary[i] for i in [0,1,2,3,4,5,7]],path,f'{arm}_r{round_index}_f{factor}_gate',arm)
                    if all(safe(r) for r in gate):gate_success=True;chosen=factor;accepted[arm]=heads(policies[arm]);break
                if not gate_success:set_heads(policies[arm],accepted[arm])
                accepted_path=candidate(OUT/'candidates'/f'{arm}_r{round_index}_accepted.npz',policies[arm]);after=[]
                for t in active:
                    rr=eval_tasks([t],accepted_path,f'{arm}_r{round_index}_post_{t["id"]}',arm,retain=not safe(parent_rows[t['id']]))[0];after.append(rr)
                    if safe(rr) and not t['original'] and all(t['id']!=v['id'] for v in pool):pool.append(t)
                    if safe(rr) and not safe(parent_rows[t['id']]):append('new_capability.jsonl',{'round':round_index,'arm':arm,'task':t['id'],'parent':parent_rows[t['id']],'candidate':rr})
                    if t['original'] and safe(rr):selected={'arm':arm,'round':round_index,'heads':str(accepted_path),'basis':'originalLEFT safeGOAL with current seven-task gate'}
                verify_compact(collection[arm]);metric.update(chosen_interpolation=chosen,seven_gate_passed=gate_success,postupdate_safe_GOAL=sum(safe(r) for r in after));round_metrics.append(metric)
            newbest=min(float(np.linalg.norm(np.asarray(t['pose'][:2])-original_xy)) for t in pool);advance=pool_best-newbest
            stalled=0 if advance>=.025 else stalled+1;pool_best=min(pool_best,newbest);stats['rounds_completed']=round_index+1
            post_successes=sum(m['postupdate_safe_GOAL'] for m in round_metrics)
            expansion_streak=expansion_streak+1 if post_successes>=3 and all(m['seven_gate_passed'] for m in round_metrics) else 0
            if expansion_streak>=plan['advance_streak_for_expansion']:
                radius_index=min(radius_index+1,len(plan['course_radii_m'])-1);expansion_streak=0
            elif post_successes==0:radius_index=max(0,radius_index-1)
            checkpoint={'heads':{a:heads(p) for a,p in policies.items()},'stats':stats.copy(),'pool':pool,'radius_index':radius_index,'expansion_streak':expansion_streak,'rng':rng.bit_generator.state,'arm_rng':{a:r.bit_generator.state for a,r in arm_rng.items()},'round':round_index,'protocol_sha256':C.sha(HERE/'protocol.json'),'complete_round_boundary':True,'optimizer_state':'Adam reset each round by protocol','exact_live_environment_resume':False}
            partial=OUT/'checkpoint_partial.pt';torch.save(checkpoint,partial);previous=OUT/'checkpoint_previous.pt';latest=OUT/'checkpoint_latest.pt'
            if previous.exists():append('checkpoint_cleanup.jsonl',{'path':str(previous),'sha256':C.sha(previous),'bytes':previous.stat().st_size,'reason':'Owned rolling predecessor, full behavior heads and raw evidence retained'});previous.unlink()
            if latest.exists():os.replace(latest,previous)
            os.replace(partial,latest)
            summary={'round':round_index,'metrics':round_metrics,'successful_reset_pool':len(pool),'best_distance_to_original_reset_m':pool_best,'advance_m':advance,'stalled_rounds':stalled,'next_radius_m':plan['course_radii_m'][radius_index],**stats,'total_physics':total()};append('rounds.jsonl',summary);print(json.dumps(summary),flush=True);status('ROUND_COMPLETE',summary)
            if selected:reason='ORIGINAL_LEFT_GOAL';break
            if stalled>=plan['stall_rounds']:reason='CURRICULUM_STALLED';break
        if reason is None:reason='ROUND_BUDGET_COMPLETE'
    except BudgetStop as error:reason=str(error)
    except BaseException:
        reason='FAILED';write('failure.json',{'state':'FAILED','stats':stats,'unrecorded_inflight_physics_upper_bound':inflight,'traceback':traceback.format_exc()});raise
    finally:
        # Final export always uses the last accepted, seven-task-checked parameters.
        finalizing=True
        exports={}
        for arm,p in policies.items():
            set_heads(p,accepted[arm]);path=OUT/'exports'/f'{arm}_final.pt';torch.save({'policy':{k:v.detach().cpu() for k,v in p.state_dict().items()},'arm':arm,'protocol_sha256':C.sha(HERE/'protocol.json'),'parent_sha256':C.sha(plan['checkpoint']),'sigma_native':SIGMA,'deterministic_architecture':'OriginalPolicy113to2','training':'critic-free head-only clipped episodic policy gradient','automatic_promotion':False},path);exports[arm]={'path':str(path),'sha256':C.sha(path)}
        # Natural final old8 evaluation is required when sufficient bounded resources remain.
        final=[];reload_rows=[];robust_rows=[];robust_tasks=[];invalid_robust_tasks=[];stable_5m=False
        if policies and reason!='FAILED' and time.time()+120<deadline:
            try:
                for arm,p in policies.items():
                    path=candidate(OUT/'candidates'/f'{arm}_final.npz',p);final+=eval_tasks(primary,path,f'{arm}_final',arm,retain=True)
                if selected:
                    arm=selected['arm'];saved=Policy().eval()
                    saved.load_state_dict(torch.load(exports[arm]['path'],map_location='cpu',weights_only=False)['policy'])
                    assert all(torch.equal(v.cpu(),saved.state_dict()[k]) for k,v in policies[arm].state_dict().items())
                    assert all(torch.equal(saved.state_dict()[k],v) for k,v in immutable.items())
                    path=candidate(OUT/'candidates'/f'{arm}_saved_reload.npz',saved)
                    reload_rows=eval_tasks(primary,path,f'{arm}_saved_reload',arm,retain=True);final+=reload_rows
                    for side_index in [6,7]:
                        origin=primary[side_index]
                        for dx in plan['robustness_delta_xy_m']:
                            for dy in plan['robustness_delta_xy_m']:
                                for deg in plan['robustness_delta_yaw_deg']:
                                    i=len(robust_tasks)+len(invalid_robust_tasks)
                                    spec={'id':f'stable5m_{origin["case"]["side"]}_{i}','case':origin['case'],'near_snapshot':origin['snapshot'],'delta':[dx,dy,deg]};t=tasks(spec)
                                    if t['valid']:robust_tasks.append(t)
                                    else:invalid_robust_tasks.append(t)
                    robust_rows=eval_tasks(robust_tasks,path,f'{arm}_stable5m',arm,retain=True);final+=robust_rows
                    stable_5m=len(reload_rows)==8 and all(safe(r) for r in reload_rows) and len(robust_rows)==54 and not invalid_robust_tasks and all(safe(r) for r in robust_rows)
            except BudgetStop as e:append('final_evaluation_incomplete.jsonl',{'reason':str(e),'completed':len(final)})
        for pipe in pipes:
            try:pipe.send(None)
            except OSError:pass
        for w in workers:
            w.join(3)
            if w.is_alive():w.terminate();w.join(3)
        if reason!='FAILED':
            for path,h in plan['input_hashes'].items():assert C.sha(path)==h,path
            result={'state':'COMPLETE','stop_reason':reason,'stats':stats,'total_physics':total(),'selected':selected,'exports':exports,'final_evaluation':final,'original_LEFT_passed':any(r['task']==primary[6]['id'] and safe(r) for r in final) if primary else False,'stable_5m_gate_passed':stable_5m,'saved_reload_old8_safe_GOAL':sum(safe(r) for r in reload_rows),'robustness_completed':len(robust_rows),'robustness_safe_GOAL':sum(safe(r) for r in robust_rows),'invalid_robust_tasks':invalid_robust_tasks,'automatic_promotion':False,'incumbent':'T159','research_parent':'T194','own_bytes':own(),'all_new_runs_bytes':allused(),'elapsed_s':time.time()-start,'all_workers_stopped':True,'scope':plan['controlled_difference']}
            assert own()<plan['max_bytes'] and allused()<1024**3 and total()<=plan['max_total_physics'];write('summary.json',result);write('status.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
