"""One bounded full-parameter native-action PPO / terminal-progress pilot."""
from support import *
import fcntl, shutil, multiprocessing as mp, copy
class BudgetStop(Exception): pass

def main():
    lock=(OUT/'train.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'status.json').exists(), 'No duplicate T210'
    plan=json.loads((HERE/'protocol.json').read_text())
    assert json.loads((OUT/'contracts.json').read_text())['protocol_sha256']==C.sha(HERE/'protocol.json')
    for source,h in plan['input_hashes'].items(): assert C.sha(source)==h,source
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(plan['seed'])
    start=time.monotonic(); deadline=start+plan['wall_seconds']; wall_deadline=time.time()+plan['wall_seconds']
    episodes=physics=training_episodes=training_physics=0; round_index=-1; finalizing=False
    pipes=[]; workers=[]; records=[]; metrics=[]; final_rows=[]; grid_rows=[]; reload_rows=[]; invalid=[]
    updates={a:0 for a in plan['arms']}; critic_updates={a:0 for a in plan['arms']}
    parent=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['policy']
    policies={a:policy(parent) for a in plan['arms']}
    torch.manual_seed(plan['seed']); initial_v=Value().state_dict()
    values={a:Value() for a in plan['arms']}
    for v in values.values(): v.load_state_dict(initial_v)
    optimizers={a:torch.optim.Adam([q for q in policies[a].parameters() if q.requires_grad],lr=plan['actor_lr']) for a in plan['arms']}
    voptimizers={a:torch.optim.Adam(values[a].parameters(),lr=plan['value_lr']) for a in plan['arms']}
    current_paths={a:Path(plan['checkpoint']) for a in plan['arms']}
    with np.load(plan['retention_targets']) as z:
        anchor_x=torch.tensor(z['states']); anchor_y=torch.tensor(z['actions']); offsets=z['offsets'].copy()
    (OUT/'models').mkdir(); reason=None; stable=False; selected=None
    def append(name,row):
        with (OUT/name).open('a') as f: f.write(json.dumps(row)+'\n')
    def safe(row): return row['complete'] and row['success'] and not row['collision']
    def guard(count=0,steps=0,pending=None):
        if time.monotonic()+(0 if finalizing else plan['final_reserve_seconds'])>=deadline: raise BudgetStop('WALL_FINAL_RESERVE')
        if episodes+count>plan['max_episodes'] or physics+steps>plan['max_physics']: raise BudgetStop('COUNT_RESERVE')
        reserve=plan['pending_bytes'] if pending is None else pending
        if used(OUT)+reserve>=plan['final_bytes'] or used(ROOT/'stage/runs')+reserve>=1024**3 or shutil.disk_usage(ROOT).free<=40*1024**3:
            raise BudgetStop('STORAGE_RESERVE')
    def status(state):
        C.save(OUT/'status.json',{'state':state,'round':round_index,'physics_steps':physics,'episodes':episodes,
            'training_physics_steps':training_physics,'training_episodes':training_episodes,'actor_optimizer_updates':updates,
            'value_optimizer_updates':critic_updates,'elapsed_s':time.monotonic()-start,'own_bytes':used(OUT),'all_new_runs_bytes':used(ROOT/'stage/runs')})
    def receive(pipe):
        while not pipe.poll(.5):
            if time.monotonic()>deadline+2: raise BudgetStop('WORKER_RESPONSE_DEADLINE')
        row=pipe.recv(); assert 'error' not in row,row
        return row
    def job(case_index,path,label,arm,role='evaluation',seed=0,snapshot=None,case=None):
        return {'id':label,'arm':arm,'round':round_index,'role':role,'seed':seed,
                'mode':'physical_gaussian' if role=='training' else 'deterministic',
                'weights':str(path),'weights_sha256':C.sha(path),'case':case or plan['primary_cases'][case_index],
                'exact_snapshot':snapshot or plan['task_snapshots'][case_index]}
    def batch(jobs):
        nonlocal episodes,physics,training_physics,training_episodes
        result=[]
        for off in range(0,len(jobs),plan['workers']):
            sub=jobs[off:off+plan['workers']]
            guard(len(sub),sum(j['case']['step_cap'] for j in sub),len(sub)*256*1024+512*1024)
            for pipe,j in zip(pipes,sub): pipe.send(('rollout',j))
            rows=[receive(pipe) for pipe in pipes[:len(sub)]]
            episodes+=len(rows); physics+=sum(r['physics_steps'] for r in rows)
            for row in rows:
                if row['role']=='training': training_episodes+=int(row['complete']); training_physics+=row['physics_steps']
                append('rollouts.jsonl',row)
            records.extend(rows); result.extend(rows); status('RUNNING')
            if not all(r['complete'] for r in rows): raise BudgetStop('WALL_CENSORED_ROLLOUT')
        return result
    def old8(path,arm,label):
        return batch([job(i,path,f'{label}_{i}',arm) for i in range(8)])
    def update(arm,rows):
        p,v=policies[arm],values[arm]; data=[]
        for row in rows:
            assert row['complete'] and C.sha(row['trace'])==row['trace_sha256']
            with np.load(row['trace']) as z:
                rew=training_rewards(z['rewards'],z['native_counts'],row['remaining_m'],plan['terminal_distance_coefficients'][arm])
                returns=monte_carlo(rew,z['native_counts'])
                data.append((z['states'].copy(),z['actions'].copy(),z['logp'].copy(),returns))
        x,a,oldlogp,returns=[torch.as_tensor(np.concatenate([d[i] for d in data]),dtype=torch.float32 if i<2 else torch.float64) for i in range(4)]
        with torch.no_grad():
            before=p.physical_logp(x,a)
            lp_error=float((before-oldlogp).abs().max()); assert lp_error<1e-5
            old_values=v(x).double(); advantage=returns-old_values
            advantage=(advantage-advantage.mean())/advantage.std(unbiased=False).clamp_min(1e-8)
        before_first=p.body.net[0].weight.detach().clone(); before_full={k:q.clone() for k,q in p.state_dict().items()}
        rng=np.random.default_rng(plan['seed']+round_index)  # identical ordering/anchor seeds across arms.
        kl=0.; losses=[]; stopped=False; first_grad=None
        actor_before=updates[arm]; value_before=critic_updates[arm]
        for epoch in range(plan['epochs']):
            for indices in np.array_split(rng.permutation(len(x)),math.ceil(len(x)/plan['minibatch'])):
                guard(pending=6*1024**2)
                b=torch.tensor(indices,dtype=torch.long)
                ratio=(p.physical_logp(x[b],a[b])-oldlogp[b]).exp()
                objective=torch.minimum(ratio*advantage[b],ratio.clamp(1-plan['clip_ratio'],1+plan['clip_ratio'])*advantage[b]).mean()
                retention=[]
                for i in range(7):
                    chosen=rng.integers(offsets[i],offsets[i+1],size=plan['anchor_per_task'])
                    retention.append((p(anchor_x[chosen])-anchor_y[chosen]).square().mean())
                mse=torch.stack(retention).mean(); loss=-objective+plan['retention_weight']*mse
                assert bool(torch.isfinite(loss)); optimizers[arm].zero_grad(); loss.backward()
                if first_grad is None: first_grad=float(p.body.net[0].weight.grad.norm())
                norm=float(torch.nn.utils.clip_grad_norm_([q for q in p.parameters() if q.requires_grad],plan['grad_clip'],error_if_nonfinite=True))
                optimizers[arm].step(); updates[arm]+=1
                with torch.no_grad():
                    logratio=p.physical_logp(x,a)-oldlogp
                    kl=float((logratio.exp()-1-logratio).mean())
                losses.append({'ppo_objective':float(objective),'old7_anchor_MSE':float(mse),'gradient_norm_before_clip':norm,'sampled_approx_KL':kl})
                if kl>plan['target_KL']: stopped=True; break
            if stopped: break
        # Detached fresh Monte Carlo targets; the baseline never sends action gradients.
        for epoch in range(plan['epochs']):
            for indices in np.array_split(rng.permutation(len(x)),math.ceil(len(x)/plan['minibatch'])):
                guard(pending=6*1024**2); b=torch.tensor(indices,dtype=torch.long)
                value_loss=(v(x[b]).double()-returns[b]).square().mean()
                assert bool(torch.isfinite(value_loss)); voptimizers[arm].zero_grad(); value_loss.backward()
                torch.nn.utils.clip_grad_norm_(v.parameters(),plan['grad_clip'],error_if_nonfinite=True)
                voptimizers[arm].step(); critic_updates[arm]+=1
        path=OUT/'models'/f'{arm}_round{round_index}.pt'
        assert not path.exists(); guard(pending=3*1024**2)
        torch.save({'policy':p.state_dict(),'parent_sha256':C.sha(plan['checkpoint']),
                    'protocol_sha256':C.sha(HERE/'protocol.json'),'arm':arm,'round':round_index,
                    'actor_optimizer_updates':updates[arm],'automatic_promotion':False,
                    'origin':'T194 warm start; full-parameter native-action PPO pilot; not from-scratch'},path)
        current_paths[arm]=path
        row={'round':round_index,'arm':arm,'decisions':len(x),'logp_reconstruction_max_error':lp_error,
             'actor_updates_this_round':updates[arm]-actor_before,'value_updates_this_round':critic_updates[arm]-value_before,
             'first_layer_gradient_norm':first_grad,'first_layer_parameter_delta_norm':float((p.body.net[0].weight-before_first).norm()),
             'total_parameter_delta_norm':float(torch.sqrt(sum((q-before_full[k]).square().sum() for k,q in p.state_dict().items()))),
             'sampled_approx_KL':kl,'KL_early_stop':stopped,'value_MC_MSE_after':float((v(x).double()-returns).square().mean().detach()),
             'mean_MC_return':float(returns.mean()),'optimizer_steps':losses,'weights':str(path),'weights_sha256':C.sha(path)}
        append('updates.jsonl',row); metrics.append(row); print(json.dumps({k:w for k,w in row.items() if k!='optimizer_steps'}),flush=True)
    try:
        ctx=mp.get_context('spawn')
        for i in range(plan['workers']):
            a,b=ctx.Pipe(); os.environ['STAGE_PROCESS_CPU']=str(i+1)
            w=ctx.Process(target=worker,args=(b,i,wall_deadline)); w.start(); b.close(); pipes.append(a); workers.append(w)
        os.environ['STAGE_PROCESS_CPU']='0'
        C.save(OUT/'workers.json',{'parent':os.getpid(),'workers':[w.pid for w in workers]}); status('RUNNING')
        baseline=old8(Path(plan['checkpoint']),'parent','parent')
        assert all(safe(baseline[i]) for i in [0,1,2,3,4,5,7]) and baseline[6]['collision']
        for round_index in range(plan['max_rounds']):
            guard(8,8*plan['primary_cases'][6]['step_cap'],10*1024**2)
            collections={}
            for arm in plan['arms']:
                collections[arm]=batch([job(6,current_paths[arm],f'g{round_index}_{arm}_train{i}',arm,'training',
                                             plan['seed']+100*round_index+i) for i in range(plan['training_episodes_per_arm_per_round'])])
            if round_index==0:
                for arm in plan['arms']:
                    row=collections[arm][0]; guard(1,row['physics_steps']); pipes[0].send(('replay',row))
                    check=receive(pipes[0]); physics+=check['physics_steps']; episodes+=1; append('replays.jsonl',check)
            for arm in plan['arms']: update(arm,collections[arm])
            evaluated=[]
            for arm in plan['arms']: evaluated+=old8(current_paths[arm],arm,f'g{round_index}_{arm}_old')
            append('rounds.jsonl',{'round':round_index,'training_safe_GOAL':{a:sum(safe(r) for r in collections[a]) for a in plan['arms']},
                                  'evaluations':evaluated,'actor_updates':dict(updates)})
            status('ROUND_COMPLETE')
            if any(safe(r) and r['case']['id']==plan['primary_cases'][6]['id'] for r in evaluated):
                reason='CENTRAL_ORIGINAL_LEFT_GOAL'; break
        if reason is None: reason='ROUND_BUDGET_COMPLETE'
    except BudgetStop as e: reason=str(e)
    except BaseException:
        reason='FAILED'; C.save(OUT/'failure.json',{'traceback':traceback.format_exc(),'physics_steps':physics,'episodes':episodes})
    finally:
        finalizing=True
        try:
            if reason!='FAILED':
                for arm in plan['arms']:
                    guard(pending=512*1024)
                    path=OUT/'models'/(arm+'_value_final.pt')
                    torch.save({'value':values[arm].state_dict(),'optimizer_updates':critic_updates[arm],
                                'protocol_sha256':C.sha(HERE/'protocol.json'),'optimizer_state_saved':False},path)
                    # Read the serialized full Actor, never an in-memory-only acceptance.
                    loaded=policy(torch.load(current_paths[arm],map_location='cpu',weights_only=False)['policy'])
                    assert all(torch.equal(q,loaded.state_dict()[k]) for k,q in policies[arm].state_dict().items())
                    final_rows+=old8(current_paths[arm],arm,f'final_reload_{arm}')
                for arm in plan['arms']:
                    arm_rows=[r for r in final_rows if r['arm']==arm]
                    if len(arm_rows)==8 and all(safe(r) for r in arm_rows): selected=arm; break
                if selected is not None:
                    reload_rows=old8(current_paths[selected],selected,'selected_repeat')
                    specs=[]
                    for i in (6,7):
                        for dx in plan['robustness_xy']:
                            for dy in plan['robustness_xy']:
                                for deg in plan['robustness_yaw']:
                                    spec={'id':f'grid_{i}_{len(specs)+len(invalid)}','case':plan['primary_cases'][i],
                                          'near_snapshot':plan['task_snapshots'][i],'delta':[dx,dy,deg]}
                                    guard(); pipes[0].send(('make_grid',spec)); g=receive(pipes[0])
                                    if g['valid']: specs.append((i,g))
                                    else: invalid.append(g)
                    grid_rows=batch([job(i,current_paths[selected],f'grid_eval_{j}',selected,snapshot=g['snapshot']) for j,(i,g) in enumerate(specs)])
                    stable=len(reload_rows)==8 and all(safe(r) for r in reload_rows) and len(grid_rows)==54 and not invalid and all(safe(r) for r in grid_rows)
        except BudgetStop as e: append('final_incomplete.jsonl',{'reason':str(e),'old8_completed':len(final_rows),'grid_completed':len(grid_rows)})
        except BaseException:
            reason='FAILED'; C.save(OUT/'failure_final.json',{'traceback':traceback.format_exc(),'physics_steps':physics,'episodes':episodes})
        finally:
            for pipe in pipes:
                try: pipe.send(None)
                except OSError: pass
            for w in workers:
                w.join(3)
                if w.is_alive(): w.terminate(); w.join(3)
        for source,h in plan['input_hashes'].items(): assert C.sha(source)==h,source
        summary={'state':'FAILED' if reason=='FAILED' else 'COMPLETE','stop_reason':reason,'round':round_index,
                 'physics_steps':physics,'episodes':episodes,'training_physics_steps':training_physics,'training_episodes':training_episodes,
                 'actor_optimizer_updates':updates,'value_optimizer_updates':critic_updates,
                 'actor_trainable_parameters':sum(q.numel() for q in policies['original'].parameters() if q.requires_grad),
                 'final_models':{a:{'path':str(path),'sha256':C.sha(path)} for a,path in current_paths.items()},
                 'final_saved_reload_old8':final_rows,'selected_arm':selected,'selected_repeat_old8':reload_rows,
                 'robustness_completed':len(grid_rows),'robustness_safe_GOAL':sum(safe(r) for r in grid_rows),
                 'invalid_grid':invalid,'stable_5m_gate_passed':stable,'all_workers_stopped':True,
                 'automatic_promotion':False,'elapsed_s':time.monotonic()-start,'own_bytes':used(OUT),'all_new_runs_bytes':used(ROOT/'stage/runs')}
        assert physics<=plan['max_physics'] and episodes<=plan['max_episodes'] and used(OUT)<plan['final_bytes'] and used(ROOT/'stage/runs')<1024**3
        C.save(OUT/'summary.json',summary); C.save(OUT/'status.json',summary)
        print(json.dumps({k:v for k,v in summary.items() if k not in ('final_saved_reload_old8','selected_repeat_old8')}),flush=True)
if __name__=='__main__':main()
