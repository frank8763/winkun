"""Bounded RND/control pilot with replay-verified compact data and shadow training."""
from support import *
import copy,fcntl,shutil,multiprocessing as mp
L=C.load(ROOT/'stage/experiments/t216_left_specialist/learner.py','stage219_learner')
class BudgetStop(Exception):pass

def main():
    lock=(OUT/'train.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'status.json').exists(),'No duplicate T219'
    plan=json.loads((HERE/'protocol.json').read_text())
    contracts=json.loads((OUT/'contracts.json').read_text());assert contracts['state']=='PASS' and contracts['all_preflight_workers_exited_cleanly']
    assert contracts['protocol_sha256']==C.sha(HERE/'protocol.json')
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(plan['seed'])
    start=time.monotonic();deadline=start+plan['wall_seconds'];wall_deadline=time.time()+plan['wall_seconds'];finalizing=False
    physics=commands=train_physics=train_episodes=replay_verified=paired_rounds=0;generation=-1;pipes=[];workers=[];rows=[];final_rows=[];exports={};reconstruction=[];reason=None
    actor_updates={a:0 for a in plan['arms']};actor_proposals={a:0 for a in plan['arms']};value_updates={a:0 for a in plan['arms']};rnd_updates={a:0 for a in plan['arms']};shadow_counts={a:{'actor_proposals':0,'actor_accepted':0,'value':0,'RND':0} for a in plan['arms']}
    parent=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['policy']
    policies={a:policy(parent) for a in plan['arms']};shadow_p={a:copy.deepcopy(policies[a]) for a in plan['arms']}
    torch.manual_seed(plan['seed']);v_initial=Value().state_dict();values={a:Value() for a in plan['arms']}
    for v in values.values():v.load_state_dict(v_initial)
    shadow_v={a:copy.deepcopy(values[a]) for a in plan['arms']}
    rnd_state=torch.load(plan['rnd_checkpoint'],map_location='cpu',weights_only=False)['RND'];rnds={}
    for a in plan['arms']:
        r=LidarRND(rnd_state['input_mean'],rnd_state['input_std']);r.load_state_dict(rnd_state);rnds[a]=r
    shadow_r={a:copy.deepcopy(rnds[a]) for a in plan['arms']}
    def adam(p,lr):return torch.optim.Adam([q for q in p.parameters() if q.requires_grad],lr=lr)
    po={a:adam(policies[a],plan['actor_lr']) for a in plan['arms']};spo={a:adam(shadow_p[a],plan['actor_lr']) for a in plan['arms']}
    vo={a:adam(values[a],plan['value_lr']) for a in plan['arms']};svo={a:adam(shadow_v[a],plan['value_lr']) for a in plan['arms']}
    ro={a:adam(rnds[a].predictor,plan['rnd_lr']) for a in plan['arms']};sro={a:adam(shadow_r[a].predictor,plan['rnd_lr']) for a in plan['arms']}
    with np.load(plan['retention_targets']) as z:anchors=torch.tensor(z['states']);targets=torch.tensor(z['actions']);offsets=z['offsets'].copy()
    with torch.no_grad():anchor_mean=policies['control'].physical_mean(anchors)
    def append(name,obj):
        with (OUT/name).open('a') as f:f.write(json.dumps(obj)+'\n')
    def safe(r):return r['complete'] and r['success'] and not r['collision']
    def guard(count=0,steps=0,pending=0,export_reserve=True):
        if time.monotonic()+(0 if finalizing else plan['final_reserve_seconds'])>=deadline:raise BudgetStop('WALL_RESERVE')
        if commands+count>plan['max_commands'] or physics+steps>plan['max_physics']:raise BudgetStop('COUNT_RESERVE')
        reserve=pending+(plan['export_reserve_bytes'] if export_reserve and not finalizing else 0)
        if used(OUT)+reserve>=plan['final_bytes'] or used(ROOT/'stage/runs')+reserve>=1024**3 or shutil.disk_usage(ROOT).free<40*1024**3:raise BudgetStop('STORAGE_RESERVE')
    def status(state):C.save(OUT/'status.json',{'state':state,'round':generation,'physics_steps':physics,'commands':commands,'training_physics_steps':train_physics,'training_episodes':train_episodes,'Actor_updates':actor_updates,'Actor_proposals':actor_proposals,'Value_updates':value_updates,'RND_updates':rnd_updates,'shadow_updates':shadow_counts,'elapsed_s':time.monotonic()-start,'own_bytes':used(OUT),'all_new_runs_MiB':used(ROOT/'stage/runs')/1024**2})
    def receive(pipe):
        while not pipe.poll(.5):
            if time.monotonic()>deadline+1:raise BudgetStop('WORKER_RESPONSE_DEADLINE')
        reply=pipe.recv();assert 'error' not in reply,reply;return reply
    def make_job(i,p,a,label,role):
        state=state_np(p)
        return {'id':label,'arm':a,'round':generation,'role':role,'seed':plan['seed']+100*generation+i if role=='training' else 0,
            'mode':'physical_gaussian' if role=='training' else 'deterministic','case':plan['primary_cases'][6 if role=='training' else i],
            'exact_snapshot':plan['task_snapshots'][6 if role=='training' else i],'policy_np':state,'policy_hash':digest(state)}
    def batch(jobs,replay=False):
        nonlocal physics,commands,train_physics,train_episodes,replay_verified
        result=[]
        for off in range(0,len(jobs),4):
            sub=jobs[off:off+4];guard(len(sub)*(2 if replay else 1),sum(j['case']['step_cap'] for j in sub)*(2 if replay else 1),len(sub)*32*1024+16384)
            for pipe,j in zip(pipes,sub):pipe.send(('rollout',j))
            fresh=[receive(pipe) for pipe in pipes[:len(sub)]]
            physics+=sum(r['row']['physics_steps'] for r in fresh);commands+=len(fresh)
            for item in fresh:
                row=item['row'];rows.append(row);append('rollouts.jsonl',row)
                if row['role']=='training':train_physics+=row['physics_steps'];train_episodes+=int(row['complete'])
            if not all(item['row']['complete'] for item in fresh):raise BudgetStop('WALL_CENSORED_ROLLOUT')
            if replay:
                for pipe,item in zip(pipes,fresh):pipe.send(('replay',item['row']))
                rebuilt=[receive(pipe) for pipe in pipes[:len(sub)]]
                physics+=sum(r['row']['physics_steps'] for r in rebuilt);commands+=len(rebuilt)
                for original,restored in zip(fresh,rebuilt):
                    assert restored['row']['verified'] and all(np.array_equal(v,restored['arrays'][k]) for k,v in original['arrays'].items())
                    replay_verified+=1
                    append('replays.jsonl',restored['row']);result.append({'row':original['row'],'arrays':restored['arrays']})
            else:result.extend(fresh)
            status('RUNNING')
        return result
    def old8(p,a,prefix):return [r['row'] for r in batch([make_job(i,p,a,f'{prefix}_{i}','evaluation') for i in range(8)])]
    def train_one(a,items):
        guard(pending=65536)
        if time.monotonic()+40+plan['final_reserve_seconds']>=deadline:raise BudgetStop('UPDATE_TIME_RESERVE')
        p,v,rnd=policies[a],values[a],rnds[a];fields=[];rewards=[];novel=[]
        before_hash=digest(p.state_dict());before_rnd=digest(rnd.state_dict())
        for item in items:
            d=item['arrays'];row=item['row'];assert row['policy_hash']==before_hash
            xr=torch.tensor(d['next_states'][:,:48]);counts=torch.tensor(d['native_counts'].astype(np.int64));collision=torch.tensor(d['end_flags'][:,0].astype(bool))
            bonus=rnd.bonus(xr,counts,collision,23334,plan['rnd_scale'],plan['rnd_episode_cap'])
            shadow_bonus=shadow_r[a].bonus(xr,counts,collision,23334,plan['rnd_scale'],plan['rnd_episode_cap'])
            assert torch.equal(bonus,shadow_bonus),'RND reward reconstruction mismatch'
            assert float(bonus.sum())<=.05+1e-8 and bool((bonus[collision]==0).all())
            ext=training_rewards(d['rewards'],d['native_counts'],row['remaining_m'],.3);combined=ext+(bonus.numpy() if a=='rnd' else 0.)
            G=monte_carlo(combined,d['native_counts']);fields.append((d['states'],d['actions'],d['means'],d['logp'],G,d['next_states'][:,:48]));rewards.append(combined);novel.append(float(bonus.sum()))
        x,actions,means,lp,G,next_rays=[torch.as_tensor(np.concatenate([d[i] for d in fields]),dtype=torch.float64 if i in (3,4) else torch.float32) for i in range(6)]
        identity=L.verify_behavior(p,x,actions,means,lp,plan)
        for shadow in (False,True):
            pp,vv,rr=(shadow_p[a],shadow_v[a],shadow_r[a]) if shadow else (p,v,rnd)
            oo,vopt,ropt=(spo[a],svo[a],sro[a]) if shadow else (po[a],vo[a],ro[a])
            with torch.no_grad():adv=G-vv(x).double();adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-8)
            result=L.actor_update(pp,oo,x,actions,lp,adv,anchors,targets,offsets,anchor_mean,plan,lambda:None)
            rng=np.random.default_rng(plan['seed']+generation);vc=0
            for epoch in range(plan['value_epochs']):
                for ind in np.array_split(rng.permutation(len(x)),math.ceil(len(x)/plan['minibatch'])):
                    b=torch.tensor(ind);loss=(vv(x[b]).double()-G[b]).square().mean();vopt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(vv.parameters(),.5,error_if_nonfinite=True);vopt.step();vc+=1
            rng=np.random.default_rng(plan['seed']+1000+generation)
            for step in range(plan['rnd_predictor_updates_per_round']):
                b=rng.integers(0,len(next_rays),128);loss=rr.error(next_rays[b]).mean();ropt.zero_grad();loss.backward();assert all(t.grad is None for t in rr.target.parameters());torch.nn.utils.clip_grad_norm_(rr.predictor.parameters(),1.,error_if_nonfinite=True);ropt.step()
            states={'policy':digest(pp.state_dict()),'Actor_optimizer':digest(oo.state_dict()),'Value':digest(vv.state_dict()),'Value_optimizer':digest(vopt.state_dict()),'RND':digest(rr.state_dict()),'RND_optimizer':digest(ropt.state_dict())}
            if not shadow:
                live_states=states;live_result=result;live_vc=vc
                actor_updates[a]+=result['accepted_updates'];actor_proposals[a]+=len(result['attempts']);value_updates[a]+=vc;rnd_updates[a]+=plan['rnd_predictor_updates_per_round']
            else:
                assert states==live_states and result==live_result and vc==live_vc,'Training recipe reconstruction mismatch'
                shadow_counts[a]['actor_proposals']+=len(result['attempts']);shadow_counts[a]['actor_accepted']+=result['accepted_updates'];shadow_counts[a]['value']+=vc;shadow_counts[a]['RND']+=plan['rnd_predictor_updates_per_round']
        record={'round':generation,'arm':a,'before_policy_sha256':before_hash,'before_RND_sha256':before_rnd,'identity':identity,
            'live_and_shadow_states':live_states,'independent_reconstruction_exact':True,'Actor_result':live_result,'Value_updates':live_vc,
            'RND_updates':plan['rnd_predictor_updates_per_round'],'potential_RND_episode_rewards':novel,'applied_RND_rewards':novel if a=='rnd' else [0.]*len(novel),
            'combined_rewards_sha256':digest(rewards),'episodes':[item['row']['id'] for item in items]}
        append('updates.jsonl',record);reconstruction.append(record);status('UPDATED')
        print(json.dumps({k:w for k,w in record.items() if k not in ('Actor_result','live_and_shadow_states')}),flush=True)
    try:
        ctx=mp.get_context('spawn')
        for i in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(i+1);w=ctx.Process(target=worker,args=(b,i,wall_deadline));w.start();b.close();pipes.append(a);workers.append(w)
        os.environ['STAGE_PROCESS_CPU']='0';C.save(OUT/'workers.json',{'parent':os.getpid(),'workers':[w.pid for w in workers]})
        ready=[receive(pipe) for pipe in pipes];assert all(r.get('ready') for r in ready)
        C.save(OUT/'worker_ready.json',ready)
        baseline=old8(policies['control'],'parent','parent');assert all(safe(baseline[i]) for i in (0,1,2,3,4,5,7)) and baseline[6]['collision']
        for generation in range(plan['max_rounds']):
            collections={}
            for a in plan['arms']:collections[a]=batch([make_job(i,policies[a],a,f'g{generation}_{a}_{i}','training') for i in range(4)],True)
            if generation==0:
                assert all(c['row']['array_hashes']==r['row']['array_hashes'] for c,r in zip(collections['control'],collections['rnd']))
                C.save(OUT/'matched_initial.json',{'all_10_arrays_equal':True,'four_paired_fresh_episodes':True})
            for a in plan['arms']:train_one(a,collections[a])
            paired_rounds+=1
            if any(safe(item['row']) for group in collections.values() for item in group):reason='SAMPLED_GOAL_STOP_AFTER_PAIRED_ROUND';break
        if reason is None:reason='ROUND_BUDGET_COMPLETE'
    except BudgetStop as e:reason=str(e)
    except BaseException:reason='FAILED';C.save(OUT/'failure.json',{'traceback':traceback.format_exc(),'physics_steps':physics,'commands':commands})
    finally:
        finalizing=True
        try:
            for a in plan['arms']:
                exports[a]={'Actor':save_gzip(OUT/f'{a}_Actor.pt.gz',{'policy':policies[a].state_dict(),'protocol_sha256':C.sha(HERE/'protocol.json'),'origin':'T194 warm-start compact paired RND pilot','automatic_promotion':False},lambda n:guard(pending=n,export_reserve=False)),
                    'RND':save_gzip(OUT/f'{a}_RND.pt.gz',{'RND':rnds[a].state_dict(),'protocol_sha256':C.sha(HERE/'protocol.json')},lambda n:guard(pending=n,export_reserve=False))}
            if reason!='FAILED':
                for a in plan['arms']:
                    saved=torch.load(io.BytesIO(gzip.decompress(Path(exports[a]['Actor']['path']).read_bytes())),map_location='cpu',weights_only=False)['policy']
                    assert digest(saved)==digest(policies[a].state_dict())
                    reloaded=policy(saved).eval().requires_grad_(False);final_rows+=old8(reloaded,a,f'final_reload_{a}')
        except BudgetStop as e:append('final_incomplete.jsonl',{'reason':str(e),'saved_reload_cases':len(final_rows)})
        except BaseException:reason='FAILED';C.save(OUT/'failure_final.json',{'traceback':traceback.format_exc()})
        for pipe in pipes:
            try:pipe.send(None)
            except OSError:pass
        for w in workers:
            w.join(3)
            if w.is_alive():w.terminate();w.join(3)
            if w.is_alive():w.kill();w.join(3)
        assert all(not w.is_alive() for w in workers)
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    summary={'state':'FAILED' if reason=='FAILED' else 'COMPLETE','stop_reason':reason,'physics_steps':physics,'commands':commands,
        'training_physics_steps':train_physics,'training_episodes':train_episodes,'Actor_updates':actor_updates,'Actor_proposals':actor_proposals,'Value_updates':value_updates,'RND_updates':rnd_updates,
        'shadow_updates':shadow_counts,'training_batches_exactly_reconstructed':len(reconstruction),'compact_replay_episodes':replay_verified,'paired_rounds_completed':paired_rounds,'reconstruction_valid':reason!='FAILED',
        'final_saved_reload_old8':final_rows,'exports':exports,'all_workers_stopped':True,'automatic_promotion':False,'stable_5m_gate_passed':False,
        'elapsed_s':time.monotonic()-start,'own_bytes':used(OUT),'all_new_runs_MiB':used(ROOT/'stage/runs')/1024**2,
        'restore_contract':'Rebuild every observation from compact actions/exact native snapshots; verified all10arrays beforelearning. Re-run recorded deterministic Actor/Value/RND optimizerrecipe frominitialhashedstates; everybatchshadow reproducedparameterANDoptimizerhashes exactly. Version/environmentlocked; intermediatepolicyfilesnotexported. Noexistingdataremoved.'}
    assert physics<=plan['max_physics'] and commands<=plan['max_commands'] and used(OUT)<plan['final_bytes'] and used(ROOT/'stage/runs')<1024**3
    C.save(OUT/'summary.json',summary);C.save(OUT/'status.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='final_saved_reload_old8'}),flush=True)
if __name__=='__main__':main()
