"""Bounded paired frozen temporal-exploration diagnostic, never RL training."""
from temporal_support import *
import multiprocessing as mp,fcntl,shutil
class Stop(Exception):pass
def main():
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'status.json').exists(),'No duplicate T220'
    plan=json.loads((HERE/'protocol.json').read_text());contracts=json.loads((OUT/'contracts.json').read_text())
    assert contracts['state']=='PASS' and contracts['protocol_sha256']==C.sha(HERE/'protocol.json')
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    start=time.monotonic();deadline=start+plan['wall_seconds'];wall=time.time()+plan['wall_seconds'];physics=commands=replays=0;rows=[];pipes=[];workers=[];reason=None;first_hashes={}
    torch.set_num_threads(1);state=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['policy'];weights=state_np(S.policy(state));ph=digest(weights)
    def append(name,obj):
        with (OUT/name).open('a') as f:f.write(json.dumps(obj)+'\n')
    def guard(n,steps,pending):
        if time.monotonic()+15>=deadline:raise Stop('WALL_RESERVE')
        if commands+n>plan['max_commands'] or physics+steps>plan['max_physics']:raise Stop('COUNT_RESERVE')
        if used(OUT)+pending>=plan['final_bytes'] or used(ROOT/'stage/runs')+pending>=plan['global_bytes'] or shutil.disk_usage(ROOT).free<40*1024**3:raise Stop('STORAGE_RESERVE')
    def receive(pipe):
        while not pipe.poll(.5):
            if time.monotonic()>deadline:raise Stop('RESPONSE_DEADLINE')
        reply=pipe.recv();assert 'error' not in reply,reply;return reply
    def status(state):
        C.save(OUT/'status.json',dict(state=state,physics_steps=physics,commands=commands,diagnostic_episodes=len(rows),exact_replay_episodes=replays,optimizer_calls=0,elapsed_s=time.monotonic()-start,own_bytes=used(OUT),all_runs_MiB=used(ROOT/'stage/runs')/1024**2))
    try:
        ctx=mp.get_context('spawn')
        for i in range(3):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(i+1);w=ctx.Process(target=worker,args=(b,i,wall));w.start();b.close();pipes.append(a);workers.append(w)
        os.environ['STAGE_PROCESS_CPU']='0';C.save(OUT/'workers.json',{'parent':os.getpid(),'workers':[w.pid for w in workers]})
        ready=[receive(pipe) for pipe in pipes];assert all(r.get('ready') for r in ready);C.save(OUT/'worker_ready.json',ready)
        for seed_index in range(plan['episodes_per_arm']):
            jobs=[]
            for label,rho in plan['arms'].items():
                case=dict(plan['case'],noise_rho=rho)
                jobs.append(dict(id=f'{label}_{seed_index}',arm=label,round=seed_index,role='frozen_diagnostic',seed=plan['seed']+seed_index,
                    mode='frozen_copula_diagnostic',case=case,exact_snapshot=plan['snapshot'],policy_np=weights,policy_hash=ph))
            n=len(jobs);guard(n*(2 if seed_index==0 else 1),n*plan['case']['step_cap']*(2 if seed_index==0 else 1),n*32*1024+32768)
            for pipe,job in zip(pipes,jobs):pipe.send(('rollout',job))
            result=[receive(pipe) for pipe in pipes];commands+=n;physics+=sum(r['row']['physics_steps'] for r in result)
            for item in result:
                row=item['row'];row['noise_rho']=row['case']['noise_rho'];row['logp_is_zero_placeholder_not_behavior_density']=True
                rows.append(row);append('rollouts.jsonl',row)
            assert all(r['row']['complete'] for r in result),'Censored rollout'
            if seed_index==0:
                # Copula starts with the identical innovation in all three arms.
                assert all(np.array_equal(result[0]['arrays']['actions'][0],r['arrays']['actions'][0]) for r in result)
                for pipe,item in zip(pipes,result):pipe.send(('replay',item['row']))
                rebuilt=[receive(pipe) for pipe in pipes];commands+=n;physics+=sum(r['row']['physics_steps'] for r in rebuilt)
                for item,other in zip(result,rebuilt):
                    assert other['row']['verified'] and all(np.array_equal(v,other['arrays'][k]) for k,v in item['arrays'].items())
                    append('replays.jsonl',other['row']);replays+=1
            status('RUNNING')
            if any(r['row']['success'] and not r['row']['collision'] for r in result):reason='SAMPLED_GOAL_PAIRED_BLOCK_STOP';break
        if reason is None:reason='EPISODE_BUDGET_COMPLETE'
    except Stop as e:reason=str(e)
    except BaseException:reason='FAILED';C.save(OUT/'failure.json',{'traceback':traceback.format_exc()})
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except OSError:pass
        for w in workers:
            w.join(3)
            if w.is_alive():w.kill();w.join(3)
        assert all(not w.is_alive() for w in workers)
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    arms={}
    for label in plan['arms']:
        r=[r for r in rows if r['arm']==label];arms[label]=dict(n=len(r),safeGOAL=sum(bool(x['success'] and not x['collision']) for x in r),CONTACT=sum(bool(x['collision']) for x in r),timeout=sum(bool(x['timeout']) for x in r),minimum_remaining_m=min([x['remaining_m'] for x in r],default=None),minimum_nearest_m=min([x['nearest_m'] for x in r],default=None))
    summary=dict(state='FAILED' if reason=='FAILED' else 'COMPLETE',stop_reason=reason,physics_steps=physics,commands=commands,diagnostic_episodes=len(rows),exact_replay_episodes=replays,optimizer_calls=0,arms=arms,
        all_workers_stopped=True,automatic_promotion=False,deterministic_LEFT_passed=False,stable_5m_gate_passed=False,elapsed_s=time.monotonic()-start,own_bytes=used(OUT),all_runs_MiB=used(ROOT/'stage/runs')/1024**2)
    C.save(OUT/'summary.json',summary);C.save(OUT/'status.json',summary);print(json.dumps(summary),flush=True)
    assert commands<=plan['max_commands'] and physics<=plan['max_physics'] and used(OUT)<plan['final_bytes'] and used(ROOT/'stage/runs')<plan['global_bytes']
if __name__=='__main__':main()
