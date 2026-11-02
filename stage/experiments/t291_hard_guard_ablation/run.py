"""One registered, paired, bounded local acquisition experiment; no resumption."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t282_capped_balance'))
from balanced_learning import *
sys.path.insert(0,str(ROOT/'stage/experiments/t287_time_aligned_credit'))
from t287_credit import *
from t291_learning import paired_capability_update
from t291_codec import pack_full,unpack_full
paired_update=paired_capability_update
from t280_worker import worker
sys.path.insert(0,str(ROOT/'stage/experiments/t278_fixed_goal_starts'))
from t278_guard import Budget
import copy,gzip,io,json,multiprocessing as mp,os,subprocess,time,traceback
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t291_hard_guard_ablation'
class Stop(Exception):pass


def main():
    assert not (OUT/'status.json').exists(),'No duplicate/restart/new grid'
    plan=json.loads((HERE/'protocol.json').read_text());start=time.time();deadline=min(start+plan['wall_seconds'],plan['overnight_stop_epoch']);learn_deadline=min(deadline-plan['final_reserve_seconds'],plan['overnight_learning_stop_epoch'])
    meter=None;workers=[];pipes=[];counts=None;training=False;active='';reason='FIXED_BUDGET_COMPLETE';raw=[];compact_count=0;training_count=0;inflight=[];models={};opts={};exports={};updates={k:[] for k in plan['arms']};accepted={k:0 for k in plan['arms']};final={};cache={};cache_hits=0;rounds={k:0 for k in plan['arms']};last_status=0.;optimizer_calls=[];zero_accept_rounds=0
    safe=lambda r:r['complete'] and r['success'] and not r['collision']
    def append(name,r,limit=2560):
        line=json.dumps(r,separators=(',',':'))+'\n';assert len(line.encode())<=limit,(name,len(line.encode()))
        with (OUT/name).open('a') as f:f.write(line)
    def status(state,force=False):
        nonlocal last_status
        now=time.time()
        if not force and now-last_status<20:return
        live=0 if counts is None else sum(counts[:]);last_status=now
        C.save(OUT/'status.json',dict(state=state,pid=os.getpid(),active_arm=active,updated_epoch=now,
            main_epoch=start,learning_deadline_epoch=learn_deadline,absolute_deadline_epoch=deadline,
            physics_steps=live,commands=0 if meter is None else meter.commands,inflight_commands=len(inflight),
            training_episodes=training_count,accepted_updates=accepted,completed_rounds=rounds,gate_by_arm={k:v.get('gate',False) for k,v in final.items()}))
    def guard():
        if time.time()>=(learn_deadline if training else deadline):raise Stop('LEARNING_DEADLINE' if training else 'TOTAL_DEADLINE')
        meter.checkpoint()
    def count_optimizer(opt,arm,phase):
        original=opt.step
        def counted(*args,**kwargs):
            event=dict(arm=arm,phase=phase,invocation=len(optimizer_calls));optimizer_calls.append(event);append('optimizer_calls.jsonl',event)
            return original(*args,**kwargs)
        opt.step=counted
    def receive(pipe):
        while not pipe.poll(.25):
            status('NATIVE_WORKERS_ACTIVE')
            if time.time()>deadline+5:raise Stop('WORKER_RESPONSE_DEADLINE')
        item=pipe.recv();assert 'error' not in item,item;return item
    def jobs_for(jobs,prefix,eta=0.,mode='deterministic'):
        return [dict(j,id=prefix+'_'+j['id'],source_reset_id=j['id'],aux_eta=eta,mode=mode,seed=0) for j in jobs]
    def save_item(item,job,role,full):
        nonlocal compact_count,training_count
        a=item['arrays'];r=item['row'];r.update(arm=active,role=role)
        if full:
            assert training_count<plan['max_training_payloads'];training_count+=1;payload=pack_full(a);limit=plan['training_episode_bytes']
        else:
            assert compact_count<plan['max_compact_payloads'];compact_count+=1
            payload={k:a[k] for k in ['actions','native_counts','end_flags','rewards','environment_rewards','aux_costs']};limit=plan['compact_episode_bytes']
        buf=io.BytesIO();np.savez_compressed(buf,**payload);blob=buf.getvalue();assert len(blob)<=limit,(role,len(blob),limit)
        path=OUT/'episodes'/(job['id']+'.npz');assert not path.exists();meter.charge_payload(len(blob));path.write_bytes(blob)
        r.update(path=str(path),sha256=C.sha(path),saved_array_keys=list(payload),storage_encoding='state-chain-v1' if full else 'compact-six-v1');append('cases.jsonl',r);raw.append(r)
    def roll(jobs,actor,role,replay=False):
        nonlocal inflight
        result=[]
        for off in range(0,len(jobs),4):
            batch=jobs[off:off+4];mult=2 if replay else 1;guard()
            if meter.commands+len(batch)*mult>plan['max_commands'] or meter.physics+sum(j['case']['step_cap'] for j in batch)*mult>plan['max_physics']:raise Stop('PHYSICAL_OR_COMMAND_RESERVE')
            state=S.state_np(actor);ph=S.digest(actor.state_dict());messages=[dict(j,policy_np=state,policy_hash=ph,deadline_epoch=learn_deadline if training else deadline) for j in batch]
            meter.commands+=len(batch);inflight=messages
            for pipe,j in zip(pipes,messages):pipe.send(j)
            items=[receive(pipe) for pipe in pipes[:len(batch)]];meter.physics=sum(counts[:]);inflight=[]
            for item,j in zip(items,batch):save_item(item,j,role,role=='training')
            result.extend(items);status(role,True)
            if any(not i['row']['complete'] or i['row']['error'] for i in items):raise Stop('PERSISTED_ADMINISTRATIVE_PREFIX_OR_NATIVE_ERROR')
            if replay:
                meter.commands+=len(batch);inflight=messages
                for pipe,j in zip(pipes,messages):pipe.send(j)
                again=[receive(pipe) for pipe in pipes[:len(batch)]];meter.physics=sum(counts[:]);inflight=[]
                partial=False
                for a,b,j in zip(items,again,batch):
                    row=b['row'];row.update(arm=active,role=role,original_id=a['row']['id'])
                    if not row['complete'] or row['error']:
                        # The physical prefix is retained under a separately reserved compact slot.
                        jj=dict(j,id=j['id']+'_partial_replay');row['id']=jj['id'];save_item(b,jj,'partial_replay',False)
                        partial=True;continue
                    append('replays.jsonl',row)
                    assert a['row']['array_hashes']==row['array_hashes'] and a['row']['integration_sha256']==row['integration_sha256']
                    assert all(np.array_equal(v,b['arrays'][k]) for k,v in a['arrays'].items())
                if partial:raise Stop('PERSISTED_PARTIAL_REPLAY')
            meter.checkpoint()
        return result
    def native_gate(actor):
        nonlocal cache_hits
        ph=S.digest(actor.state_dict())
        if ph in cache:cache_hits+=1;return cache[ph]
        rr=[i['row'] for i in roll(jobs_for(plan['acceptance_guard_jobs'],'guard'+str(len(cache))),actor,'old20_guard')]
        value=dict(policy_hash=ph,cases=len(rr),safe=sum(safe(r) for r in rr),all_safe=all(safe(r) for r in rr),failed_ids=[r['source_reset_id'] for r in rr if not safe(r)])
        cache[ph]=value;return value
    def export(arm):
        p=models[arm];opt=opts[arm];payload=dict(policy=p.state_dict(),optimizer=opt.state_dict(),protocol_sha256=C.sha(HERE/'protocol.json'),accepted_RL_updates=accepted[arm],arm=arm,source_policy_hash=plan['policy_hash'],memory_on=False)
        buf=io.BytesIO();torch.save(payload,buf);data=gzip.compress(buf.getvalue(),compresslevel=6,mtime=0);assert len(data)<=plan['model_bytes']
        path=OUT/(arm+'_final_checkpoint.pt.gz');assert not path.exists();meter.charge_payload(len(data));path.write_bytes(data)
        loaded=L.read(path);assert S.digest(loaded)==S.digest(payload);q=policy(plan);q.load_state_dict(loaded['policy']);q.eval().requires_grad_(False)
        exports[arm]=dict(path=str(path),sha256=C.sha(path),bytes=len(data),policy_hash=S.digest(q.state_dict()),optimizer_hash=S.digest(loaded['optimizer']));return q
    try:
        for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
        assert json.loads((OUT/'preflight.json').read_text())['state']=='PASS'
        meter=Budget(ROOT,OUT,plan,start_epoch=start);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        frozen=policy(plan).eval().requires_grad_(False);ax,ay,offset=anchors(plan,frozen)
        for arm in plan['arms']:
            models[arm]=policy(plan);opts[arm]=L.optimizer(models[arm],plan);count_optimizer(opts[arm],arm,'live')
        ctx=mp.get_context('spawn');counts=ctx.Array('q',4)
        for index in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(index+1);w=ctx.Process(target=worker,args=(b,index,str(OUT),plan,counts));w.start();b.close();workers.append(w);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(pipe) for pipe in pipes]
        assert all(r['ready'] and r['CPU_affinity']==[i+1] for i,r in enumerate(ready))
        C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[w.pid for w in workers],ready=ready,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
        status('INITIALIZED',True);training=True;first={}
        for rnd in range(plan['rounds']):
            batches={}
            for arm,eta in plan['arms'].items():
                active=arm;guard();jobs=jobs_for(plan['training_jobs'],f'{arm}_r{rnd}',eta,'conditional_AR')
                jobs=[dict(j,id=j['id']+f'_s{k}',seed=plan['seed']+100*rnd+k) for j in jobs for k in range(4)]
                batches[arm]=roll(jobs,models[arm],'training',True)
            if rnd==0:
                actual=batches['time_aligned_credit'];assert len(actual)==16
                physical_keys=['states','actions','means','logp','native_counts','next_states','poses','next_poses','end_flags','environment_rewards','aux_costs','rewards']
                for item,ref in zip(actual,plan['historical_first16']):
                    assert C.sha(ref['path'])==ref['sha256']
                    with np.load(ref['path']) as z:assert all(np.array_equal(item['arrays'][key],z[key]) for key in physical_keys)
                    assert item['row']['integration_sha256']==ref['integration_sha256']
                screen=reward_screen(actual);C.save(OUT/'first_batch_screen.json',dict(first16_full12_arrays_and_integration_exact_historical_T280=True,historical_control_not_independent_new_arm=True,**screen))
                if not screen['passed']:raise Stop('NO_REGISTERED_FAILURE_SORTING_SIGNAL')
            for arm in plan['arms']:
                active=arm;p=models[arm];opt=opts[arm];data=L.training_batch(batches[arm],p)
                append('density.jsonl',dict(arm=arm,round=rnd,**data['density']))
                shadow=copy.deepcopy(p);shadow_opt=L.optimizer(shadow,plan);shadow_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
                count_optimizer(shadow_opt,arm,'shadow')
                live,audit=paired_update(p,opt,shadow,shadow_opt,data,ax,ay,offset,plan,rnd,guard,native_gate,
                    lambda phase,event:append('proposal_events.jsonl',dict(arm=arm,round=rnd,phase=phase,**event),32768))
                assert live==audit and S.digest(p.state_dict())==S.digest(shadow.state_dict()) and S.digest(opt.state_dict())==S.digest(shadow_opt.state_dict())
                n=sum(r['accepted'] for r in live);accepted[arm]+=n;rounds[arm]+=1;updates[arm].append(live)
                append('round_updates.jsonl',dict(arm=arm,round=rnd,updates=live,policy_hash=S.digest(p.state_dict()),optimizer_hash=S.digest(opt.state_dict())),32768)
                rr=[i['row'] for i in roll(jobs_for(plan['old_centers']+plan['centers']+plan['local_centers'],f'{arm}_centers{rnd}'),p,'round_centers')]
                if not all(safe(r) for r in rr[:8]):raise Stop('OLD_CENTER_LOSS')
                status('RL_ROUND_COMPLETE',True)
                zero_accept_rounds=zero_accept_rounds+1 if n==0 else 0
                if zero_accept_rounds>=2:raise Stop('TWO_CONSECUTIVE_ZERO_ACCEPT_ROUNDS')
    except Stop as e:reason=str(e)
    except BaseException:
        reason='FAILED';C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),inflight_jobs=[j['id'] for j in inflight],inflight_physics_upper_bound=sum(j['case']['step_cap'] for j in inflight)))
    training=False
    if meter is not None and models:
        for arm in models:
            try:export(arm)
            except BaseException:C.save(OUT/(arm+'_export_failure.json'),dict(traceback=traceback.format_exc()))
    if meter is not None and len(exports)==len(plan['arms']) and not inflight:
        try:
            for arm in plan['arms']:
                active=arm;q=policy(plan);q.load_state_dict(L.read(exports[arm]['path'])['policy']);q.eval().requires_grad_(False)
                old=roll(jobs_for(plan['old_development']+plan['old_regression64'],arm+'_old'),q,'old_regression')
                dev=roll(jobs_for(plan['development_jobs'],arm+'_dev'),q,'development');gate=False
                if len(old)==275 and all(safe(i['row']) for i in old) and len(dev)==54 and all(safe(i['row']) for i in dev):
                    held=roll(jobs_for(plan['heldout_jobs'],arm+'_heldout'),q,'heldout')
                    if all(safe(i['row']) for i in held):
                        refs={i['row']['source_reset_id']:i['row'] for i in old+dev+held}
                        qq=policy(plan);qq.load_state_dict(L.read(exports[arm]['path'])['policy']);qq.eval().requires_grad_(False)
                        vv=roll(jobs_for(plan['development_jobs']+plan['heldout_jobs']+plan['old_centers'],arm+'_reload'),qq,'independent_reload')
                        assert len(vv)==78
                        for i in vv:
                            r=i['row'];ref=refs[r['source_reset_id']];assert r['array_hashes']==ref['array_hashes'] and r['integration_sha256']==ref['integration_sha256']
                        gate=all(safe(i['row']) for i in vv)
                final[arm]=dict(gate=gate);status('ARM_EVALUATED',True)
        except BaseException:
            if reason!='FAILED':reason='FINAL_EVALUATION_INTERRUPTED'
            C.save(OUT/'final_failure.json',dict(traceback=traceback.format_exc(),inflight_jobs=[j['id'] for j in inflight]))
    for pipe in pipes:
        try:pipe.send(None)
        except (OSError,BrokenPipeError):pass
    for w in workers:
        w.join(3)
        if w.is_alive():w.kill();w.join(3)
    C.save(OUT/'native_guard_cache.json',dict(cache=cache,exact_policy_cache_hits=cache_hits))
    if meter is not None:meter.physics=0 if counts is None else sum(counts[:]);meter.close()
    summaries={}
    for arm in plan['arms']:
        def rows(role):return [r for r in raw if r['arm']==arm and r['role']==role]
        summaries[arm]=dict(accepted_RL_updates=accepted[arm],completed_rounds=rounds[arm],training_n=len(rows('training')),training_safe=sum(safe(r) for r in rows('training')),
            old_n=len(rows('old_regression')),old_safe=sum(safe(r) for r in rows('old_regression')),six_m_n=len(rows('development')),six_m_safe=sum(safe(r) for r in rows('development')),
            heldout_n=len(rows('heldout')),independent_reload_n=len(rows('independent_reload')),gate=final.get(arm,{}).get('gate',False),export=exports.get(arm))
    result=dict(state='FAILED' if reason=='FAILED' else 'COMPLETE',stop_reason=reason,arms=summaries,physics_steps=0 if meter is None else meter.physics,
        commands=0 if meter is None else meter.commands,optimizer_calls_live=sum(r['phase']=='live' for r in optimizer_calls),optimizer_calls_shadow=sum(r['phase']=='shadow' for r in optimizer_calls),
        all_workers_stopped=all(not w.is_alive() for w in workers),inflight_jobs=[j['id'] for j in inflight],elapsed_s=time.time()-start,
        original_safe_T255_unchanged=C.sha(plan['checkpoint'])==plan['checkpoint_sha256'],native20_cache_policies=len(cache),native20_exact_policy_cache_hits=cache_hits,
        own_bytes=S.used(OUT),global_bytes=S.used(ROOT/'stage/runs'),new_runs_budget=plan['global_bytes'],automatic_next_training=False,seven_or_eight_started=False,model_promoted=False,research_only_not_deployment=True,
        limitations='T291 research-only old20 hard-acceptance ablation against consumed historical T287 two-round control. Same safeT255freshAdam/credit/eta.1/AR/PCGrad/soft/trust/first16seeds/two-round max. Native old20 still measured and reported exactly; numeric-pass candidates may be accepted despite old failure, never deployed/promoted. State-chain storage lossless. First16 historical replay not independent new data. Old275 and six54 final all reported; missing/censored separate. Not independent randomized replication or novelty proof, no7/8trainingautomatic')
    C.save(OUT/'summary.json',result);C.save(OUT/'status.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
