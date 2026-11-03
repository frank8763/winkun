from ppo_learning import *
from ppo_worker import worker
import argparse,contextlib,multiprocessing as mp,signal,subprocess

def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    assert S.used(ROOT/'stage/runs')<=p['global_bytes'] and S.used(OUT)<=p['final_bytes']
    assert shutil.disk_usage(ROOT).free>=p['min_free_bytes']

def preflight():
    assert not any((OUT/f).exists() for f in ['preflight.json','launch.json','status.json'])
    p=read(HERE/'protocol.json');verify(p);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    # Real observations with explicitly SYNTHETIC short finite-task rewards.
    old=read(ROOT/'stage/runs/t309_sequence_goal_value/cases.json');items=[]
    for family in p['goal_families']:
        for draw in [0,1]:
            r=next(r for r in old if r['source_reset_id']==family and r['arm']=='center' and r['seed']==p['preflight_reference_seeds'][family][draw]) if 'source_reset_id' in old[0] else next(r for r in old if r['original_job']['source_reset_id']==family and r['arm']=='center' and r['seed']==p['preflight_reference_seeds'][family][draw])
            a=B.arrays(r['path']);a={k:v[:3].copy() for k,v in a.items() if k in B.NATIVE_KEYS}
            a['next_states'][:-1]=a['states'][1:];a['end_flags'][:]=0;a['end_flags'][-1]=[0,1,0];a['rewards'][:]=[0.,0.,1.+.1*draw]
            items.append(dict(row=dict(id=family+str(draw),goal_family=family,draw=draw),arrays=a,returns=S.monte_carlo(a['rewards'],a['native_counts'])))
    q=policy(p);shadow=copy.deepcopy(q);opt=T.L.optimizer(q,p);so=T.L.optimizer(shadow,p);before=S.digest(q.state_dict());ob=S.digest(opt.state_dict());calls=[]
    small=dict(p,value_steps=1,actor_steps=1);vals,models,opts,vr=fit_values(small,items,0,lambda:None,lambda k,e:calls.append((k,e)))
    vals2,models2,opts2,vr2=fit_values(small,items,0,lambda:None,lambda k,e:calls.append((k,e)))
    assert vr==vr2 and all(np.array_equal(a,b) for a,b in zip(vals,vals2))
    target,metrics=build(p,items,vals);x=torch.tensor(np.concatenate([i['arrays']['states'] for i in items]));offset=np.linspace(0,len(x),9,dtype=int).tolist()
    with torch.no_grad():y=q.step(x)[0]
    a=actor_steps(small,q,opt,items,target,'mc',0,x,y,offset,lambda:None,lambda k,e:calls.append((k,e)))
    b=actor_steps(small,shadow,so,items,target,'mc',0,x,y,offset,lambda:None,lambda k,e:calls.append((k,e)))
    assert a==b and S.digest(q.state_dict())==S.digest(shadow.state_dict()) and S.digest(opt.state_dict())==S.digest(so.state_dict())
    # Shared rollback path is exact for both policy AND adaptive moments.
    from_state=policy(p).state_dict();empty=T.L.optimizer(policy(p),p).state_dict()
    for actor,optimizer in [(q,opt),(shadow,so)]:actor.load_state_dict(from_state);optimizer.load_state_dict(empty)
    assert S.digest(q.state_dict())==S.digest(shadow.state_dict())==before and S.digest(opt.state_dict())==S.digest(so.state_dict())==ob
    param_bytes=sum(v.numel()*v.element_size() for v in q.state_dict().values());vbytes=sum(v.numel()*v.element_size() for v in models[0].state_dict().values())
    assert param_bytes*3+vbytes*6+65536<=p['actor_model_bytes']
    assert 229504+16*256<=p['training_episode_bytes']
    assert 234*20+4*256<=p['old_compact_bytes'] and 280*20+4*256<=p['new_compact_bytes']
    assert len(calls)==6
    C.save(OUT/'preflight.json',dict(state='PASS',native_commands=0,physics_steps=0,Actor_navigation_updates=0,disposable_synthetic_Adam_calls=6,
        exact_Actor_Adam_shadow_and_rollback=True,whole_trajectory_crossfit_disjoint=True,canonical_real_inputs_used_with_synthetic_rewards=True,
        serialized_final_model_upper_bound=param_bytes*3+vbytes*6+65536,full_worstcase_reserved_bytes=S.used(ROOT/'stage/runs')-S.used(OUT)+p['final_bytes'],protocol_sha256=C.sha(HERE/'protocol.json')))
    print('T312 preflight PASS;6 disposable SYNTHETIC Adam calls,0native/navigation update')

def main():
    assert not (OUT/'status.json').exists();p=read(HERE/'protocol.json');start=time.time();deadline=start+p['wall_seconds'];training_deadline=deadline-p['final_reserve_seconds']
    meter=None;workers=[];pipes=[];raw=[];replays=[];geometries=[];calls=[];updates=[];exports={};state='FAILED';stop='STARTING';phase='STARTING';active=[];last_status=0
    ctx=mp.get_context('spawn');counts=ctx.Array('q',4);actors={};opts={};vlast={};fitted={};screens={};gram_records=[];native_cache={}
    safe=lambda r:r['complete'] and r['success'] and not r['collision'] and not r.get('error')
    def status(force=False):
        nonlocal last_status
        if not force and time.time()-last_status<10:return
        last_status=time.time();C.save(OUT/'status.json',dict(state=phase,pid=os.getpid(),main_epoch=start,updated_epoch=last_status,absolute_deadline_epoch=deadline,training_deadline_epoch=training_deadline,
            commands=0 if meter is None else meter.commands,physics_steps=sum(counts[:]),training_episodes=sum(r['role']=='training' for r in raw),exact_replays=len(replays),optimizer_calls=len(calls),round_blocks=len(updates),inflight=[j['id'] for j in active]))
    def guard():
        assert time.time()<training_deadline;meter.hold();status()
    def event(arm,replica,kind,e):
        row=dict(arm=arm,replica=replica,kind=kind,**e);calls.append(row)
        with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps(row,separators=(',',':'))+'\n')
    def receive(pipe):
        while not pipe.poll(.25):status();assert time.time()<deadline-10
        r=pipe.recv();assert 'error' not in r,r;return r
    def roll(jobs,q,role,arm,rnd=None,replay=False,geometry=False):
        nonlocal phase,active
        phase=role;result=[]
        for off in range(0,len(jobs),4):
            group=[]
            for j in jobs[off:off+4]:
                jj=dict(j,role=role,arm=arm,round=rnd,policy_hash=S.digest(q.state_dict()) if q is not None else p['policy_hash'],deadline_epoch=training_deadline if role=='training' else deadline-20)
                if geometry:jj['geometry_only']=True
                elif role!='training':jj.update(mode='deterministic',seed=0,aux_eta=0.)
                if q is not None:jj['policy_np']=S.state_np(q)
                group.append(jj)
            for again in ([False,True] if replay else [False]):
                meter.checkpoint();assert meter.commands+len(group)<=p['max_commands'];assert sum(counts[:])+(0 if geometry else sum(j['case']['step_cap'] for j in group))<=p['max_physics']
                meter.commands+=len(group);active=group;status(True)
                for pipe,j in zip(pipes,group):pipe.send(j)
                batch=[receive(pipe) for pipe in pipes[:len(group)]];meter.physics=sum(counts[:]);active=[]
                if geometry:
                    geometries.extend(batch);C.save(OUT/'geometry.json',geometries);assert all(r['valid'] and r['physics_steps']==0 for r in batch);continue
                for item in batch:
                    r=item['row'];a=item['arrays'];r.pop('original_job',None);r.pop('forced_first_action_diagnostic',None)
                    # policy arrays are an IPC input, never duplicated in JSON.
                    if 'job' in r:r['job'].pop('policy_np',None)
                    complete=r['complete'] and not r.get('error')
                    if again and complete:
                        ref=next(i for i in result if i['row']['id']==r['id']);assert r['array_hashes']==ref['row']['array_hashes'] and r['integration_sha256']==ref['row']['integration_sha256'] and all(np.array_equal(v,ref['arrays'][k]) for k,v in a.items())
                        r['original_id']=r['id'];replays.append(r);C.save(OUT/'replays.json',replays)
                    else:
                        data=B.packed(a) if role=='training' else compact(a);buf=io.BytesIO();np.savez_compressed(buf,**data);blob=buf.getvalue()
                        limit=p['training_episode_bytes'] if role=='training' else (p['old_compact_bytes'] if r['job']['case']['step_cap']<=23334 else p['new_compact_bytes'])
                        assert len(blob)<=limit,(r['id'],len(blob),limit);meter.charge_payload(len(blob));f=OUT/'episodes'/(r['id']+('_partial_replay' if again else '')+'.npz');assert not f.exists();f.write_bytes(blob)
                        r.update(path=str(f),sha256=C.sha(f),saved_array_keys=list(data));raw.append(r)
                        with (OUT/'cases.jsonl').open('a') as f:f.write(json.dumps(r,separators=(',',':'))+'\n')
                        if not again:result.append(item)
                    assert complete,'Partial/invalid native evidence preserved, no learning/continuation'
                status(True)
        return result
    def deterministic(jobs,prefix):return [dict(j,id=prefix+'_'+j['id'],source_reset_id=j['id']) for j in jobs]
    def native_guard(q,arm,rnd):
        ph=S.digest(q.state_dict())
        if ph in native_cache:return dict(native_cache[ph],reused_same_policy_hash=True)
        rr=[i['row'] for i in roll(deterministic(p['acceptance_guard_jobs'],f'{arm}_r{rnd}_guard'),q,'old_guard',arm,rnd)]
        answer=dict(policy_hash=ph,cases=len(rr),safe=sum(safe(r) for r in rr),all_safe=len(rr)==20 and all(safe(r) for r in rr),failed_ids=[r['source_reset_id'] for r in rr if not safe(r)])
        native_cache[ph]=answer;return answer
    def alarm(*_):raise TimeoutError('T312 absolute bounded pilot deadline')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(p['wall_seconds']-5)
    try:
        verify(p);assert read(OUT/'preflight.json')['state']=='PASS';torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        meter=Budget(ROOT,OUT,p,start_epoch=start)
        actors={arm:policy(p) for arm in ['mc','gae']};opts={arm:T.L.optimizer(q,p) for arm,q in actors.items()}
        parent=copy.deepcopy(actors['mc']).eval().requires_grad_(False);ax,ay,offset=T.anchors(p,parent)
        for i in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(i+1);w=ctx.Process(target=worker,args=(b,i,p,counts));w.start();b.close();pipes.append(a);workers.append(w)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(pipe) for pipe in pipes];assert all(r['ready'] and r['CPU_affinity']==[i+1] for i,r in enumerate(ready))
        C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[w.pid for w in workers],ready=ready));status(True)
        roll(p['geometry_jobs'],None,'geometry','shared',geometry=True)
        for rnd in range(p['rounds']):
            round_items={}
            for arm in ['mc','gae']:
                jobs=[]
                for k,base in enumerate(p['training_tasks']):
                    for draw in [0,1]:
                        j=dict(base,id=f'{arm}_r{rnd}_g{k}_draw{draw}',family=f'r{rnd}_g{k}_draw{draw}',goal_family=base['source_reset_id'],draw=draw,side=base['case']['side'],split='train',arm=arm,seed=p['seed']+10000*rnd+100*k+draw,mode='iid',first_steer_innovation_offset=0.,aux_eta=0.)
                        jobs.append(j)
                round_items[arm]=roll(jobs,actors[arm],'training',arm,rnd,replay=True)
            if rnd==0:
                assert all(a['row']['array_hashes']==b['row']['array_hashes'] and a['row']['integration_sha256']==b['row']['integration_sha256'] for a,b in zip(round_items['mc'],round_items['gae']))
                C.save(OUT/'first_batch_overlap.json',dict(exact_same16_trajectories=True,not32_independent_samples=True,physical_work_charged_twice=True))
            for arm in ['mc','gae']:
                guard();phase='VALUE_AND_ADVANTAGE_'+arm;items=round_items[arm]
                for i in items:i['returns']=S.monte_carlo(i['arrays']['rewards'],i['arrays']['native_counts'])
                values,vm,vo,vr=fit_values(p,items,rnd,guard,lambda k,e:event(arm,'live',k,e))
                vv,sm,so,sr=fit_values(p,items,rnd,guard,lambda k,e:event(arm,'shadow',k,e))
                assert vr==sr and all(np.array_equal(a,b) for a,b in zip(values,vv));vlast[arm]=(vm,vo);target,metrics=build(p,items,values)
                screen,gram=gradient_screen(p,actors[arm],items,target,guard);screen.update(value=metrics,value_fits=vr);screens[f'{arm}_{rnd}']=screen
                buf=io.BytesIO();np.savez_compressed(buf,gram=gram,**{f'value{k}':v for k,v in enumerate(values)});blob=buf.getvalue();assert len(blob)<=128*1024;meter.charge_payload(len(blob));(OUT/'learning'/f'{arm}_{rnd}.npz').write_bytes(blob)
                C.save(OUT/'screens.json',screens);fitted[arm]=(items,target)
            if not all(screens[f'{arm}_{rnd}']['value']['heldout_value_gate'] and screens[f'{arm}_{rnd}']['variance_gate'] for arm in ['mc','gae']):stop='CROSS_FIT_VALUE_OR_LOCAL_VARIANCE_SCREEN_FAILED';break
            for arm in ['mc','gae']:
                phase='PPO_UPDATE_'+arm;q=actors[arm];opt=opts[arm];before=copy.deepcopy(q.state_dict());before_opt=copy.deepcopy(opt.state_dict());shadow=copy.deepcopy(q);shadow_opt=T.L.optimizer(shadow,p);shadow_opt.load_state_dict(before_opt);items,target=fitted[arm]
                try:
                    live=actor_steps(p,q,opt,items,target,arm,rnd,ax,ay,offset,guard,lambda k,e:event(arm,'live',k,e))
                    check=actor_steps(p,shadow,shadow_opt,items,target,arm,rnd,ax,ay,offset,guard,lambda k,e:event(arm,'shadow',k,e))
                    assert live==check and S.digest(q.state_dict())==S.digest(shadow.state_dict()) and S.digest(opt.state_dict())==S.digest(shadow_opt.state_dict())
                    guard_result=native_guard(q,arm,rnd) if all(r['numeric_pass'] for r in live) else None
                    accepted=bool(guard_result and guard_result['all_safe']);candidate_policy=S.digest(q.state_dict());candidate_opt=S.digest(opt.state_dict())
                except BaseException:
                    q.load_state_dict(before);opt.load_state_dict(before_opt);raise
                if not accepted:q.load_state_dict(before);opt.load_state_dict(before_opt)
                updates.append(dict(arm=arm,round=rnd,steps=live,accepted=accepted,candidate_policy_hash=candidate_policy,candidate_optimizer_hash=candidate_opt,native_guard=guard_result,retained_policy_hash=S.digest(q.state_dict()),retained_optimizer_hash=S.digest(opt.state_dict()),rejected_snapshot_preserved_by_exact_recipe_and_training_arrays=True))
                C.save(OUT/'round_updates.json',updates)
            for arm in ['mc','gae']:
                roll(deterministic(p['old_centers']+p['centers'],f'{arm}_r{rnd}_centers'),actors[arm],'round_centers',arm,rnd)
            stop='FIXED_BUDGET_COMPLETE'
        phase='FINAL_EXPORT';status(True)
        for arm,q in actors.items():
            payload=dict(policy=q.state_dict(),optimizer=opts[arm].state_dict(),value_models=[v.state_dict() for v in vlast.get(arm,([],[]))[0]],value_optimizers=[v.state_dict() for v in vlast.get(arm,([],[]))[1]],protocol_sha256=C.sha(HERE/'protocol.json'),updates=[u for u in updates if u['arm']==arm],initial_policy_hash=p['policy_hash'],arm=arm)
            b=io.BytesIO();torch.save(payload,b);blob=gzip.compress(b.getvalue(),compresslevel=6,mtime=0);assert len(blob)<=p['actor_model_bytes'];meter.charge_payload(len(blob));f=OUT/(arm+'_final_checkpoint.pt.gz');f.write_bytes(blob)
            loaded=torch.load(io.BytesIO(gzip.decompress(blob)),map_location='cpu',weights_only=False);assert S.digest(payload)==S.digest(loaded)
            q.load_state_dict(loaded['policy']);exports[arm]=dict(path=str(f),sha256=C.sha(f),policy_hash=S.digest(q.state_dict()),optimizer_hash=S.digest(opts[arm].state_dict()),bytes=len(blob))
        for arm in ['mc','gae']:
            old=roll(deterministic(p['old_development']+p['old_regression64'],arm+'_old'),actors[arm],'old_regression',arm)
            dev=roll(deterministic(p['development_jobs'],arm+'_six'),actors[arm],'six_development',arm)
            if all(safe(i['row']) for i in old+dev):
                held=roll(deterministic(p['heldout_jobs'],arm+'_fresh'),actors[arm],'heldout',arm)
                if all(safe(i['row']) for i in held):
                    refs={i['row']['source_reset_id']:i['row'] for i in dev+held}
                    for i in raw:
                        if i['arm']==arm and i['role']=='old_regression' and i['source_reset_id'] in {j['id'] for j in p['old_centers']}:refs[i['source_reset_id']]=i
                    reload=roll(deterministic(p['development_jobs']+p['heldout_jobs']+p['old_centers'],arm+'_reload'),actors[arm],'independent_reload',arm)
                    assert all(i['row']['array_hashes']==refs[i['row']['source_reset_id']]['array_hashes'] and i['row']['integration_sha256']==refs[i['row']['source_reset_id']]['integration_sha256'] for i in reload)
        state='COMPLETE'
    except BaseException:C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),phase=phase,commands=0 if meter is None else meter.commands,physics_steps=sum(counts[:]),inflight=[j['id'] for j in active],optimizer_calls=len(calls)))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except OSError:pass
        for w in workers:
            w.join(5)
            if w.is_alive():w.kill();w.join(3)
        # Reserved final space also retains the current state after interruption.
        for arm,q in actors.items():
            if arm in exports:continue
            try:
                vm,vo=vlast.get(arm,([],[]))
                if not vm and VALUE_INFLIGHT:vm,vo=map(list,zip(*VALUE_INFLIGHT.values()))
                payload=dict(policy=q.state_dict(),optimizer=opts[arm].state_dict(),value_models=[v.state_dict() for v in vm],value_optimizers=[v.state_dict() for v in vo],partial=True,arm=arm,protocol_sha256=C.sha(HERE/'protocol.json'))
                buf=io.BytesIO();torch.save(payload,buf);blob=gzip.compress(buf.getvalue(),compresslevel=6,mtime=0);assert len(blob)<=p['actor_model_bytes'];meter.charge_payload(len(blob));f=OUT/(arm+'_partial_checkpoint.pt.gz');assert not f.exists();f.write_bytes(blob)
                exports[arm]=dict(path=str(f),sha256=C.sha(f),policy_hash=S.digest(q.state_dict()),optimizer_hash=S.digest(opts[arm].state_dict()),bytes=len(blob),partial=True)
            except BaseException:C.save(OUT/(arm+'_partial_export_error.json'),dict(traceback=traceback.format_exc()));state='FAILED'
        signal.alarm(0)
        if meter is not None:meter.close()
    s=dict(state=state,stop_reason=stop,pid=os.getpid(),commands=0 if meter is None else meter.commands,physics_steps=sum(counts[:]),raw_record_physics=sum(r['physics_steps'] for r in raw+replays),geometry_commands=len(geometries),unique_training_episodes=sum(r['role']=='training' for r in raw),exact_replays=len(replays),optimizer_calls=len(calls),live_Actor_calls=sum(c['kind']=='Actor' and c['replica']=='live' for c in calls),shadow_Actor_calls=sum(c['kind']=='Actor' and c['replica']=='shadow' for c in calls),live_V_calls=sum(c['kind']=='V' and c['replica']=='live' for c in calls),shadow_V_calls=sum(c['kind']=='V' and c['replica']=='shadow' for c in calls),accepted_blocks=sum(u['accepted'] for u in updates),models=exports,all_workers_stopped=all(not w.is_alive() for w in workers),inflight=[j['id'] for j in active],main_epoch=start,absolute_deadline_epoch=deadline,training_deadline_epoch=training_deadline,elapsed_s=time.time()-start,own_bytes=S.used(OUT),global_bytes=S.used(ROOT/'stage/runs'),seven_eight_entered=False,automatic_next_training=False)
    C.save(OUT/'summary.json',s);C.save(OUT/'status.json',s);print(json.dumps(s),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['preflight','main']);args=parser.parse_args();globals()[args.mode]()
