from ar_support import *
import multiprocessing as mp,fcntl,shutil
class Stop(Exception):pass
def main():
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'status.json').exists(),'No duplicateT239'
    plan=json.loads((HERE/'protocol.json').read_text());start=time.monotonic();wall=time.time()+plan['wall_seconds'];physics=commands=accepted=shadow_accepted=0;workers=[];pipes=[];rows=[];checks=[];roundchecks=[];dev=[];hold=[];export=None;reason=None;last_checks=[];replay_count=0
    safe=lambda r:r['complete'] and r['success'] and not r['collision']
    def append(name,r):
        with (OUT/name).open('a') as f:f.write(json.dumps(r)+'\n')
    def status(state):C.save(OUT/'status.json',dict(state=state,physics_steps=physics,commands=commands,RL_accepted_updates=accepted,shadow_accepted_updates=shadow_accepted,unique_training_episodes=len(rows),exact_replays=replay_count,development_n=len(dev),heldout_n=len(hold),elapsed_s=time.monotonic()-start))
    def guard(n=0,count=0,pending=0,reserve=0):
        if time.time()+reserve>=wall:raise Stop('WALL_RESERVE')
        if commands+n>plan['max_commands'] or physics+count>plan['max_physics']:raise Stop('COUNT_RESERVE')
        if used(OUT)+pending>=plan['final_bytes'] or used(ROOT/'stage/runs')+pending>=plan['global_bytes'] or shutil.disk_usage(ROOT).free<40*1024**3:raise Stop('STORAGE_RESERVE')
    def receive(pipe):
        while not pipe.poll(.5):
            if time.time()>wall:raise Stop('RESPONSE_DEADLINE')
        r=pipe.recv();assert 'error' not in r,r;return r
    def roll(jobs,policies,name,replay=False):
        nonlocal physics,commands,replay_count
        result=[]
        for off in range(0,len(jobs),4):
            batchjobs=jobs[off:off+4];mult=2 if replay else 1;guard(len(batchjobs)*mult,sum(j['case']['step_cap'] for j in batchjobs)*mult,pending=len(batchjobs)*64*1024+8192)
            jj=[dict(j,policy_np=state_np(policies[j['arm']]),policy_hash=digest(policies[j['arm']].state_dict())) for j in batchjobs]
            for pipe,j in zip(pipes,jj):pipe.send(j)
            first=[receive(p) for p in pipes[:len(jj)]];commands+=len(first);physics+=sum(i['row']['physics_steps'] for i in first)
            for j,item in zip(batchjobs,first):
                a=item['arrays'];compact={k:v for k,v in a.items() if k not in ['states','next_states']};buf=io.BytesIO();np.savez_compressed(buf,**compact);b=buf.getvalue();assert len(b)<64*1024;guard(pending=len(b)+8192);folder=OUT/'episodes';folder.mkdir(exist_ok=True);path=folder/(j['id']+'.npz');assert not path.exists();path.write_bytes(b);item['row'].update(path=str(path),sha256=C.sha(path));append(name,item['row']);result.append(item)
            if replay:
                for pipe,j in zip(pipes,jj):pipe.send(j)
                second=[receive(p) for p in pipes[:len(jj)]];commands+=len(second);physics+=sum(i['row']['physics_steps'] for i in second)
                for a,b in zip(first,second):
                    assert a['row']['array_hashes']==b['row']['array_hashes'] and a['row']['integration_sha256']==b['row']['integration_sha256'] and all(np.array_equal(v,b['arrays'][k]) for k,v in a['arrays'].items());replay_count+=1;append('replays.jsonl',b['row'])
            if not all(i['row']['complete'] for i in first):raise Stop('CENSORED_EPISODE')
            status(name)
        return result
    def eval_jobs(jobs,prefix):return [dict(j,id=prefix+'_'+j['id'],source_reset_id=j['id'],arm='learner',round=-1,seed=0,mode='deterministic') for j in jobs]
    try:
        for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
        contract=json.loads((OUT/'synthetic_contracts.json').read_text());assert contract['state']=='PASS' and contract['protocol_sha256']==C.sha(HERE/'protocol.json')
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(plan['seed']);p=S.policy(read(plan['checkpoint']));frozen=copy.deepcopy(p).eval().requires_grad_(False);opt=torch.optim.Adam([q for q in p.parameters() if q.requires_grad],lr=plan['lr']);ax,ay,offset=anchors(plan,frozen)
        assert len(plan['hard_jobs'])==2 and len(plan['center_jobs'])==8
        C.save(OUT/'contracts.json',dict(state='PASS',density_preflight='T238',initial_policy_hash=digest(p.state_dict()),anchor_states=len(ax),anchor_offsets=offset,anchor_state_hash=digest(ax),anchor_target_hash=digest(ay),trainable_parameters=sum(q.numel() for q in p.parameters() if q.requires_grad),physics_steps=0,optimizer_calls=0))
        ctx=mp.get_context('spawn')
        for i in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(i+1);w=ctx.Process(target=worker,args=(b,i,wall));w.start();b.close();workers.append(w);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';assert all(receive(p)['ready'] for p in pipes);C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[w.pid for w in workers]))
        baseline=roll(eval_jobs(plan['hard_jobs'],'baseline'),dict(learner=p),'baseline.jsonl')
        prior_rows=list(map(json.loads,(ROOT/'stage/runs/t231_retained_behavior_repair/development.jsonl').read_text().splitlines()))
        for item in baseline:
            r=item['row'];old=next(x for x in prior_rows if x['id']==r['job']['source_reset_id']);assert r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
        for rnd in range(plan['rounds']):
            guard(reserve=plan['final_reserve_seconds']);jobs=[]
            for arm in (['learner'] if rnd==0 else ['learner','frozen']):
                for j in plan['hard_jobs']:
                    for k in range(plan['draws_per_reset']):jobs.append(dict(j,id=f'r{rnd}_{arm}_{j["id"]}_s{k}',source_reset_id=j['id'],arm=arm,round=rnd,seed=plan['seed']+10*rnd+k,mode='conditional_AR'))
            items=roll(jobs,dict(learner=p,frozen=frozen),'training.jsonl',replay=True);rows.extend(i['row'] for i in items);train=[i for i in items if i['row']['arm']=='learner'];controls=[i for i in items if i['row']['arm']=='frozen']
            if rnd==0:
                for i in train:append('shared_control_aliases.jsonl',dict(control_arm='frozen',shared_learner_episode=i['row']['id'],reason='Identicalinitialpolicyandseed, notanindependentsample',success=i['row']['success'],collision=i['row']['collision']))
                controls=train
            data=batch(train,p,plan);before=copy.deepcopy(p);shadow_opt=torch.optim.Adam([q for q in before.parameters() if q.requires_grad],lr=plan['lr']);shadow_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
            live=update(p,opt,data,ax,ay,offset,plan,rnd,lambda:guard(reserve=plan['final_reserve_seconds']));accepted+=sum(r['accepted'] for r in live)
            shadow=update(before,shadow_opt,data,ax,ay,offset,plan,rnd,lambda:guard(reserve=plan['final_reserve_seconds']));shadow_accepted+=sum(r['accepted'] for r in shadow)
            assert live==shadow and digest(p.state_dict())==digest(before.state_dict()) and digest(opt.state_dict())==digest(shadow_opt.state_dict())
            report=dict(round=rnd,density_check=data['density_check'],learner_safe=sum(safe(i['row']) for i in train),frozen_safe=sum(safe(i['row']) for i in controls),episodes_per_arm=8,shared_initial_control=rnd==0,updates=live,independent_shadow_exact=True,policy_hash=digest(p.state_dict()),optimizer_hash=digest(opt.state_dict()));checks.append(report);append('updates.jsonl',report);status('RL_UPDATED')
            last_checks=roll(eval_jobs(plan['center_jobs']+plan['hard_jobs'],f'check{rnd}'),dict(learner=p),'round_checks.jsonl');roundchecks.extend(i['row'] for i in last_checks)
            centers=last_checks[:8]
            if not all(safe(i['row']) for i in centers):reason='ORIGINAL_CENTER_RETENTION_FAILED';break
        if reason is None:reason='FIXED_RL_BUDGET_COMPLETE'
        export=S.save_gzip(OUT/'final_Actor.pt.gz',dict(policy=p.state_dict(),protocol_sha256=C.sha(HERE/'protocol.json'),source='T239freshconditionalARMonteCarlopolicygradient',RL_accepted_updates=accepted),lambda n:guard(pending=n))
        loaded=S.policy(read(export['path'])).eval().requires_grad_(False);assert digest(p.state_dict())==digest(loaded.state_dict())
        C.save(OUT/'recipe.json',dict(updates=checks,final_policy_hash=digest(loaded.state_dict()),final_optimizer_hash=digest(opt.state_dict()),independent_shadow_exact=True,export=export))
        final=roll(eval_jobs(plan['development_jobs'],'final'),dict(learner=loaded),'development.jsonl');dev=[i['row'] for i in final]
        overlap={i['row']['job']['source_reset_id']:i['row'] for i in last_checks}
        for r in dev:
            key=r['job']['source_reset_id']
            if key in overlap:assert r['array_hashes']==overlap[key]['array_hashes'] and r['integration_sha256']==overlap[key]['integration_sha256']
        if len(dev)==211 and all(safe(r) for r in dev):hold=[i['row'] for i in roll(eval_jobs(plan['heldout_jobs'],'heldout'),dict(learner=loaded),'heldout.jsonl')]
    except Stop as e:reason=str(e)
    except BaseException:reason='FAILED';C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physics_steps=physics,commands=commands,inflight_physics_upper_bound=4*23334))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except OSError:pass
        for w in workers:
            w.join(3)
            if w.is_alive():w.kill();w.join(3)
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    scores={}
    for group in ['task'+str(i) for i in range(8)]:
        rr=[r for r in dev if r['job']['group']==group];hh=[r for r in hold if r['job']['group']==group];scores[group]=dict(n=len(rr),safeGOAL=sum(safe(r) for r in rr),CONTACT=sum(r['collision'] for r in rr),heldout_n=len(hh),heldout_safe=sum(safe(r) for r in hh))
    gate=len(dev)==211 and all(safe(r) for r in dev) and len(hold)==64 and all(safe(r) for r in hold)
    summary=dict(state='FAILED' if reason=='FAILED' else 'COMPLETE',stop_reason=reason,scores=scores,unique_training_episodes=len(rows),exact_replays=replay_count,RL_accepted_updates=accepted,shadow_accepted_updates=shadow_accepted,rounds=len(checks),development_n=len(dev),development_safeGOAL=sum(safe(r) for r in dev),known_invalid_initial_states=5,heldout_n=len(hold),heldout_safe=sum(safe(r) for r in hold),stable_valid_local_3m_4m_5m_gate=gate,export=export,physics_steps=physics,commands=commands,all_workers_stopped=all(not w.is_alive() for w in workers),automatic_promotion=False,elapsed_s=time.monotonic()-start,own_bytes=used(OUT),runs_MiB=used(ROOT/'stage/runs')/1024**2)
    C.save(OUT/'summary.json',summary);C.save(OUT/'status.json',summary);print(json.dumps(summary),flush=True)
    assert physics<=plan['max_physics'] and commands<=plan['max_commands'] and used(OUT)<plan['final_bytes'] and used(ROOT/'stage/runs')<plan['global_bytes']
if __name__=='__main__':main()
