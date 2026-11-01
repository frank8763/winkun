from closed_loop_learning import *
from distance_worker import worker
import multiprocessing as mp,fcntl,shutil,io,gzip,json,math,subprocess
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t258_closed_loop_retention'
class Stop(Exception):pass
def main():
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'status.json').exists(),'No duplicate/restart T258'
    plan=json.loads((HERE/'protocol.json').read_text());start=time.time();wall=start+plan['wall_seconds'];training=False;physics=commands=replays=accepted=shadow_accepted=0;workers=[];pipes=[];updates=[];training_rows=[];old=[];dev=[];hold=[];verification=[];nextrows=[];invalid=[];export=None;reason='PREPARING';gate=False;p=opt=None;inflight=0
    safe=lambda r:r['complete'] and r['success'] and not r['collision']
    def append(name,r):
        with (OUT/name).open('a') as f:f.write(json.dumps(r)+'\n')
    def status(state):C.save(OUT/'status.json',dict(state=state,pid=os.getpid(),primary_distance_m=plan['primary_distance_m'],physics_steps=physics,commands=commands,unique_training_episodes=len(training_rows),exact_replays=replays,accepted_RL_updates=accepted,shadow_accepted_updates=shadow_accepted,rounds=len(updates),old_regression_n=len(old),new_development_n=len(dev),new_heldout_n=len(hold),gate=gate,updated_epoch=time.time(),elapsed_s=time.time()-start))
    def guard(n=0,count=0,pending=0,check_time=True):
        if check_time and time.time()>=wall-(plan['final_reserve_seconds'] if training else 0):raise Stop('TRAINING_FINAL_RESERVE' if training else 'WALL')
        reserve=8*1024**2 if training else 0
        if commands+n>plan['max_commands'] or physics+count>plan['max_physics']:raise Stop('COUNTS')
        if S.used(OUT)+pending+reserve>=plan['final_bytes'] or S.used(ROOT/'stage/runs')+pending+reserve>=plan['global_bytes'] or shutil.disk_usage(ROOT).free<plan['min_free_bytes']:raise Stop('STORAGE')
    def receive(pipe):
        while not pipe.poll(.5):
            if time.time()>=wall:raise Stop('RESPONSE_DEADLINE')
        r=pipe.recv();assert 'error' not in r,r;return r
    def jobs_for(jobs,prefix):return [dict(j,id=prefix+'_'+j['id'],source_reset_id=j['id'],mode='deterministic',seed=0,memory_on=False,model_seed=2026103501) for j in jobs]
    def geometry(jobs):
        nonlocal commands
        result=[]
        for off in range(0,len(jobs),4):
            jj=jobs[off:off+4];guard(len(jj))
            for pipe,j in zip(pipes,jj):pipe.send(dict(j,geometry_only=True,source_reset_id=j['id']))
            rr=[receive(pipe) for pipe in pipes[:len(jj)]];commands+=len(rr)
            for r in rr:append('geometry.jsonl',r);result.append(r)
            status('INITIAL_GEOMETRY')
        return result
    def roll(jobs,actor,name,replay=False):
        nonlocal commands,physics,replays,inflight
        result=[]
        for off in range(0,len(jobs),4):
            jj=jobs[off:off+4];mult=2 if replay else 1;guard(len(jj)*mult,sum(j['case']['step_cap'] for j in jj)*mult,len(jj)*128*1024);state=S.state_np(actor);ph=S.digest(actor.state_dict());messages=[dict(j,policy_np=state,policy_hash=ph) for j in jj];inflight=sum(j['case']['step_cap'] for j in jj)
            for pipe,j in zip(pipes,messages):pipe.send(j)
            first=[receive(pipe) for pipe in pipes[:len(jj)]];commands+=len(first);physics+=sum(i['row']['physics_steps'] for i in first);inflight=0
            for item,j in zip(first,jj):
                a=item['arrays'];buf=io.BytesIO();np.savez_compressed(buf,**{k:v for k,v in a.items() if k not in ['states','next_states']});data=buf.getvalue();assert len(data)<128*1024;guard(pending=len(data)+8192,check_time=False);path=OUT/'episodes'/(j['id']+'.npz');path.parent.mkdir(exist_ok=True);assert not path.exists();path.write_bytes(data);item['row'].update(path=str(path),sha256=C.sha(path))
                if name=='training.jsonl':
                    ref=historical.get((item['row']['source_reset_id'],int(item['row']['seed'])));item['row']['exact_historical_overlap']=bool(ref is not None and item['row']['array_hashes']==ref['array_hashes'] and item['row']['integration_sha256']==ref['integration_sha256'])
                append(name,item['row']);result.append(item)
            if replay:
                inflight=sum(j['case']['step_cap'] for j in jj)
                for pipe,j in zip(pipes,messages):pipe.send(j)
                second=[receive(pipe) for pipe in pipes[:len(jj)]];commands+=len(second);physics+=sum(i['row']['physics_steps'] for i in second);inflight=0
                for a,b in zip(first,second):
                    append('replays.jsonl',b['row']);assert a['row']['array_hashes']==b['row']['array_hashes'] and a['row']['integration_sha256']==b['row']['integration_sha256'] and all(np.array_equal(v,b['arrays'][k]) for k,v in a['arrays'].items());replays+=1
            assert all(i['row']['complete'] for i in first),'Censored trajectories excluded from training';status(name)
        return result
    native_cache={};native_queries=0;native_cache_hits=0
    historical={(r['source_reset_id'],int(r['seed'])):r for r in map(json.loads,Path(plan['historical_training_rows']).read_text().splitlines())}
    def native_gate(actor):
        nonlocal native_queries,native_cache_hits
        ph=S.digest(actor.state_dict());native_queries+=1
        if ph in native_cache:native_cache_hits+=1;return native_cache[ph]
        prefix='nativeguard'+str(len(native_cache));items=roll(jobs_for(plan['acceptance_guard_jobs'],prefix),actor,'candidate_guards.jsonl');rr=[i['row'] for i in items]
        result=dict(policy_hash=ph,cases=len(rr),safe=sum(safe(r) for r in rr),all_safe=len(rr)==12 and all(safe(r) for r in rr),failed_ids=[r['source_reset_id'] for r in rr if not safe(r)])
        native_cache[ph]=result;C.save(OUT/'native_guard_cache_index.json',dict(queries=native_queries,cache_hits=native_cache_hits,cache=native_cache));return result
    def save_model(name):
        payload=dict(policy=p.state_dict(),optimizer=opt.state_dict(),protocol_sha256=C.sha(HERE/'protocol.json'),accepted_RL_updates=accepted,source_checkpoint=plan['checkpoint'],source_policy_hash=plan['policy_hash'],memory_on=False,seed=plan['seed'],updates=updates);buf=io.BytesIO();torch.save(payload,buf);data=gzip.compress(buf.getvalue(),compresslevel=6,mtime=0);guard(pending=len(data),check_time=False);path=OUT/name;assert not path.exists();path.write_bytes(data);loaded=L.read(path);assert S.digest(loaded)==S.digest(payload);q=policy(plan);q.load_state_dict(loaded['policy']);q.eval().requires_grad_(False);return q,dict(path=str(path),sha256=C.sha(path),bytes=len(data),policy_hash=S.digest(q.state_dict()),optimizer_hash=S.digest(loaded['optimizer']))
    try:
        for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
        contracts=json.loads((OUT/'contracts.json').read_text());assert contracts['state']=='PASS' and contracts['protocol_sha256']==C.sha(HERE/'protocol.json')
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);p=policy(plan);frozen=copy.deepcopy(p).eval().requires_grad_(False);opt=L.optimizer(p,plan);ax,ay,offset=anchors(plan,frozen);C.save(OUT/'retention_contract.json',dict(states=len(ax),episodes=211,state_hash=S.digest(ax),target_hash=S.digest(ay),source='T256actualaccepted5m120states',unused_old64_labels=True))
        ctx=mp.get_context('spawn')
        for i in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(i+1);w=ctx.Process(target=worker,args=(b,i,str(OUT),plan,wall));w.start();b.close();workers.append(w);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';assert all(receive(pipe)['ready'] for pipe in pipes);C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[w.pid for w in workers],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
        gg=geometry(plan['development_jobs']+plan['heldout_jobs']);valid={r['id'] for r in gg if r['valid']};invalid=[r for r in gg if not r['valid']];valid_dev=[j for j in plan['development_jobs'] if j['id'] in valid];assert all(j['id'] in valid for j in plan['heldout_jobs']) and all(sum(j['case']['side']==side for j in valid_dev)>=20 for side in ['LEFT','RIGHT'])
        centers=[j for j in valid_dev if j['delta_x_m']==j['delta_y_m']==j['delta_yaw_deg']==0];assert len(centers)==2;outer={side:[j for j in valid_dev if j['case']['side']==side and j['delta_yaw_deg']==0 and (j['delta_x_m']!=0 or j['delta_y_m']!=0)] for side in ['LEFT','RIGHT']};assert all(len(v)>=6 for v in outer.values())
        bridge_bases=geometry(plan['bridge_cases']);assert all(r['valid'] for r in bridge_bases);bridge_by_side={j['case']['side']:dict(j,exact_snapshot=r['snapshot']) for j,r in zip(plan['bridge_cases'],bridge_bases)}
        def bridge_job(j):
            b=bridge_by_side[j['case']['side']];return {**{k:v for k,v in j.items() if k not in ['snapshot_destination']},'id':'bridge_'+j['id'],'source_reset_id':'bridge_'+j['id'],'case':b['case'],'exact_snapshot':b['exact_snapshot'],'group':'bridge_'+j['group']}
        bridge_preflight=geometry([bridge_job(j) for j in valid_dev if j['delta_yaw_deg']==0]);assert len(bridge_preflight)==18 and all(r['valid'] for r in bridge_preflight)
        C.save(OUT/'locked_valid_sets_before_learning.json',dict(development=[j['id'] for j in valid_dev],heldout=[j['id'] for j in plan['heldout_jobs']],invalid=invalid,bridge_bases=bridge_bases,bridge_preflight=bridge_preflight,optimizer_calls=0))
        baseline_tasks=centers+[outer['LEFT'][0],outer['RIGHT'][0]];baseline=roll(jobs_for(baseline_tasks,'baseline'),p,'baseline.jsonl');prior=[json.loads(x) for x in (ROOT/'stage/runs/t256_distance_frontier/distance.jsonl').read_text().splitlines()]
        for item,j in zip(baseline,baseline_tasks):
            ref=next(r for r in prior if r['group']==j['group'] and r['job']['delta_x_m']==j['delta_x_m'] and r['job']['delta_y_m']==j['delta_y_m'] and r['job']['delta_yaw_deg']==0);assert item['row']['array_hashes']==ref['array_hashes'] and item['row']['integration_sha256']==ref['integration_sha256']
        training=True;zero=0
        try:
            for rnd in range(plan['rounds']):
                guard();selected=centers+[outer['LEFT'][rnd%len(outer['LEFT'])],outer['RIGHT'][rnd%len(outer['RIGHT'])]]
                if rnd<plan['bridge_rounds']:selected=[bridge_job(j) for j in selected]
                jobs=jobs_for(selected,f'r{rnd}');jobs=[dict(j,id=j['id']+f'_s{k}',mode='conditional_AR',seed=plan['seed']+100*rnd+k) for j in jobs for k in range(4)];items=roll(jobs,p,'training.jsonl',True);training_rows.extend(i['row'] for i in items);data=L.training_batch(items,p)
                if rnd==0:assert len(items)==16 and all(i['row']['exact_historical_overlap'] for i in items),'Initial16 must reproduce historical T257; not new independent data'
                shadow=copy.deepcopy(p);so=L.optimizer(shadow,plan);so.load_state_dict(copy.deepcopy(opt.state_dict()));live,audit=paired_update(p,opt,shadow,so,data,ax,ay,offset,plan,rnd,guard,native_gate,lambda phase,event:append('proposal_events.jsonl',dict(round=rnd,phase=phase,**event)));assert live==audit and S.digest(p.state_dict())==S.digest(shadow.state_dict()) and S.digest(opt.state_dict())==S.digest(so.state_dict());n=sum(r['accepted'] for r in live);accepted+=n;shadow_accepted+=sum(r['accepted'] for r in audit);zero=zero+1 if n==0 else 0;r=dict(round=rnd,training_distance_m=selected[0]['case']['distance_m'],safe_exploration=sum(safe(i['row']) for i in items),episodes=len(items),density=data['density'],updates=live,shadow_exact=True);updates.append(r);append('updates.jsonl',r)
                checks=roll(jobs_for(plan['old_centers']+centers,f'check{rnd}'),p,'round_centers.jsonl')
                if not all(safe(i['row']) for i in checks[:8]):reason='OLD5M_CENTER_LOSS';break
                if zero>=2:reason='TWO_ZERO_UPDATE_ROUNDS';break
                status('RL_ROUND_COMPLETE')
            else:reason='FIXED_BUDGET_COMPLETE'
        except Stop as e:reason=str(e)
        training=False;loaded,export=save_model('final_checkpoint.pt.gz');old=[i['row'] for i in roll(jobs_for(plan['old_development']+plan['old_regression64'],'oldregression'),loaded,'old_regression.jsonl')];dev=[i['row'] for i in roll(jobs_for(valid_dev,'newdev'),loaded,'development.jsonl')]
        if len(old)==275 and all(safe(r) for r in old) and len(dev)==len(valid_dev) and all(safe(r) for r in dev):
            hold=[i['row'] for i in roll(jobs_for(plan['heldout_jobs'],'heldout'),loaded,'heldout.jsonl')]
            if len(hold)==16 and all(safe(r) for r in hold):
                q=policy(plan);q.load_state_dict(L.read(export['path'])['policy']);q.eval().requires_grad_(False);refs={r['source_reset_id']:r for r in old+dev+hold};verification=[i['row'] for i in roll(jobs_for(valid_dev+plan['heldout_jobs']+plan['old_centers'],'verify'),q,'verification.jsonl')]
                for r in verification:
                    ref=refs[r['source_reset_id']];assert r['array_hashes']==ref['array_hashes'] and r['integration_sha256']==ref['integration_sha256']
                gate=len(verification)==len(valid_dev)+24 and all(safe(r) for r in verification)
                if gate:nextrows=[i['row'] for i in roll(jobs_for(plan['next_distance_jobs'],'nextdistance'),q,'next_distance.jsonl')]
    except Stop as e:reason=str(e)
    except BaseException:
        reason='FAILED';C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physics_steps=physics,commands=commands,inflight_physics_upper_bound=inflight))
        training=False
        if p is not None and opt is not None and export is None:
            try:_,export=save_model('failed_current_checkpoint.pt.gz')
            except BaseException:C.save(OUT/'export_failure.json',dict(traceback=traceback.format_exc()))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except OSError:pass
        for w in workers:
            w.join(3)
            if w.is_alive():w.kill();w.join(3)
    result=dict(state='FAILED' if reason=='FAILED' else 'COMPLETE',stop_reason=reason,primary_distance_m=plan['primary_distance_m'],unique_training_episodes=len(training_rows),exact_replays=replays,accepted_RL_updates=accepted,shadow_accepted_updates=shadow_accepted,rounds=len(updates),old_regression_n=len(old),old_regression_safe=sum(safe(r) for r in old),new_development_n=len(dev),new_development_safe=sum(safe(r) for r in dev),new_development_failures=[dict(id=r['source_reset_id'],collision=r['collision'],timeout=r['timeout']) for r in dev if not safe(r)],heldout_n=len(hold),heldout_safe=sum(safe(r) for r in hold),reload_exact_verification_n=len(verification),new_distance_gate=gate,next_distance=nextrows,invalid_initial_states=invalid,export=export,physics_steps=physics,commands=commands,inflight_physics_upper_bound=inflight,all_workers_stopped=all(not w.is_alive() for w in workers),elapsed_s=time.time()-start,own_bytes=S.used(OUT),runs_MiB=S.used(ROOT/'stage/runs')/1024**2,automatic_next_training=False)
    eventpath=OUT/'proposal_events.jsonl';events=list(map(json.loads,eventpath.read_text().splitlines())) if eventpath.exists() else []
    result.update(native_guard_unique_policies=len(native_cache),native_guard_queries=native_queries,native_guard_exact_hash_cache_hits=native_cache_hits,optimizer_proposals_live=sum(e['phase']=='live' for e in events),optimizer_proposals_shadow=sum(e['phase']=='shadow' for e in events),accepted_live_events_before_round_rollback=sum(e['phase']=='live' and e.get('accepted',False) for e in events),rolled_back_partial_rounds=sum(e['phase']=='rollback' for e in events),exact_historical_training_overlap=sum(r.get('exact_historical_overlap',False) for r in training_rows),new_policy_induced_training_records=sum(not r.get('exact_historical_overlap',False) for r in training_rows),native_guard_not_global_safety_proof=True)
    C.save(OUT/'summary.json',result);C.save(OUT/'status.json',result)
    report=['# T258 新距离真实强化学习结果','',f'目标距离：{plan["primary_distance_m"]}米；状态：{result["state"]}；停止原因：{reason}。',f'真实探索回合：{len(training_rows)}；独立精确重生：{replays}；真实Actor接受更新：{accepted}；shadow接受：{shadow_accepted}。',f'原五米已知回归：{result["old_regression_safe"]}/{len(old)}；新距离开发：{result["new_development_safe"]}/{len(dev)}；新留出：{result["heldout_safe"]}/{len(hold)}；读回复验：{len(verification)}。',f'新距离门槛：{gate}；全部worker停止：{result["all_workers_stopped"]}。',f'全部物理步：{physics}；命令：{commands}；本轮逻辑字节：{result["own_bytes"]}。','',f'新距离失败：{result["new_development_failures"]}。','数量不足表示未执行/删失，不记为0成功率。旧64是已知回归，不能称新增泛化测试。单场景局部结果不是算法创新或跨地图/真机稳定证据。冻结五米父模型和所有失败保留，不自动追加训练。新增源/证据尚未远端同步，保留本地且全局仍1GiB，Phase隔离。']
    (ROOT/'stage/docs/long_distance_20261101/T258_RESULTS.md').write_text('\n'.join(report)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
