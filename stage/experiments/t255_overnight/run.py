"""Unique, bounded, restart-refusing Stage overnight pipeline."""
from learning import *
import multiprocessing as mp,fcntl,shutil,subprocess,signal
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t255_overnight'
class Stop(Exception):pass
def main():
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'status.json').exists(),'No duplicate/restart T255'
    plan=json.loads((HERE/'protocol.json').read_text());start=time.time();wall=min(start+plan['max_wall_seconds'],plan['hard_deadline_epoch']);evaluation_wall=min(wall-300,plan['evaluation_deadline_epoch']);training_wall=min(start+4*3600,plan['train_deadline_epoch'],evaluation_wall-2400,plan['no_new_training_epoch'])
    physics=commands=replays=0;workers=[];pipes=[];arms={};active={};reports={};dev={};hold=[];verify=[];eight=[];videos=[];reason='PREPARING';gate=False;selected=None;training=False;inflight=0
    safe=lambda r:r['complete'] and r['success'] and not r['collision']
    def append(name,r):
        with (OUT/name).open('a') as f:f.write(json.dumps(r)+'\n')
    def status(state):
        C.save(OUT/'status.json',dict(state=state,pid=os.getpid(),updated_epoch=time.time(),physics_steps=physics,commands=commands,exact_replays=replays,arms=reports,active=active,development={a:dict(n=len(rr),safe=sum(safe(r) for r in rr)) for a,rr in dev.items()},heldout_n=len(hold),verified_n=len(verify),stable_5m_gate=gate,selected=selected,training_deadline_epoch=training_wall,evaluation_deadline_epoch=evaluation_wall,hard_deadline_epoch=wall,elapsed_s=time.time()-start))
    def guard(n=0,count=0,pending=0):
        if time.time()>= (training_wall if training else evaluation_wall):raise Stop('TRAINING_WALL' if training else 'EVALUATION_WALL')
        if commands+n>plan['max_commands'] or physics+count>plan['max_physics']:raise Stop('COUNT_LIMIT')
        reserve=40*1024**2 if training else 0
        if used(OUT)+pending+reserve>=plan['final_bytes'] or used(ROOT/'stage/runs')+pending+reserve>=plan['global_bytes'] or shutil.disk_usage(ROOT).free<plan['min_free_bytes']:raise Stop('STORAGE_LIMIT')
    def receive(pipe):
        while not pipe.poll(.5):
            if time.time()>=evaluation_wall:raise Stop('WORKER_RESPONSE_DEADLINE')
        r=pipe.recv();assert 'error' not in r,r;return r
    def roll(jobs,p,name,replay=False):
        nonlocal commands,physics,replays,inflight
        result=[]
        for off in range(0,len(jobs),4):
            batch=jobs[off:off+4];mult=2 if replay else 1;guard(len(batch)*mult,sum(j['case']['step_cap'] for j in batch)*mult,len(batch)*96*1024)
            state=state_np(p);ph=digest(p.state_dict());jj=[dict(j,policy_np=state,policy_hash=ph) for j in batch];inflight=sum(j['case']['step_cap'] for j in batch)
            for pipe,j in zip(pipes,jj):pipe.send(j)
            first=[receive(pipe) for pipe in pipes[:len(jj)]];commands+=len(first);physics+=sum(i['row']['physics_steps'] for i in first);inflight=0
            for j,item in zip(batch,first):
                a=item['arrays'];buf=io.BytesIO();np.savez_compressed(buf,**{k:v for k,v in a.items() if k not in ['states','next_states']});b=buf.getvalue();assert len(b)<96*1024
                guard(pending=len(b)+8192);folder=OUT/'episodes';folder.mkdir(exist_ok=True);path=folder/(j['id']+'.npz');assert not path.exists();path.write_bytes(b);item['row'].update(path=str(path),sha256=C.sha(path));append(name,item['row']);result.append(item)
            if replay:
                inflight=sum(j['case']['step_cap'] for j in batch)
                for pipe,j in zip(pipes,jj):pipe.send(j)
                second=[receive(pipe) for pipe in pipes[:len(jj)]];commands+=len(second);physics+=sum(i['row']['physics_steps'] for i in second);inflight=0
                for a,b in zip(first,second):
                    append('replays.jsonl',b['row']);assert a['row']['array_hashes']==b['row']['array_hashes'] and a['row']['integration_sha256']==b['row']['integration_sha256'] and all(np.array_equal(v,b['arrays'][k]) for k,v in a['arrays'].items());replays+=1
            assert all(i['row']['complete'] for i in first),'Censored episodes are never learning data'
            status(name)
        return result
    def jobs_for(jobs,arm,prefix,mode='deterministic',rnd=-1):
        cfg=next(a for a in plan['arms'] if a['id']==arm)
        return [dict(j,id=prefix+'_'+arm+'_'+j['id'],source_reset_id=j['id'],arm=arm,round=rnd,seed=0,mode=mode,memory_on=cfg['memory_on'],model_seed=cfg['seed']) for j in jobs]
    def export(arm):
        p,opt=arms[arm];payload=dict(policy=p.state_dict(),optimizer=opt.state_dict(),memory_on=p.memory_on,seed=next(a['seed'] for a in plan['arms'] if a['id']==arm),protocol_sha256=C.sha(HERE/'protocol.json'),research_warm_start='T250; historical own-success consolidation plus actual RL, not from-scratch RL',report=reports[arm]);buf=io.BytesIO();torch.save(payload,buf);z=gzip.compress(buf.getvalue(),compresslevel=6,mtime=0);guard(pending=len(z));path=OUT/(arm+'_final_checkpoint.pt.gz');assert not path.exists();path.write_bytes(z);loaded=read(path);assert digest(loaded)==digest(payload);q=Actor(parent_state,p.memory_on,payload['seed']);q.load_state_dict(loaded['policy']);q.eval().requires_grad_(False);assert digest(q.state_dict())==digest(p.state_dict());reports[arm]['export']=dict(path=str(path),sha256=C.sha(path),bytes=len(z),policy_hash=digest(q.state_dict()),optimizer_hash=digest(loaded['optimizer']));return q
    try:
        for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
        contract=json.loads((OUT/'synthetic_contracts.json').read_text());assert contract['state']=='PASS' and contract['protocol_sha256']==C.sha(HERE/'protocol.json')
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);parent_state=read(plan['checkpoint'])['policy'];parent=AS.policy(parent_state).eval().requires_grad_(False);anchor=retention(plan,parent)
        for cfg in plan['arms']:
            p=Actor(parent_state,cfg['memory_on'],cfg['seed']);arms[cfg['id']]=(p,optimizer(p,plan));active[cfg['id']]=True;reports[cfg['id']]=dict(rounds=0,training_episodes=0,accepted_updates=0,shadow_accepted_updates=0,consecutive_zero_update_rounds=0,stop_reason=None)
        C.save(OUT/'launch_runtime.json',dict(pid=os.getpid(),start_epoch=start,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),protocol_sha256=C.sha(HERE/'protocol.json'),training_deadline_epoch=training_wall,evaluation_deadline_epoch=evaluation_wall,hard_deadline_epoch=wall,retention_ids=anchor['ids']))
        status('NATIVE_PREFLIGHT');ctx=mp.get_context('spawn')
        for i in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(i+1);w=ctx.Process(target=worker,args=(b,i,str(OUT),plan['checkpoint'],evaluation_wall));w.start();b.close();workers.append(w);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';assert all(receive(pipe)['ready'] for pipe in pipes);C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[w.pid for w in workers]))
        oldrows={r['job']['source_reset_id']:r for r in map(json.loads,(ROOT/'stage/runs/t250_actuator_policy_gradient/development.jsonl').read_text().splitlines())}
        controls=[j for j in plan['development_jobs'] if j['id'] in plan['initial_control_ids']]
        base_results=[]
        for cfg in plan['arms'][:2]:
            rr=roll(jobs_for(controls,cfg['id'],'baseline'),arms[cfg['id']][0],'baseline.jsonl')
            for item in rr:
                r=item['row'];old=oldrows[r['source_reset_id']];assert r['base_array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
            base_results.append([r['row']['array_hashes'] for r in rr])
        assert base_results[0]==base_results[1];C.save(OUT/'native_contracts.json',dict(state='PASS',controls_per_arm=len(controls),both_memory_flags_all_BASE_arrays_and_integration_match_T250=True,full120_arrays_match_between_arms=True))
        training=True;other=[j for j in plan['left_jobs'] if j['id']!='dev_task6_24'];hard=next(j for j in plan['left_jobs'] if j['id']=='dev_task6_24')
        try:
            for rnd in range(plan['rounds']):
                if not any(active.values()):break
                selected_tasks=[hard]+[other[(3*rnd+k)%26] for k in range(3)]
                for cfg in plan['arms']:
                    arm=cfg['id']
                    if not active[arm]:continue
                    guard();p,opt=arms[arm];jobs=jobs_for(selected_tasks,arm,f'r{rnd}','conditional_AR',rnd)
                    jobs=[dict(j,id=j['id']+f'_s{k}',seed=cfg['seed']+100*rnd+k) for j in jobs for k in range(4)]
                    items=roll(jobs,p,'training.jsonl',True);data=training_batch(items,p);shadow=copy.deepcopy(p);so=optimizer(shadow,plan);so.load_state_dict(copy.deepcopy(opt.state_dict()));live=update(p,opt,data,anchor,plan,cfg['seed'],rnd,guard);audit=update(shadow,so,data,anchor,plan,cfg['seed'],rnd,guard)
                    assert live==audit and digest(p.state_dict())==digest(shadow.state_dict()) and digest(opt.state_dict())==digest(so.state_dict());n=sum(u['accepted'] for u in live);r=reports[arm];r['training_episodes']+=len(items);r['rounds']+=1;r['accepted_updates']+=n;r['shadow_accepted_updates']+=sum(u['accepted'] for u in audit);r['consecutive_zero_update_rounds']=0 if n else r['consecutive_zero_update_rounds']+1
                    append('updates.jsonl',dict(arm=arm,round=rnd,task_ids=[j['id'] for j in selected_tasks],density=data['density'],safe_exploration=sum(safe(i['row']) for i in items),updates=live,shadow_exact=True))
                    check=roll(jobs_for(plan['center_jobs'],arm,f'center{rnd}'),p,'round_centers.jsonl');r['last_centers_safe']=sum(safe(i['row']) for i in check)
                    if not all(safe(i['row']) for i in check):active[arm]=False;r['stop_reason']='ORIGINAL_CENTER_LOSS'
                    elif r['consecutive_zero_update_rounds']>=2:active[arm]=False;r['stop_reason']='TWO_ZERO_UPDATE_ROUNDS'
                    status('RL_ROUND_COMPLETE')
            reason='FIXED_TRAINING_BUDGET_OR_PREDECLARED_STOP'
        except Stop as e:reason=str(e)
        training=False;status('EXPORT_AND_FROZEN_ACCEPTANCE');loaded={}
        for cfg in plan['arms']:
            arm=cfg['id'];loaded[arm]=export(arm);dev[arm]=[i['row'] for i in roll(jobs_for(plan['development_jobs'],arm,'final'),loaded[arm],arm+'_development.jsonl')]
        # One priority candidate only; do not use heldout to rank the four arms.
        eligible=[a['id'] for a in plan['arms'] if len(dev[a['id']])==211 and all(safe(r) for r in dev[a['id']])]
        if eligible:
            selected=eligible[0];p=loaded[selected];hold=[i['row'] for i in roll(jobs_for(plan['heldout_jobs'],selected,'heldout'),p,'heldout.jsonl')]
            if len(hold)==64 and all(safe(r) for r in hold):
                payload=read(reports[selected]['export']['path']);p2=Actor(parent_state,payload['memory_on'],payload['seed']);p2.load_state_dict(payload['policy']);p2.eval().requires_grad_(False)
                verify=[i['row'] for i in roll(jobs_for(plan['development_jobs']+plan['heldout_jobs'],selected,'verify'),p2,'verification.jsonl')];prior={r['source_reset_id']:r for r in dev[selected]+hold}
                for r in verify:
                    old=prior[r['source_reset_id']];assert r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
                gate=len(verify)==275 and all(safe(r) for r in verify);status('STABLE_5M_GATE_PASS' if gate else 'STABLE_5M_GATE_FAIL')
                if gate:
                    ej=[]
                    for j in plan['center_jobs'][6:8]:
                        case=dict(j['case'],id='8m_'+j['case']['id'],distance_m=8.,step_cap=37334);ej.append(dict(j,case=case,id='8m_'+j['id']))
                    ej=[{k:v for k,v in j.items() if k!='exact_snapshot'} for j in ej]
                    first=roll(jobs_for(ej,selected,'8m_first'),p2,'8m.jsonl');second=roll(jobs_for(ej,selected,'8m_repeat'),p2,'8m_repeat.jsonl')
                    for a,b in zip(first,second):assert a['row']['array_hashes']==b['row']['array_hashes'] and a['row']['integration_sha256']==b['row']['integration_sha256']
                    eight=[i['row'] for i in first]
                    from video import record
                    vjobs=[j for j in plan['development_jobs'] if j['id'] in ['dev_task6_13','dev_task7_13','dev_task6_24']];vj=jobs_for(vjobs,selected,'video');videos=record(OUT/'video',vj,p2,prior,evaluation_wall,guard);commands+=len(videos);physics+=sum(v['row']['physics_steps'] for v in videos);C.save(OUT/'video.json',dict(state='VERIFIED',cases=videos))
    except Stop as e:reason=str(e)
    except BaseException:
        reason='FAILED';C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physics_steps=physics,commands=commands,inflight_physics_upper_bound=inflight))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except OSError:pass
        for w in workers:
            w.join(3)
            if w.is_alive():w.kill();w.join(3)
    summary=dict(state='FAILED' if reason=='FAILED' else 'COMPLETE',stop_reason=reason,arms=reports,development={a:dict(n=len(rr),safe=sum(safe(r) for r in rr),failed_ids=[r['source_reset_id'] for r in rr if not safe(r)],scores={g:dict(n=sum(r['group']==g for r in rr),safe=sum(r['group']==g and safe(r) for r in rr)) for g in ['task'+str(i) for i in range(8)]}) for a,rr in dev.items()},selected=selected,heldout_n=len(hold),heldout_safe=sum(safe(r) for r in hold),independent_reload_verification_n=len(verify),stable_5m_gate=gate,eight_m=eight,video=videos,physics_steps=physics,commands=commands,exact_replays=replays,inflight_physics_upper_bound=inflight,known_invalid_original_starts=5,all_workers_stopped=all(not w.is_alive() for w in workers),elapsed_s=time.time()-start,own_bytes=used(OUT),runs_MiB=used(ROOT/'stage/runs')/1024**2,automatic_next_training=False)
    C.save(OUT/'summary.json',summary);C.save(OUT/'status.json',summary)
    lines=['# Stage 今夜结果（T255）','',f'结束状态：{summary["state"]}；原因：{reason}。',f'5米严格门槛：{gate}；64留出：{len(hold)}；独立读回复验：{len(verify)}。',f'物理步：{physics}；命令：{commands}；精确重生：{replays}；全部worker停止：{summary["all_workers_stopped"]}。','']
    for arm,s in summary['development'].items():lines.append(f'- {arm}：有效开发 {s["safe"]}/{s["n"]}；失败 {s["failed_ids"]}；RL接受 {reports[arm]["accepted_updates"]}；训练回合 {reports[arm]["training_episodes"]}。')
    lines+=['',f'8米实际执行：{len(eight)}例；新MuJoCo视频：{len(videos)}例。','若数量不足，属于未执行/预算删失，不能记为全部失败或通过。本结果仅为同场景局部开发范围，不是新地图/真机泛化或创新证明。','模型含历史自身成功整合谱系，本轮为真实深度策略梯度更新；shadow、精确重生不是独立样本。','Stage/Phase隔离；无地图、外部规划器、路点、运行时教师或策略路由。']
    (ROOT/'stage/docs/overnight_20261031/MORNING_REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary),flush=True)
    assert physics<=plan['max_physics'] and commands<=plan['max_commands'] and used(OUT)<plan['final_bytes'] and used(ROOT/'stage/runs')<plan['global_bytes']
if __name__=='__main__':main()
