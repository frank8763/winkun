"""Two registered supervision objectives, unchanged single causal Actor deployment."""
from pathlib import Path
import ast, copy, gzip, importlib.util, io, json, multiprocessing as mp, os
import signal, subprocess, sys, time, traceback

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t390r_worker_entry_recovery'
DOC=ROOT/'stage/docs/review_20261106'
spec=importlib.util.spec_from_file_location('stage_t390_locked_contract',ROOT/'stage/experiments/t387_full_horizon_night/night_common.py')
N=importlib.util.module_from_spec(spec);spec.loader.exec_module(N)
np,torch,read,save,sha,digest=N.np,N.torch,N.read,N.save,N.sha,N.digest
PRIOR=ROOT/'stage/runs/t387_full_horizon_night'
TEACHER=PRIOR/'phase_warp/mixed.npz'
ARMS=['prefix_complete','coherent_teacher_only']


def dataset(base,q,arm):
    if arm=='prefix_complete':
        groups=N.data(q,base,TEACHER,lambda:None)
        teacher=groups[-1]
        groups.append(dict(teacher,id='successful_actor_prefix20',first=0,last=20,weight=.5))
        old=N.old_data(q,base,lambda:None)
    else:
        a=N.S.arrays(TEACHER);x=torch.tensor(a['states'])
        groups=[dict(id='whole_successful_trajectory',x=x,b=q.base_actions(x),
                     target=torch.tensor(a['actions'],dtype=torch.float64),first=0,weight=1.)]
        old=None
    return groups,old


def fit(q,opt,groups,old,steps,guard,event):
    params=[v for v in q.parameters() if v.requires_grad]
    fixed=digest({n:v for n,v in q.state_dict().items() if n.startswith(('base.','future_head.'))})
    trace=[]
    for k in range(steps):
        guard();opt.zero_grad(set_to_none=True);values=[]
        for g in [*groups,*([old[k%len(old)]] if old is not None else [])]:
            loc=q.sequence(g['x'],g['b']);first,last=g['first'],g.get('last',len(loc))
            loss=g['weight']*(loc[first:last]-g['target'][first:last]).square().mean()
            loss.backward();values.append(float(loss.detach()))
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
        event('BEFORE',k);opt.step();event('AFTER',k)
        assert fixed==digest({n:v for n,v in q.state_dict().items() if n.startswith(('base.','future_head.'))})
        trace.append([*values,norm])
    return np.asarray(trace,np.float64)


def parameter_hashes(q,opt):
    return {k:digest(v) for k,v in N.learned_state(q,opt).items()}


def native_worker(pipe,index,base,counts):
    return N.native_worker(pipe,index,base,counts)


def prepare():
    assert not OUT.exists()
    original=ROOT/'stage/runs/t390_coherent_success_transfer'
    failed=read(original/'summary.json');assert failed['state']=='FAILED'
    assert failed['physics']==failed['commands']==sum(sum(v.values()) for v in failed['updates'].values())==0
    assert failed['all_workers_stopped']
    for pid in [failed['pid'],read(original/'launch.json')['pid'],read(original/'driver_started.json')['pid']]:
        assert not Path('/proc',str(pid)).exists()
    p=read(original/'protocol.json')
    native=read(original/'main_started.json');driver=read(original/'driver_started.json')
    p.update(id='T390R_ZERO_STEP_WORKER_ENTRY_RECOVERY',absolute_native_deadline=native['deadline'],
             absolute_learning_deadline=native['learning_deadline'],absolute_all_deadline=driver['all_deadline'],
             max_commands=p['max_commands']-2,preflight_worker_setup_commands=2,
             final_bytes=p['final_bytes']-N.K.S.used(original),original_failed_directory=str(original))
    extra=[*HERE.glob('*.py'),DOC/'T390R_RECOVERY_PLAN.md',original/'summary.json',original/'failure.json',
           original/'preflight.json',ROOT/'stage/experiments/t390_coherent_success_transfer/run.py']
    p['inputs'].update({str(f):sha(f) for f in extra})
    assert time.time()+600<p['absolute_learning_deadline']
    assert N.K.S.used(ROOT/'stage/runs')+p['final_bytes']<=1024**3
    OUT.mkdir();(OUT/'episodes').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    print('T390R_REGISTERED_ORIGINAL_COMBINED_BUDGET_AND_DEADLINES')



def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    base=N.science(p)
    N.capacity(OUT,p['final_bytes'])
    return base


def preflight():
    N.offline();p=read(OUT/'protocol.json');base=verify(p)
    assert not (OUT/'preflight.json').exists()
    original=Path(p['original_failed_directory'])
    assert read(original/'preflight.json')['discarded_Adam']==8
    def fun(text,name):
        return ast.dump(next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name))
    before=(ROOT/'stage/experiments/t390_coherent_success_transfer/run.py').read_text()
    after=(HERE/'run.py').read_text()
    for name in ['fit','dataset','parameter_hashes','audit']:
        assert fun(before,name)==fun(after,name),name
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    ctx=mp.get_context('spawn');counts=ctx.Array('q',2);procs=[];pipes=[];ready=[]
    base.update(out=str(OUT/'worker_preflight'))
    try:
        for index in range(2):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(index+1)
            proc=ctx.Process(target=native_worker,args=(b,index,base,counts));proc.start();b.close();procs.append(proc);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0'
        for pipe in pipes:
            assert pipe.poll(45),'Startup timeout';r=pipe.recv();assert r.get('ready'),r;ready.append(r)
        assert all(r['affinity']==[k+1] for k,r in enumerate(ready))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except (OSError,BrokenPipeError):pass
        for proc in procs:
            proc.join(15)
            if proc.is_alive():proc.kill();proc.join(5)
    assert len(procs)==2 and all(p.exitcode==0 for p in procs) and sum(counts[:])==0
    save(OUT/'preflight.json',dict(state='PASS',native=0,geometry=2,discarded_Adam=0,reused_original_discarded_Adam=8,
                                  scientific_fit_dataset_and_audit_AST_exact=True,actual_spawn_entry_ready=ready,
                                  both_workers_exit_zero=True,original_error_preserved=True))
    print('T390R_REAL_WORKER_STARTUP_PASS_ZERO_NATIVE_ZERO_ADAM')



def main():
    N.offline();p=read(OUT/'protocol.json');base=verify(p);base.update(out=str(OUT));N.configure(OUT)
    assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'summary.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.time();deadline=min(start+p['main_seconds'],p['absolute_native_deadline']);learnend=min(start+p['learning_seconds'],p['absolute_learning_deadline'])
    save(OUT/'main_started.json',dict(pid=os.getpid(),epoch=start,deadline=deadline,learning_deadline=learnend))
    phase='START';commands=0;rows=[];replays=[];results={};processes=[];pipes=[];last=0.;calls={a:dict(live=0,shadow=0) for a in ARMS}
    ctx=mp.get_context('spawn');counts=ctx.Array('q',2)
    def status():
        nonlocal last
        assert time.time()<deadline-10;N.capacity(OUT,p['final_bytes'])
        if time.time()-last>3:
            last=time.time();save(OUT/'status.json',dict(state=phase,pid=os.getpid(),updated_epoch=last,commands=commands,physics=sum(counts[:]),updates=calls,deadline=deadline))
    def guard():assert time.time()<learnend;status()
    def receive(pipe):
        while not pipe.poll(.25):status()
        return pipe.recv()
    def export(q,opt,folder):
        b=io.BytesIO();torch.save(dict(state=N.learned_state(q,opt),hashes=parameter_hashes(q,opt),policy_hash=q.phash(),new_RL_updates=0),b)
        data=gzip.compress(b.getvalue(),mtime=0);assert len(data)<=384*1024
        path=folder/'candidate.pt.gz';path.write_bytes(data)
        return dict(path=str(path),sha256=sha(path),policy_hash=q.phash(),hashes=parameter_hashes(q,opt))
    def evaluate(js,role,q,model,reference=None):
        nonlocal phase,commands
        phase=role;items=[]
        for offset in range(0,len(js),2):
            group=js[offset:offset+2];assert time.time()+40<deadline
            assert commands+len(group)<=p['max_commands'] and sum(counts[:])+sum(j['case']['step_cap'] for j in group)<=p['max_physics']
            commands+=len(group);status()
            for pipe,j in zip(pipes,group):
                pipe.send(dict(j,id=role+'_'+j['id'],source_reset_id=j['id'],role=role,arm='memory',rho=0.,
                               learned=q.learned(),policy_hash=q.phash(),deadline_epoch=deadline-15,stochastic=False,
                               readback_model=model['path'] if reference is not None else None))
            batch=[receive(pipe) for pipe in pipes[:len(group)]]
            for item in batch:
                assert 'error' not in item,item.get('error');r,a=item['row'],item['arrays']
                if not r['complete']:
                    N.write_npz(OUT/'partial.npz',N.S.packed(a),640*1024);save(OUT/'partial_case.json',r)
                    raise RuntimeError('Native censored evidence preserved')
                array_digest=digest(a)
                if reference is not None:
                    before=reference[r['job_id']]
                    assert array_digest==before['array_digest'] and r['integration_sha256']==before['integration_sha256']
                    rec=dict(job_id=r['job_id'],role=role,physics_steps=r['physics_steps'],array_digest=array_digest,exact=True,independent_sample=False)
                    replays.append(rec)
                    with (OUT/'replays.jsonl').open('a') as f:f.write(json.dumps(rec,separators=(',',':'))+'\n')
                else:
                    path=OUT/'episodes'/f'{len(rows):04}.npz'
                    N.write_npz(path,{k:a[k] for k in ['actions','native_counts','end_flags']},8*1024)
                    rec={k:r[k] for k in ['job_id','success','collision','timeout','complete','physics_steps','remaining_m','integration_sha256']}
                    rec.update(role=role,path=str(path),sha256=sha(path),array_digest=array_digest)
                    assert len(json.dumps(rec).encode())<=1024
                    rows.append(rec)
                    with (OUT/'cases.jsonl').open('a') as f:f.write(json.dumps(rec,separators=(',',':'))+'\n')
                    if role.endswith('_centre'):
                        N.write_npz(OUT/role.split('_centre')[0]/'centre.npz',N.S.packed(a),640*1024)
                items.append(rec)
            status()
        return items
    def count(items):return dict(success=sum(v['success'] for v in items),collision=sum(v['collision'] for v in items),timeout=sum(v['timeout'] for v in items),valid=len(items),invalid=0,censored=0)
    state='FAILED'
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('T390 external stop')))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T390 deadline')));signal.alarm(max(1,int(deadline-start)-5))
    try:
        for index in range(2):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(index+1)
            proc=ctx.Process(target=native_worker,args=(b,index,base,counts));proc.start();b.close();pipes.append(a);processes.append(proc)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(pipe) for pipe in pipes]
        assert all(v['ready'] and v['affinity']==[k+1] for k,v in enumerate(ready))
        commands+=2
        save(OUT/'workers.json',dict(parent=os.getpid(),workers=[v.pid for v in processes],zero_native_setup_commands=2))
        for arm in ARMS:
            folder=OUT/arm;folder.mkdir();phase=arm+'_FIT';guard()
            q,opt=N.learner(base);shadow,shadowopt=N.learner(base)
            assert q.phash()==p['cached_initial_policy']
            groups,old=dataset(base,q,arm)
            with gzip.GzipFile(filename=str(folder/'optimizer_calls.jsonl.gz'),mode='wb',mtime=0) as events:
                def event(replica,where,k):
                    if where=='AFTER':calls[arm][replica]+=1
                    events.write((json.dumps([replica,where,k])+'\n').encode())
                trace=fit(q,opt,groups,old,1000,guard,lambda w,k:event('live',w,k))
                other=fit(shadow,shadowopt,groups,old,1000,guard,lambda w,k:event('shadow',w,k))
            assert N.equal(trace,other) and parameter_hashes(q,opt)==parameter_hashes(shadow,shadowopt)
            N.write_npz(folder/'trace.npz',dict(trace=trace),96*1024);model=export(q,opt,folder)
            centre=evaluate([base['narrow_jobs'][0]],arm+'_centre',q,model)
            evaluate([base['narrow_jobs'][0]],arm+'_centre_readback',q,model,{centre[0]['job_id']:centre[0]})
            result=dict(model=model,centre=count(centre),new070=None,old20=None,original2=None,old275=None,originalsix54=None,fresh16=None,reload78=None,stable_original_six=False)
            if centre[0]['success']:
                rest=evaluate(base['narrow_jobs'][1:],arm+'_new070',q,model)
                old20=evaluate(base['old_jobs'],arm+'_old20',q,model);orig=evaluate(base['original_jobs'],arm+'_original2',q,model)
                result.update(new070=count([*centre,*rest]),old20=count(old20),original2=count(orig))
                if result['new070']['success']==5 and result['old20']['success']==20 and result['original2']['success']==2:
                    oldall=evaluate(base['final_old_jobs'],arm+'_old275',q,model);six=evaluate(base['six_jobs'],arm+'_originalsix54',q,model)
                    result.update(old275=count(oldall),originalsix54=count(six))
                    if result['old275']['success']==275 and result['originalsix54']['success']==54:
                        fresh=evaluate(base['fresh_jobs'],arm+'_fresh16',q,model);result['fresh16']=count(fresh)
                        if result['fresh16']['success']==16:
                            mapping={v['job_id']:v for v in [*oldall,*six,*fresh]}
                            reload=evaluate(base['reload_jobs'],arm+'_reload78',q,model,mapping)
                            result.update(reload78=dict(exact=len(reload)==78),stable_original_six=len(reload)==78)
            results[arm]=result;save(folder/'result.json',result);status()
        state='COMPLETE'
    except BaseException:save(OUT/'failure.json',dict(error=traceback.format_exc(),phase=phase,commands=commands,physics=sum(counts[:]),updates=calls))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except (OSError,BrokenPipeError):pass
        for proc in processes:
            proc.join(5)
            if proc.is_alive():proc.kill();proc.join(3)
        signal.alarm(0)
        summary=dict(state=state,pid=os.getpid(),elapsed_s=time.time()-start,commands=commands,physics=sum(counts[:]),updates=calls,
                     worker_setup_commands=2 if len(processes)==2 else len(processes),results=results,
                     cached_initial_old275=p['initial_old275_historical'],cached_initial_is_not_new_native=True,
                     all_workers_stopped=all(not x.is_alive() for x in processes),new_RL_updates=0,
                     files={str(f.relative_to(OUT)):sha(f) for f in OUT.rglob('*') if f.is_file() and f.name not in ['main.log','status.json']})
        save(OUT/'summary.json',summary);save(OUT/'status.json',dict(state=state,updated_epoch=time.time(),commands=commands,physics=sum(counts[:]),updates=calls))
        N.capacity(OUT,p['final_bytes'])
    assert state=='COMPLETE'


def audit():
    N.offline();p=read(OUT/'protocol.json');base=verify(p);s=read(OUT/'summary.json')
    assert s['state']=='COMPLETE' and s['all_workers_stopped'] and not (OUT/'audit.json').exists()
    for pid in [s['pid'],*read(OUT/'workers.json')['workers']]:assert not Path('/proc',str(pid)).exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for arm in ARMS:
        q,opt=N.learner(base);groups,old=dataset(base,q,arm)
        trace=fit(q,opt,groups,old,1000,lambda:None,lambda *_:None)
        assert N.equal(trace,N.load_npz(OUT/arm/'trace.npz')['trace'])
        assert parameter_hashes(q,opt)==s['results'][arm]['model']['hashes']
        assert sha(OUT/arm/'candidate.pt.gz')==s['results'][arm]['model']['sha256']
        events=gzip.decompress((OUT/arm/'optimizer_calls.jsonl.gz').read_bytes()).splitlines()
        assert len(events)==4000 and s['updates'][arm]==dict(live=1000,shadow=1000)
    cases=[json.loads(v) for v in (OUT/'cases.jsonl').read_text().splitlines()]
    reps=[json.loads(v) for v in (OUT/'replays.jsonl').read_text().splitlines()]
    for row in cases:
        assert sha(row['path'])==row['sha256'];a=N.load_npz(row['path'])
        assert int(a['native_counts'].sum())==row['physics_steps'] and bool(a['end_flags'][-1,1])==row['success']
    assert s['commands']==len(cases)+len(reps)+s['worker_setup_commands']
    assert s['physics']==sum(v['physics_steps'] for v in [*cases,*reps])
    for f,h in s['files'].items():assert sha(OUT/f)==h,f
    save(OUT/'audit.json',dict(state='PASS',native=0,new_Actor_updates=0,discarded_historical_Adam=2000,
                             both_live_shadow_and_independent_Adam_exact=True,all_raw_counts_and_SHA_exact=True,
                             physics=s['physics'],commands=s['commands'],stable_original_six={k:v['stable_original_six'] for k,v in s['results'].items()}))
    N.manifest(OUT);print('T390_INDEPENDENT_AUDIT_PASS')


if __name__=='__main__':globals()[sys.argv[1]]()
