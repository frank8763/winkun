"""Isolated native worlds share policy bytes, never mutable simulator state."""
from t400_contract import *

def worker():
    N.offline();p=read(OUT/'protocol.json');base=verify(p);forbid_learning()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    concurrency,index=map(int,sys.argv[2:4]);assert concurrency in p['concurrency'] and 0<=index<concurrency
    folder=OUT/str(concurrency)/str(index);N.configure(folder)
    deadline=read(OUT/'main_started.json')['native_deadline'];budget=len(range(index,4,concurrency))*p['physics_per_job'];physics=0;s=w=None;cases=[];phase='SETUP';last=0.;status='FAILED';command_count=0
    started=time.perf_counter();q=frozen(base,p);fixed=digest(q.state_dict())
    def guard(physical=False):
        nonlocal last
        C.check_stop();assert time.time()<deadline-2
        if not physical and time.time()-last>2:
            last=time.time();N.capacity(OUT,p['final_bytes'])
            save(folder/'status.json',dict(state=phase,pid=os.getpid(),updated_epoch=last,physics=physics,budget=budget))
    def charge():
        nonlocal physics
        physics+=1;assert physics<=budget
    counts={'actor_s':0.,'native_checked_s':0.};original_actor=q.step
    def measured_actor(*a,**k):
        t=time.perf_counter()
        try:return original_actor(*a,**k)
        finally:counts['actor_s']+=time.perf_counter()-t
    q.step=measured_actor
    C.install_stop_handlers()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(C.StageStop('T400 worker native deadline')));signal.alarm(max(1,int(deadline-time.time())))
    try:
        m,native,create,w,_=N.S.P.C.B.setup(folder/'native',base);command_count+=1;original=copy.deepcopy(m.ACTIVE);original_native=native.step_checked
        def measured_native(*a,**k):
            t=time.perf_counter()
            try:return original_native(*a,**k)
            finally:counts['native_checked_s']+=time.perf_counter()-t
        native.step_checked=measured_native;C.install_stop_handlers()
        setup_s=time.perf_counter()-started;save(folder/'ready.json',dict(pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)),setup_s=setup_s))
        while not (OUT/str(concurrency)/'GO.json').exists():guard();time.sleep(.005)
        loop_start=time.perf_counter();sample_s=save_s=0.
        for job_index in range(index,4,concurrency):
            guard();phase='JOB_'+str(job_index);job=p['jobs'][job_index]['job'];command_count+=1
            q.sample=False;q.reset_noise();m.ACTIVE=copy.deepcopy(original);m.ACTIVE['circle_shift_x_m']=job['circle_shift_x_m']
            s=C.Session(job,native,create,w,guard,charge,q);C.install_stop_handlers()
            t=time.perf_counter()
            for _ in range(p['holds']):s.step(owner=0);assert not s.done(),'Unexpected task terminal before benchmark cut'
            duration=time.perf_counter()-t;sample_s+=duration
            a=payload(s);assert int(a['native_counts'].sum())==p['physics_per_job'] and not a['end_flags'].any()
            t=time.perf_counter();f=folder/('job'+str(job_index)+'.npz');N.write_npz(f,N.S.packed(a),p['full_bytes']);save_s+=time.perf_counter()-t
            cases.append(dict(id=job_index,path=str(f),sha256=sha(f),array_digest=digest(a),physics=s.steps,sample_s=duration,administrative_benchmark_censor=True,navigation_success_rate_not_applicable=True,policy_hash=q.phash()))
            assert digest(q.state_dict())==fixed;s.close();s=None
        status='COMPLETE';loop_s=time.perf_counter()-loop_start
    except BaseException:
        loop_s=None;setup_s=None;sample_s=save_s=None
        if s is not None and s.rows:
            try:N.write_npz(folder/'partial.npz',N.S.packed(payload(s)),p['full_bytes'])
            except BaseException:save(folder/'partial_save_error.json',dict(error=traceback.format_exc()))
        save(folder/'failure.json',dict(error=traceback.format_exc(),phase=phase,physics=physics))
    finally:
        if s:s.close()
        if w:w.close()
        signal.alarm(0)
        save(folder/'summary.json',dict(state=status,pid=os.getpid(),concurrency=concurrency,index=index,physics=physics,commands=command_count,cases=cases,setup_s=setup_s,loop_wall_s=loop_s,sample_s=sample_s,save_s=save_s,component_timing=counts,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,cpu_affinity=sorted(os.sched_getaffinity(0)),native_closed=True,Actor_updates=0,optimizer_updates=0,backward_calls=0,policy_hash=q.phash()))
    assert status=='COMPLETE'

def main():
    N.offline();p=read(OUT/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'summary.json').exists()
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time();deadline=started+p['main_seconds'];save(OUT/'main_started.json',dict(pid=os.getpid(),epoch=started,native_deadline=deadline))
    outside=N.K.S.used(ROOT/'stage/runs')-N.K.S.used(OUT);children=[];logs=[];pids=[];phases=[];status='FAILED'
    def guard():
        C.check_stop();assert time.time()<deadline-2;N.capacity(OUT,p['final_bytes'])
        assert N.K.S.used(ROOT/'stage/runs')-N.K.S.used(OUT)==outside
    C.install_stop_handlers()
    try:
        for concurrency in p['concurrency']:
            guard();folder=OUT/str(concurrency);folder.mkdir();begin=time.perf_counter();children=[];logs=[]
            for index in range(concurrency):
                sub=folder/str(index);sub.mkdir();log=(sub/'stdout.log').open('wb');logs.append(log)
                child=subprocess.Popen(['taskset','-c',str(index),str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'worker',str(concurrency),str(index)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                children.append(child);pids.append(child.pid)
            save(OUT/'workers.json',dict(parent=os.getpid(),workers=pids))
            while not all((folder/str(i)/'ready.json').exists() for i in range(concurrency)):
                guard();assert all(child.poll() is None for child in children),'Worker failed before READY';time.sleep(.01)
            initialization_s=time.perf_counter()-begin;go=time.perf_counter();save(folder/'GO.json',dict(epoch=time.time()))
            while any(child.poll() is None for child in children):
                guard();save(OUT/'status.json',dict(state='CONCURRENCY_'+str(concurrency),pid=os.getpid(),updated_epoch=time.time(),workers=[c.pid for c in children],deadline=deadline));time.sleep(.03)
            assert all(c.wait()==0 for c in children)
            for log in logs:log.close()
            logs=[];values=[read(folder/str(i)/'summary.json') for i in range(concurrency)]
            assert all(v['state']=='COMPLETE' for v in values)
            cases=sorted([r for v in values for r in v['cases']],key=lambda r:r['id'])
            assert [r['id'] for r in cases]==list(range(4)) and sum(v['physics'] for v in values)==20000
            row=dict(concurrency=concurrency,total_wall_s=time.perf_counter()-begin,initialization_wall_s=initialization_s,after_GO_wall_s=time.perf_counter()-go,physics=20000,commands=sum(v['commands'] for v in values),cases=cases,worker_summaries=values)
            if phases:assert [r['array_digest'] for r in cases]==[r['array_digest'] for r in phases[0]['cases']],'Concurrent trajectory difference'
            phases.append(row);save(OUT/'phases.json',phases)
        status='COMPLETE'
    except BaseException:save(OUT/'failure.json',dict(error=traceback.format_exc()))
    finally:
        for child in children:
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM);time.sleep(.1);os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            child.wait()
        for log in logs:log.close()
        assert all(not Path('/proc',str(pid)).exists() for pid in pids)
        all_summaries=[read(f) for f in OUT.glob('*/*/summary.json')];physics=sum(v['physics'] for v in all_summaries);commands=sum(v['commands'] for v in all_summaries)
        summary=dict(state=status,pid=os.getpid(),elapsed_s=time.time()-started,physics=physics,commands=commands,phases=phases,workers=pids,all_workers_stopped=True,all_native_sessions_closed=True,Actor_updates=0,optimizer_updates=0,backward_calls=0,old275=None,main54=None,fresh16=None,reload78=None,full_six_passed=False,seven_eight_entered=False,model_promoted=False,administrative_benchmark_censors=sum(len(v['cases']) for v in all_summaries),navigation_success_rate=None,policy_hash=p['policy_hash'])
        save(OUT/'summary.json',summary);save(OUT/'status.json',dict(state=status,updated_epoch=time.time(),physics=physics,commands=commands,Actor_updates=0));N.manifest(OUT)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert status=='COMPLETE'

def audit():
    N.offline();p=read(OUT/'protocol.json');base=verify(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists()
    assert all(not Path('/proc',str(pid)).exists() for pid in [s['pid'],*s['workers']])
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);forbid_learning();q=frozen(base,p);fixed=digest(q.state_dict());reference={};physics=commands=0
    for phase in s['phases']:
        n=phase['concurrency'];assert n in [1,2,4];commands+=n
        for row in phase['cases']:
            assert sha(row['path'])==row['sha256'];a=N.S.arrays(row['path']);audit_array(q,a,0)
            assert len(a['states'])==50 and not a['end_flags'].any() and a['native_counts'].sum()==5000
            assert digest(a)==row['array_digest'];physics+=5000;commands+=1
            if row['id'] not in reference:reference[row['id']]=row['array_digest']
            else:assert reference[row['id']]==row['array_digest']
        for worker in phase['worker_summaries']:
            assert worker['cpu_affinity']==[worker['index']]
    assert physics==s['physics']==60000 and commands==s['commands']==19 and commands+2<=p['max_commands']
    assert s['administrative_benchmark_censors']==12 and s['navigation_success_rate'] is None and digest(q.state_dict())==fixed
    assert s['Actor_updates']==s['optimizer_updates']==s['backward_calls']==0
    for row in read(OUT/'local_manifest.json')['files']:assert sha(ROOT/row['path'])==row['sha256']
    best=min(s['phases'],key=lambda row:row['total_wall_s']);serial=s['phases'][0]
    save(OUT/'audit.json',dict(state='PASS',native=0,optimizer=0,backward=0,weights_unchanged=True,all_cross_concurrency_arrays_history_hidden_rewards_exact=True,physics=physics,commands=commands,fastest_tested_concurrency=best['concurrency'],end_to_end_speedup_over_serial=serial['total_wall_s']/best['total_wall_s'],timing_single_fixed_order_small_workload=True,full_six_passed=False));N.manifest(OUT)
    print('T400_ZERO_NATIVE_ZERO_OPTIMIZER_AUDIT_PASS')

if __name__=='__main__':globals()[sys.argv[1]]()
