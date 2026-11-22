"""Only never-executed T699 blocks, same models/tasks/budgets/deadlines.

The original FAILED root, expired plain audit and all consumed work are immutable.
This is an engineering continuation, not a new scientific candidate or training.
"""
from pathlib import Path
import ast,copy,fcntl,gzip,json,os,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stage/experiments/t699_matched_value'))
import t699_common as C
read,save,sha,used,resources=C.read,C.save,C.sha,C.used,C.resources
OLD=C.OUT;OUT=ROOT/'stage/runs/t699_unexecuted_continuation';DOC=C.DOC/'continuation';UNIT='stage-t699-continuation.service'
_outside=None;_old_size=None

def stopped():
    s=read(OLD/'SEALED.json');assert s['all_scientific_processes_stopped'] and s['helper_stopped']
    ids=s['scientific_pids']+[read(OLD/'DRIVER_STARTED.json')['pid']]
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    assert read(OLD/'FINAL_REPORT.json')['state']=='FAILED'
    props=subprocess.check_output(['systemctl','--user','show','stage-t699.service','-p','ActiveState','-p','MainPID'],text=True)
    assert 'MainPID=0' in props and ('ActiveState=inactive' in props or 'ActiveState=failed' in props)

def verify(p,*args,force=False):
    global _outside,_old_size
    root=read(OUT/'protocol.json')
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert time.time()<min(p['absolute_all_stop'],root['absolute_all_stop'])
    assert root['absolute_all_stop']==read(OLD/'protocol.json')['absolute_all_stop']
    assert sha(OLD/'protocol.json')==root['original_protocol_sha256']
    own=used(OUT)
    if _outside is None or force:
        actual=used(ROOT/'stage/runs')-own
        if _outside is not None:assert actual==_outside,'Another Stage run changed while exclusive audit/native was active'
        _outside=actual
        size=used(OLD)
        if _old_size is not None:assert size==_old_size,'Completed failed root changed'
        _old_size=size
    assert own+_outside<root['global_bytes'] and own+_old_size<root['combined_original_bytes']
    assert own<root['max_bytes']
    if p.get('out')!=str(OUT):assert used(p['out'])<root['child_bytes']
    import shutil
    assert shutil.disk_usage(ROOT).free>40*2**30
    resources()
    for f,h in root['inputs'].items():
        q=Path(f);assert [q.stat().st_size,q.stat().st_mtime_ns]==root['input_stats'][f],f
        if force:assert sha(q)==h,f
    for f,h in root['continuation_inputs'].items():assert sha(f)==h,f

def scoped_signal(pid,sig):
    try:
        assert pid>1 and os.getpgid(pid)==pid and '/'+UNIT in Path('/proc',str(pid),'cgroup').read_text()
        os.killpg(pid,sig)
    except (FileNotFoundError,ProcessLookupError):pass

def helper():
    p=read(OUT/'protocol.json');os.sched_setaffinity(0,{15})
    cg=next(x.split(':',2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0:'))
    assert cg.endswith('/'+UNIT);cg=Path('/sys/fs/cgroup'+cg)
    save(DOC/'AFFINITY_GUARD.json',dict(state='ACTIVE_RESOURCE_ONLY',pid=os.getpid(),cgroup=str(cg)))
    while time.time()<p['absolute_all_stop'] and not (OUT/'HELPER_STOP').exists():
        status=C.read_status(OUT/'status.json')
        try:resources();clear=True
        except AssertionError as error:
            if not isinstance(error.args[0],tuple) or not str(error.args[0][0]).startswith('Phase physical CPU overlap'):raise
            clear=False
        if status.get('child_pid'):scoped_signal(status['child_pid'],signal.SIGCONT if clear else signal.SIGSTOP)
        for pid in (cg/'cgroup.procs').read_text().split():
            try:
                for task in Path('/proc',pid,'task').iterdir():
                    tid=int(task.name);mask=os.sched_getaffinity(tid)
                    if not mask<=set(range(8,16)):os.sched_setaffinity(tid,mask&set(range(8,16)) or {15})
            except (FileNotFoundError,ProcessLookupError):pass
        time.sleep(.5)
    save(DOC/'AFFINITY_GUARD.json',dict(state='ENDED',pid=os.getpid(),epoch=time.time()))

def child(action,name):
    import t699_native as N
    N.OUT=OUT;N.DOC=DOC;N.verify=verify
    # Keep the original canonical execution, native inspector, model loader,
    # task list and all exact repeats. Replace only the storage/manifest guard.
    if action=='audit':
        lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:N.audit(name)
        finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    else:N.evaluate(name)

def register():
    stopped();resources();assert read(C.DOC/'AUDIT_VERIFICATION_RED.json')['state']=='RED'
    assert read(C.DOC/'AUDIT_VERIFICATION_GREEN.json')['state']=='GREEN'
    assert not OUT.exists();old=read(OLD/'protocol.json');raw=json.loads((OLD/'protocol.json').read_text())
    completed=read(OLD/'LEDGER.json')['totals'];assert completed['physics']==11001515 and completed['commands']==514
    assert not any((OLD/('known_'+n)).exists() for n in old['final_models'] if n!='plain')
    assert not any((OLD/('new_'+n)).exists() for n in old['transfer_models'])
    p=copy.deepcopy(raw)
    p.update(id='T699_ENGINEERING_CONTINUATION_ONLY_NEVER_EXECUTED_BLOCKS',out=str(OUT),original_failed_root=str(OLD),
        original_protocol_sha256=sha(OLD/'protocol.json'),original_failed_audit_remains_FAILED_EXPIRED=True,
        max_physics=old['max_physics']-completed['physics'],max_commands=old['max_commands']-completed['commands'],
        max_Actor_live=0,max_Actor_shadow=0,max_Actor_independent=0,new_Value=0,new_RL=0,
        combined_original_bytes=old['max_bytes'],max_bytes=old['max_bytes']-used(OLD),
        original_registration_epoch=old['registration_epoch'],continuation_registration_epoch=time.time(),
        priority_change_declared_before_any_replay_native=True,
        schedule=[['known','replay_1736']]+[['new',n] for n in old['transfer_models']]+[['known',n] for n in ['replay_108','replay_434','replay_1736_seed2']],
        no_guarantee_all_remaining_blocks_fit_wall_deadline=True)
    # No original absolute deadline, per-child wall cap, learned parameter,
    # sampling/tuning choice, reset or physical/command envelope is increased.
    assert time.time()+old['known_main_seconds']+old['known_audit_seconds']+60<old['absolute_all_stop']
    assert time.time()+old['known_main_seconds']+10<old['absolute_native_stop']
    l=read(C.WEEKEND/'LEDGER.json');assert not any(j['id']=='T699_CONTINUATION' for j in l['jobs'])
    assert l['physics']+p['max_physics']<1500000000 and l['commands']+p['max_commands']<45000
    assert used(ROOT/'stage/runs')+p['max_bytes']<8*2**30
    OUT.mkdir();(OUT/'fit').mkdir()
    for filename in ['summary.json','audit.json']:(OUT/'fit'/filename).write_bytes((OLD/'fit'/filename).read_bytes())
    save(OUT/'REUSED_FIT.json',dict(original_summary_sha256=sha(OLD/'fit/summary.json'),original_audit_sha256=sha(OLD/'fit/audit.json'),new_optimizer=0,new_Actor=0,all_model_paths_unchanged=True))
    extra=set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'AUTHORIZATION.json',C.DOC/'AUDIT_VERIFICATION_RED.json',C.DOC/'AUDIT_VERIFICATION_GREEN.json'}
    p['continuation_inputs']={str(f):sha(f) for f in sorted(extra)}
    blob=json.dumps(p,separators=(',',':'))+'\n';assert len(blob.encode())<2**21
    (HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob)
    l['jobs'].append(dict(id='T699_CONTINUATION',state='RESERVED',out=str(OUT),max_physics=p['max_physics'],max_commands=p['max_commands'],Actor_each_cap=0,Value_each_cap=0,
        original_T699_root_absolute_unchanged=p['absolute_all_stop'],completed_children=[]));save(C.WEEKEND/'LEDGER.json',l)
    save(DOC/'REGISTERED.json',dict(state='REGISTERED_NOT_LAUNCHED',epoch=time.time(),absolute_all_stop=p['absolute_all_stop'],max_physics=p['max_physics'],max_commands=p['max_commands'],new_Actor=0))
    verify(dict(read(OUT/'protocol.json'),out=str(OUT)),force=True);print('CONTINUATION_REGISTERED_ONLY_UNEXECUTED')

def launch():
    stopped();p=read(OUT/'protocol.json');verify(p,force=True);assert not (OUT/'launch.json').exists()
    r=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert r['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time()-3)
    cmd=['systemd-run','--user','--unit=stage-t699-continuation','--collect','--property=Type=exec','--property=CPUQuota=800%',
        '--property=CPUWeight=10','--property=Nice=19','--property=AllowedCPUs=8 9 10 11 12 13 14 15','--property=MemoryMax=8G',
        '--property=MemorySwapMax=0','--property=KillMode=control-group','--property=TimeoutStopSec=5',f'--property=RuntimeMaxSec={seconds}',
        f'--property=WorkingDirectory={ROOT}',f'--property=StandardOutput=append:{OUT}/driver.log','--property=StandardError=inherit',
        '--setenv=CUDA_VISIBLE_DEVICES=','systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage T699 continuation',
        '--why=Only never-executed comparison blocks',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'driver']
    c=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True);assert c.returncode==0,c.stderr
    save(OUT/'launch.json',dict(epoch=time.time(),unit=UNIT,remote_commit=r['remote_commit']));print(c.stderr)

def account(items,state):
    totals=dict(physics=sum(i.get('physics',0) for i in items),commands=sum(i.get('commands',0) for i in items),
        science_seconds=sum(i['elapsed_seconds'] for i in items if not i['audit']),audit_seconds=sum(i['elapsed_seconds'] for i in items if i['audit']))
    save(OUT/'LEDGER.json',dict(children=items,totals=totals,new_Actor=0,new_optimizer=0))
    l=read(C.WEEKEND/'LEDGER.json');j=next(j for j in l['jobs'] if j['id']=='T699_CONTINUATION');before=j.get('accounted',{})
    for k,v in totals.items():assert v>=before.get(k,0);l[k]+=v-before.get(k,0)
    j.update(state=state,accounted=totals,completed_children=[i['name'] for i in items]);save(C.WEEKEND/'LEDGER.json',l)

def driver():
    p=read(OUT/'protocol.json');verify(p,force=True);assert not (OUT/'DRIVER_STARTED.json').exists()
    os.sched_setaffinity(0,{15});save(OUT/'DRIVER_STARTED.json',dict(pid=os.getpid(),epoch=time.time()))
    log=(OUT/'helper.log').open('xb');h=subprocess.Popen(['/usr/bin/python3',str(HERE/'run.py'),'helper'],stdout=log,stderr=log)
    ids=[h.pid];items=[];error=None;missing=[]
    try:
        until=time.time()+5
        while not (DOC/'AFFINITY_GUARD.json').exists() and time.time()<until:time.sleep(.05)
        assert read(DOC/'AFFINITY_GUARD.json')['pid']==h.pid
        for i,(suite,name) in enumerate(p['schedule']):
            main=p['known_main_seconds'] if suite=='known' else p['new_main_seconds'];audit=p['known_audit_seconds'] if suite=='known' else p['new_audit_seconds']
            if time.time()+main+audit+60>=p['absolute_all_stop'] or time.time()+main+10>=p['absolute_native_stop']:
                missing=p['schedule'][i:];break
            arm=suite+'_'+name;folder=OUT/arm;folder.mkdir();(folder/'episodes').mkdir()
            jobs=p['evaluation_jobs']+p['screen_jobs'] if suite=='known' else p['new_jobs']+p['new_exact_jobs']
            c=dict(arm=arm,suite=suite,model_name=name,out=str(folder),_root_sha256=sha(OUT/'protocol.json'),
                main_seconds=main,audit_seconds=audit,child_bytes=p['child_bytes'],max_physics=sum(j['case']['step_cap'] for j in jobs),
                max_commands=len(jobs)+14,absolute_native_stop=min(time.time()+main,p['absolute_native_stop']),
                absolute_all_stop=min(time.time()+main+audit+60,p['absolute_all_stop']))
            save(folder/'protocol.json',c)
            for action,seconds in [('evaluate',main),('audit',audit)]:
                assert h.poll() is None
                with (folder/(action+'.log')).open('xb') as f:
                    child=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'run.py'),action,arm],cwd=ROOT,stdout=f,stderr=f,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
                    ids.append(child.pid);start=time.time();save(OUT/'status.json',dict(state='ACTIVE',action=action,child=arm,child_pid=child.pid,epoch=start,absolute_action_stop=start+seconds))
                    try:child.wait(timeout=seconds);assert child.returncode==0,(arm,action,child.returncode)
                    except BaseException:
                        scoped_signal(child.pid,signal.SIGCONT);scoped_signal(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=3)
                        except subprocess.TimeoutExpired:scoped_signal(child.pid,signal.SIGKILL);child.wait()
                        raise
                    finally:
                        item=dict(name=arm+'_'+action,pid=child.pid,audit=action=='audit',returncode=child.returncode,elapsed_seconds=time.time()-start)
                        if action=='evaluate':
                            s=read(folder/'summary.json') if (folder/'summary.json').exists() else {}
                            item.update(physics=s.get('physics',c['max_physics']),commands=s.get('commands',c['max_commands']),counts_conservative_upper_if_summary_missing=not bool(s))
                        items.append(item);account(items,'ACTIVE')
            assert read(folder/'audit.json')['state']=='PASS'
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,no_restart=True,all_raw_preserved=True))
    finally:
        (OUT/'HELPER_STOP').touch()
        try:h.wait(timeout=5)
        except subprocess.TimeoutExpired:h.terminate();h.wait(timeout=3)
        log.close()
        for f in OUT.glob('*/worker_generations.json'):
            for g in read(f):ids.extend([g['parent'],*g['workers']])
        alive=[pid for pid in set(ids) if Path('/proc',str(pid)).exists()]
        state='FAILED' if error or alive else 'PARTIAL_TIME_ADMISSION' if missing else 'COMPLETE'
        account(items,state);report=dict(state=state,error=error,missing_unexecuted_blocks=missing,scientific_pids=sorted(set(ids)),all_scientific_processes_stopped=not alive,
            helper_pid=h.pid,helper_stopped=h.poll() is not None,original_T699_stays_FAILED=True,old_plain_audit_stays_FAILED_EXPIRED=True,new_Actor=0,new_optimizer=0,retained='T698',epoch=time.time())
        save(OUT/'SEALED.json',report);save(OUT/'FINAL_REPORT.json',report);save(DOC/'FINAL_REPORT.json',report);save(OUT/'status.json',report)

if __name__=='__main__':
    mode=sys.argv[1]
    if mode in ['evaluate','audit']:child(mode,sys.argv[2])
    else:{'register':register,'launch':launch,'driver':driver,'helper':helper}[mode]()
