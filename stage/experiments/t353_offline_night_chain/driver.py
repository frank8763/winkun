"""Runs sealed local protocols without any internet call or model service."""
from night_core import *
OUT=ROOT/'stage/runs/t353_offline_night_chain'
ACTIVE=None


def terminate_active():
    global ACTIVE
    if ACTIVE is not None and ACTIVE.poll() is None:
        try:os.killpg(ACTIVE.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:ACTIVE.wait(1)
        except subprocess.TimeoutExpired:
            try:os.killpg(ACTIVE.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            ACTIVE.wait()


def call(script,args,seconds,log):
    global ACTIVE
    assert seconds>0
    with log.open('xb') as f:
        ACTIVE=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(script),*args],cwd=ROOT,
            env=dict(os.environ,STAGE_PROCESS_CPU='0'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=ACTIVE.wait(timeout=seconds)
            if code:raise subprocess.CalledProcessError(code,str(script))
        except BaseException:terminate_active();raise
        finally:ACTIVE=None


def manifest(directory):
    files=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=K.C.sha(f))
           for f in sorted(directory.rglob('*')) if f.is_file() and f.name!='local_manifest.json']
    save(directory/'local_manifest.json',dict(state='LOCAL_BYTES_SHA_VERIFIED_REMOTE_PENDING',files=files,
        remote_upload_attempted=False,remote_verified=False,offline=True,epoch=time.time()))
    for r in files:assert K.C.sha(ROOT/r['path'])==r['sha256']


def main():
    protocol=read(HERE/'protocol.json');assert not (OUT/'driver_started.json').exists()
    launch=read(OUT/'launch.json');deadline=protocol['all_work_deadline'];start=time.time()
    assert time.time()<deadline and protocol['preflight']['state']=='PASS'
    for f,h in protocol['base']['input_hashes'].items():assert K.C.sha(f)==h,f
    save(OUT/'driver_started.json',dict(pid=os.getpid(),epoch=start,network_required=False))
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('Night stop signal')))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('08:00 absolute stop')));signal.alarm(max(1,int(deadline-start)))
    ledger=dict(state='RUNNING_LOCAL_OFFLINE_CHAIN',pid=os.getpid(),start_epoch=start,network_required=False,remote_backups_pending=True,
        limits=protocol['night_limits'],actual=copy.deepcopy(protocol['prior_T351_actual']),completed=[],current=None,
        preflight_geometry_commands=6,preflight_discarded_Adam=34,entered7or8=False,absolute_stop_epoch=deadline)
    ledger['actual']['commands']+=6
    model=protocol['base']['initial_model'];complete=False
    try:
        for spec in protocol['stages']:
            # Full5400s native+1200s audit+120s local archive must fit, with
            # learning stopping06:30 and all simulation07:15.
            now=time.time()
            if now+5400+1200+120>deadline or now+4200>protocol['learning_deadline']:
                ledger['completed'].append(dict(experiment=spec['id'],state='NOT_STARTED_INSUFFICIENT_WALL_RESERVE'));break
            if any(ledger['actual'][k]+reserve>protocol['night_limits'][k] for k,reserve in [('physics',18_000_000),('commands',1000),('main_wall_s',5400)]):
                ledger['completed'].append(dict(experiment=spec['id'],state='NOT_STARTED_GLOBAL_BUDGET'));break
            p=copy.deepcopy(protocol['base']);p.update(spec)
            directory=ROOT/'stage/runs'/spec['folder'];assert not directory.exists()
            p.update(out=str(directory),initial_model=model,initial_is_T349=model==protocol['base']['initial_model'],
                learning_deadline_epoch=min(now+4200,protocol['learning_deadline']),native_deadline_epoch=min(now+5400,protocol['native_deadline']))
            p['input_hashes'][model]=K.C.sha(model)
            used=K.S.used(ROOT/'stage/runs');free=shutil.disk_usage(ROOT).free
            if used+p['final_bytes']+65536>1024**3 or free<40*1024**3:
                ledger['completed'].append(dict(experiment=spec['id'],state='NOT_STARTED_STORAGE_RESERVE',used=used,own_max=p['final_bytes']));break
            for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
            directory.mkdir();(directory/'episodes').mkdir();save(directory/'protocol.json',p)
            save(directory/'launch.json',dict(epoch=now,night_launch_sha256=K.C.sha(OUT/'launch.json'),local_protocol_sha256=K.C.sha(directory/'protocol.json'),
                source_local_commit=launch['source_local_commit'],source_remote_commit=launch.get('source_remote_commit'),input_model_sha256=K.C.sha(model),offline=True))
            ledger['current']=dict(experiment=spec['id'],directory=str(directory),phase='NATIVE',reserved=dict(physics=18_000_000,commands=1000,wall_s=5400,bytes=p['final_bytes']))
            save(DOC/'NIGHT_LEDGER.json',ledger);save(DOC/'NIGHT_STATUS.json',dict(ledger['current'],updated_epoch=time.time(),driver_pid=os.getpid()))
            try:call(HERE/'phase.py',[str(directory)],min(5400,p['native_deadline_epoch']-time.time())+1,directory/'main.log')
            except BaseException:
                save(directory/'entry_or_timeout_error.json',dict(error=traceback.format_exc(),restart=False));raise
            s=read(directory/'summary.json')
            for pid in [s['pid'],*read(directory/'workers.json')['workers']]:assert not Path('/proc',str(pid)).exists()
            ledger['actual']['physics']+=s['physics_steps'];ledger['actual']['commands']+=s['commands'];ledger['actual']['main_wall_s']+=s['elapsed_s']
            ledger['current']['phase']='AUDIT';save(DOC/'NIGHT_LEDGER.json',ledger)
            if s['state']!='COMPLETE':manifest(directory);raise RuntimeError('Scientific phase failed; raw retained, shared numerical failure not retried')
            try:call(HERE/'audit.py',[str(directory)],min(1200,deadline-time.time()-30),directory/'audit_console.log')
            except BaseException:
                save(directory/'audit_error.json',dict(error=traceback.format_exc(),native_restarted=False));manifest(directory);raise
            assert read(directory/'audit.json')['state']=='PASS';manifest(directory)
            selected=s['selected'];model=s['models'][selected]['path']
            ledger['completed'].append(dict(experiment=spec['id'],state='COMPLETE_LOCAL_AUDIT_PASS_REMOTE_PENDING',summary=s,
                selected_model=model,selected_model_sha256=K.C.sha(model),local_manifest_sha256=K.C.sha(directory/'local_manifest.json')))
            ledger['current']=None;save(DOC/'NIGHT_LEDGER.json',ledger)
            if s['stable_six']:
                # No unimplemented seven/eight metre job may be invented while
                # offline. Preserve passed model and report explicit next gate.
                ledger['six_gate_pass']=True;ledger['seven_eight_unexecuted_reason']='No preimplemented independently verified seven/eight protocol; require next registered source and real capacity.';break
        complete=True
    except BaseException:
        terminate_active();save(DOC/'NIGHT_ERROR.json',dict(error=traceback.format_exc(),epoch=time.time(),current=ledger['current'],training_restarted=False))
    finally:
        terminate_active();signal.alarm(0)
        ledger.update(state='COMPLETE_PREPROGRAMMED_SEQUENCE' if complete else 'STOPPED_WITH_ERROR_OR_CUTOFF',finished_epoch=time.time(),
            all_owned_children_stopped=True,remote_backups_pending=True,global_runs_bytes=K.S.used(ROOT/'stage/runs'),free_bytes=shutil.disk_usage(ROOT).free)
        save(DOC/'NIGHT_LEDGER.json',ledger);save(DOC/'FINAL_REPORT.json',ledger)
        text='# Network-independent Stage night result\n\n'+f"State: {ledger['state']}. Local result/audit retained; remote backup pending. No seven/eight metre training executed.\n\n"
        for record in ledger['completed']:
            text+=f"## {record['experiment']}: {record['state']}\n\n"
            if 'summary' in record:text+='```json\n'+json.dumps(record['summary'],ensure_ascii=False,indent=2)+'\n```\n\n'
        text+='All original failures/partial/model bytes retained. No claim of stable original six metres without full275/54/fresh16/78 gate.\n'
        (DOC/'FINAL_REPORT.md').write_text(text)
        save(DOC/'NIGHT_STATUS.json',dict(state=ledger['state'],updated_epoch=time.time(),pid=os.getpid(),all_children_stopped=True,remote_pending=True))
        print(json.dumps(dict(state=ledger['state'],completed=len(ledger['completed']),remote_pending=True)),flush=True)


def launch():
    p=read(HERE/'protocol.json');assert read(OUT/'preflight.json')['state']=='PASS'
    assert not (OUT/'launch.json').exists() and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    for f,h in p['base']['input_hashes'].items():assert K.C.sha(f)==h,f
    local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');remote=receipt['remote_commit'] if receipt['root_tree']==tree else None
    epoch=time.time();assert epoch+6720<p['all_work_deadline']
    with (DOC/'NIGHT_CONSOLE.log').open('xb') as log:
        proc=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--why=Stage offline preprogrammed bounded night',
            'timeout','--signal=TERM','--kill-after=2s',str(int(p['all_work_deadline']-epoch))+'s','taskset','-c','0',str(ROOT/'run_stage.sh'),str(Path(__file__)),'main'],
            cwd=ROOT,env=dict(os.environ,STAGE_PROCESS_CPU='0'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    launch=dict(pid=proc.pid,epoch=epoch,source_local_commit=local,source_root_tree=tree,source_remote_commit=remote,
        protocol_sha256=K.C.sha(HERE/'protocol.json'),absolute_stop_epoch=p['all_work_deadline'],network_required=False,offline_remote_pending=remote is None)
    save(OUT/'launch.json',launch);save(DOC/'NIGHT_LAUNCH.json',launch)
    print(json.dumps(dict(state='ACTUALLY_LAUNCHED_OFFLINE_NIGHT',pid=proc.pid,deadline_epoch=p['all_work_deadline'],remote_source_verified=remote is not None)))


if __name__=='__main__':{'launch':launch,'main':main}[sys.argv[1]]()
