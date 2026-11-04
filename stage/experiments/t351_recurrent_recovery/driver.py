"""Unique finalizer; never starts overnight or another navigation run."""
from t351_core import *
import subprocess, signal


def launch():
    p=read(HERE/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS'
    assert not any((OUT/f).exists() for f in ['launch.json','status.json','summary.json'])
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    epoch=time.time();assert epoch+p['all_work_seconds']<p['absolute_stop_epoch']
    with (DOC/'T351_DRIVER_CONSOLE.log').open('xb') as log:
        proc=subprocess.Popen(['systemd-inhibit','--what=sleep','--mode=block','--why=Stage bounded six-metre recovery gate before overnight',
            'timeout','--signal=TERM','--kill-after=1s',str(p['all_work_seconds'])+'s','taskset','-c','0',str(ROOT/'run_stage.sh'),str(Path(__file__)),'finalize'],
            cwd=ROOT,env=dict(os.environ,STAGE_PROCESS_CPU='0'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=proc.pid,launch_epoch=epoch,native_deadline_epoch=epoch+p['wall_seconds'],
        learning_deadline_epoch=epoch+p['wall_seconds']-p['final_reserve_seconds'],all_work_deadline_epoch=epoch+p['all_work_seconds'],
        source_remote_commit=receipt['remote_commit'],root_tree=receipt['root_tree'],protocol_sha256=K.C.sha(HERE/'protocol.json'),overnight_started=False))
    print(json.dumps(dict(state='LAUNCHED_BOUNDED_SIX_NOT_OVERNIGHT',pid=proc.pid,native_deadline_epoch=epoch+p['wall_seconds'])))


def finalize():
    for _ in range(40):
        if (OUT/'launch.json').exists():break
        time.sleep(.1)
    p=read(HERE/'protocol.json');launch=read(OUT/'launch.json');deadline=launch['all_work_deadline_epoch']
    report=dict(state='DRIVER_RUNNING',pid=os.getpid(),overnight_started=False,entered7or8=False)
    def state(phase):save(DOC/'T351_DRIVER.json',dict(report,phase=phase,updated_epoch=time.time(),deadline_epoch=deadline))
    def call(script,args,limit,logpath=None):
        remaining=min(limit,deadline-time.time()-10);assert remaining>0
        command=['taskset','-c','0',str(ROOT/'run_stage.sh'),str(script),*args]
        log=None if logpath is None else logpath.open('xb')
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT if log else None,start_new_session=True)
        try:
            code=child.wait(timeout=remaining)
            if code:raise subprocess.CalledProcessError(code,command)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(1)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise
        finally:
            if log:log.close()
    try:
        state('MAIN');call(HERE/'run.py',[],p['wall_seconds']+2,OUT/'main.log')
        s=read(OUT/'summary.json');report['summary']=s
        for pid in [s['pid'],*read(OUT/'workers.json')['workers']]:assert not Path('/proc',str(pid)).exists()
        state('AUDIT')
        try:call(HERE/'audit.py',[],p['audit_seconds'],OUT/'audit_console.log')
        except BaseException:save(DOC/'T351_AUDIT_ERROR.json',dict(error=traceback.format_exc(),new_training_restarted=False))
        # Nothing under OUT is modified from this point onward.
        state('RAW_BACKUP');receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T351.json'
        for attempt in range(2):
            if receipt.exists():break
            try:call(ROOT/'stage/experiments/t260_distance_archive/small_parts.py',['T351'],600)
            except BaseException:save(DOC/f'T351_BACKUP_TRANSPORT_ERROR_{attempt+1}.json',dict(error=traceback.format_exc(),native_restarted=False))
        backed=receipt.exists() and read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
        audited=(DOC/'T351_AUDIT.json').exists() and read(DOC/'T351_AUDIT.json')['state']=='PASS'
        report.update(state='COMPLETE' if s['state']=='COMPLETE' and audited and backed else 'INCOMPLETE_OR_FAILED',
            audit_pass=audited,all_raw_remote_readback_pass=backed,stable_six=bool(s['stable_six'] and audited and backed),
            night_gate_pass=bool(s['stable_six'] and audited and backed),finished_epoch=time.time())
        save(DOC/'T351_FINAL_REPORT.json',report)
        text=f"# T351 actual result\n\nNative state: {s['state']}. Commands {s['commands']}; physical steps {s['physics_steps']}; live/shadow updates {s['updates']}.\n\nResult (null means unexecuted):\n\n```json\n{json.dumps(s['result'],ensure_ascii=False,indent=2)}\n```\n\nIndependent audit={audited}; complete raw remote byte readback={backed}; full-six gate={report['night_gate_pass']}. Overnight/7m/8m were NOT started by this driver. No automatic continuation of failed budget. Stage only; Phase isolated.\n"
        (DOC/'T351_FINDINGS.md').write_text(text)
        status=ROOT/'stage/docs/RESEARCH_STATUS.md'
        status.write_text(f"2026-11-04 T351 evening bounded six-metre job ended: native={s['state']}, audit={audited}, fullrawremote={backed}, commands={s['commands']}, physics={s['physics_steps']}, live/shadow={s['updates']}, fullsix={report['night_gate_pass']}. Overnight/7/8 not started. Initial old275 is preexisting research baseline, not all failure attributable to this RL. See review_20261104/T351_FINAL_REPORT.json/T351_FINDINGS.md; missing/null is unexecuted, no restart/seed/grid extension. Stage/Phase fully isolated.\n\n"+status.read_text())
        state('SOURCE_SYNC')
        names=['stage/docs/RESEARCH_STATUS.md',*subprocess.check_output(['git','ls-files','--others','--exclude-standard','stage/docs/review_20261104/T351*','stage/docs/long_distance_20261101/*T351*'],cwd=ROOT,text=True).splitlines()]
        names=[n for n in names if not n.endswith('.log')]
        subprocess.run(['git','add',*names],cwd=ROOT,check=True)
        subprocess.run(['git','commit','-m','Archive bounded T351 recurrent six-metre experiment and actual gate result'],cwd=ROOT,check=True)
        call(ROOT/'stage/experiments/t224_archive/sync_source.py',[],300)
        # This small finalizer receipt is intentionally not claimed as part of
        # the prior raw archive or already committed source tree.
        save(DOC/'T351_DRIVER_FINISHED.json',dict(state='FINALIZER_FINISHED',finished_epoch=time.time(),source_receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json'),night_gate_pass=report['night_gate_pass']))
    except BaseException:
        save(DOC/'T351_DRIVER_ERROR.json',dict(error=traceback.format_exc(),phase=read(DOC/'T351_DRIVER.json').get('phase'),native_restart=False,overnight_started=False));raise


if __name__=='__main__':{'launch':launch,'finalize':finalize}[sys.argv[1]]()
