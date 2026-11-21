"""One zero-native, zero-update audit recovery within original deadlines."""
from pathlib import Path
import json,os,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t694r_audit_recovery';DOC=ROOT/'stage/docs/weekend_20261120_23/t694r'
from status_io import save
import t694r_wait as Wait
read=lambda f:json.loads(Path(f).read_text())
def sha(f):
    import hashlib
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def register():
    assert not OUT.exists();p=read(HERE/'protocol.json');assert not p['authorization_approved']
    assert read(DOC/'PREFLIGHT.json')['state']=='PASS_GUARD_SCHEDULE_ONLY_ZERO_NATIVE_ZERO_UPDATE'
    Wait.Raw.resources();auth=read(p['weekend_authorization']);l=read(DOC.parent/'LEDGER.json')
    assert auth['approved'] and not any(j['id']=='T694R' for j in l['jobs'])
    for f,h in p['inputs'].items():assert sha(f)==h,f
    assert time.time()+p['audit_seconds']+90<p['absolute_all_stop']
    assert l['audit_seconds']+p['audit_seconds']<=auth['max_audit_seconds']
    assert l['science_seconds']+p['preflight_seconds']<=auth['max_science_seconds']
    l['science_seconds']+=p['preflight_seconds'];l['jobs'].append(dict(id='T694R',state='RESERVED',out=str(OUT),max_physics=0,max_commands=0,
        main_seconds=0,audit_seconds=p['audit_seconds'],preflight_seconds=p['preflight_seconds'],Actor_each_cap=0,Value_each_cap=0,
        scope='ONLY_UNCOMMITTED_ORIGINAL_NATIVE_AUDIT_NO_FIT_OR_NATIVE_REEXECUTION'))
    save(DOC.parent/'LEDGER.json',l);p.update(authorization_approved=True,registration_epoch=time.time());OUT.mkdir()
    save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    print('T694R_REGISTERED_NOT_LAUNCHED')


def launch():
    p=read(OUT/'protocol.json');assert p['authorization_approved'] and not (OUT/'launch.json').exists();Wait.Raw.resources()
    r=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert r['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    left=int(p['absolute_all_stop']-time.time()-2);assert left>p['audit_seconds']+60
    cmd=['systemd-run','--user','--unit=stage-t694r','--collect','--property=Type=exec','--property=CPUQuota=400%','--property=CPUWeight=10','--property=Nice=19',
        '--property=AllowedCPUs=8 10 12 14','--property=MemoryMax=4G','--property=MemorySwapMax=0','--property=KillMode=control-group','--property=TimeoutStopSec=5',
        f'--property=RuntimeMaxSec={left}',f'--property=WorkingDirectory={ROOT}',f'--property=StandardOutput=append:{OUT}/driver.log',
        '--property=StandardError=inherit','--setenv=CUDA_VISIBLE_DEVICES=',
        'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage T694R','--why=Unfinished audit only',str(ROOT/'run_stage.sh'),str(HERE/'control.py'),'driver']
    r2=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True);assert r2.returncode==0,r2.stderr
    save(OUT/'launch.json',dict(epoch=time.time(),unit='stage-t694r',source_remote=r['remote_commit']));print(r2.stderr.strip())


def driver():
    p=read(OUT/'protocol.json');assert p['authorization_approved'] and not (OUT/'DRIVER_STARTED.json').exists()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    Wait.resources();os.sched_setaffinity(0,{14});save(OUT/'DRIVER_STARTED.json',dict(pid=os.getpid(),epoch=time.time(),science_network=False))
    log=(OUT/'affinity.log').open('xb');helper=subprocess.Popen(['/usr/bin/python3',str(HERE/'affinity_guard.py')],cwd=ROOT,stdout=log,stderr=log)
    c=None;error=None;elapsed=0.
    try:
        until=time.time()+5
        while not (DOC/'AFFINITY_GUARD.json').exists() and time.time()<until:assert helper.poll() is None;time.sleep(.05)
        assert read(DOC/'AFFINITY_GUARD.json')['pid']==helper.pid
        with (OUT/'audit.log').open('xb') as f:
            c=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'recovery.py')],cwd=ROOT,stdout=f,stderr=f,start_new_session=True,
                               env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
            save(OUT/'status.json',dict(state='ACTIVE',action='audit',child_pid=c.pid,epoch=time.time()));started=time.time()
            try:c.wait(timeout=p['audit_seconds']);assert c.returncode==0,c.returncode
            except BaseException:
                try:os.killpg(c.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:c.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
                raise
            finally:elapsed=time.time()-started
        assert read(OUT/'audit.json')['state']=='PASS'
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,all_original_evidence_preserved=True,no_restart=True))
    ids=[] if c is None else [c.pid];gone=all(not Path('/proc',str(pid)).exists() for pid in ids)
    save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=gone,scientific_pids=ids,epoch=time.time()))
    (OUT/'AFFINITY_STOP').touch()
    try:helper.wait(timeout=5)
    except subprocess.TimeoutExpired:helper.terminate();helper.wait(timeout=3)
    log.close();report=dict(state='FAILED' if error else 'COMPLETE',error=error,all_scientific_processes_stopped=gone,helper_pid=helper.pid,
        helper_stopped=helper.poll() is not None,physics=0,commands=0,science_seconds=0,audit_seconds=elapsed,
        Actor_calls=dict(Actor_live=0,Actor_shadow=0,Actor_independent=0),Value_calls=dict(Value_live=0,Value_shadow=0,Value_independent=0),
        audit=read(OUT/'audit.json') if (OUT/'audit.json').exists() else None,original_failed_root_preserved=True,epoch=time.time())
    save(OUT/'FINAL_REPORT.json',report);save(DOC/'FINAL_REPORT.json',report);save(OUT/'status.json',dict(state=report['state'],all_science_stopped=gone,epoch=time.time()))
    l=read(DOC.parent/'LEDGER.json');j=next(j for j in l['jobs'] if j['id']=='T694R');assert j['state']=='RESERVED'
    l['audit_seconds']+=elapsed;j.update(state=report['state'],actual_physics=0,actual_commands=0,actual_science_seconds=0,actual_audit_seconds=elapsed,Actor_calls=report['Actor_calls'])
    save(DOC.parent/'LEDGER.json',l);print(report['state'])


if __name__=='__main__':{'register':register,'launch':launch,'driver':driver}[sys.argv[1]]()
