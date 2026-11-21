"""Unique bounded launch and independent post-collection audit."""
from pathlib import Path
import json,os,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t685_fresh_proxy_validity';DOC=ROOT/'stage/docs/weekend_20261120_23/t685'
from status_io import save
import t685_wait as Wait
def read(f):return json.loads(Path(f).read_text())
def sha(f):
    import hashlib
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()

def prepare():
    import t685_prepare as P
    P.main()


def register():
    assert not OUT.exists();p=read(HERE/'protocol.json')
    assert not p['authorization_approved'] and read(DOC/'PREFLIGHT.json')['state']=='PASS_ZERO_NATIVE_ZERO_OPTIMIZER'
    a=read(p['weekend_authorization']);assert a['approved'] and time.time()+1800<a['absolute_all_stop']
    assert p['new_Actor']==p['new_Value']==p['new_optimizer']==0
    Wait.Raw.resources()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    ledger=DOC.parent/'LEDGER.json'
    prior=read(ledger) if ledger.exists() else dict(physics=0,commands=0,science_seconds=0,audit_seconds=0,transport_seconds=0,Actor_live=0,Actor_shadow=0,Actor_independent=0,Value_live=0,Value_shadow=0,Value_independent=0,optimizer_preflight=0,jobs=[])
    assert 'T685' not in [j['id'] for j in prior['jobs']]
    assert prior['physics']+p['max_physics']<=a['max_new_physics'] and prior['commands']+p['max_commands']<=a['max_new_commands']
    assert prior['science_seconds']+p['preflight_seconds']+p['main_seconds']<=a['max_science_seconds'] and prior['audit_seconds']+p['audit_seconds']<=a['max_audit_seconds']
    prior['science_seconds']+=p['preflight_seconds']
    prior['jobs'].append(dict(id='T685',state='RESERVED',max_physics=p['max_physics'],max_commands=p['max_commands'],main_seconds=p['main_seconds'],audit_seconds=p['audit_seconds'],out=str(OUT),Actor_each=0,preflight_seconds=p['preflight_seconds']))
    assert all(prior[k]<=a[k+'_cap'] for k in ['Actor_live','Actor_shadow','Actor_independent'])
    save(ledger,prior);p.update(authorization_approved=True,registration_epoch=time.time())
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'REGISTERED.json',dict(state='REGISTERED_NOT_LAUNCHED',epoch=time.time(),new_optimizer=0))
    print('T685_REGISTERED_NOT_LAUNCHED')


def launch():
    p=read(OUT/'protocol.json');assert p['authorization_approved'] and not (OUT/'launch.json').exists()
    Wait.Raw.resources();receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time()-2);assert seconds>p['main_seconds']+p['audit_seconds']
    cmd=['systemd-run','--user','--unit=stage-t685','--collect','--property=Type=exec','--property=CPUQuota=400%',
        '--property=CPUWeight=10','--property=Nice=19','--property=AllowedCPUs=8 10 12 14','--property=MemoryMax=4G',
        '--property=MemorySwapMax=0','--property=KillMode=control-group','--property=TimeoutStopSec=5',
        f'--property=RuntimeMaxSec={seconds}',f'--property=WorkingDirectory={ROOT}',
        f'--property=StandardOutput=append:{OUT}/driver.log','--property=StandardError=inherit','--setenv=CUDA_VISIBLE_DEVICES=',
        'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage T685','--why=Finite forgetting full history',str(ROOT/'run_stage.sh'),str(HERE/'control.py'),'driver']
    c=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True);assert c.returncode==0,c.stderr
    save(OUT/'launch.json',dict(epoch=time.time(),unit='stage-t685',source_remote=receipt['remote_commit']))
    print(c.stderr.strip())


def actor_counts():
    f=OUT/'optimizer_calls.jsonl'
    rows=[json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []
    result={key:sum(r[0]==who and r[2:]==['actor','BEFORE'] for r in rows) for who,key in [('live','Actor_live'),('shadow','Actor_shadow'),('independent_history','Actor_independent')]}
    assert all(v==0 for v in result.values())
    return result


def driver():
    p=read(OUT/'protocol.json');assert p['authorization_approved'] and not (OUT/'DRIVER_STARTED.json').exists()
    assert sha(OUT/'protocol.json')==sha(HERE/'protocol.json')
    for f,h in p['inputs'].items():assert sha(f)==h,f
    os.sched_setaffinity(0,{14});Wait.resources()
    save(OUT/'DRIVER_STARTED.json',dict(pid=os.getpid(),epoch=time.time(),science_network=False))
    log=(OUT/'affinity.log').open('xb');helper=subprocess.Popen(['/usr/bin/python3',str(HERE/'affinity_guard.py')],stdout=log,stderr=log,cwd=ROOT)
    ids=[];error=None;elapsed={};failure_action=None
    def call(action,seconds):
        nonlocal failure_action
        assert helper.poll() is None and time.time()+seconds<p['absolute_all_stop']
        with (OUT/(action+'.log')).open('xb') as f:
            child=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'t685_runtime.py'),action],cwd=ROOT,stdout=f,stderr=f,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
            ids.append(child.pid);save(OUT/'status.json',dict(state='ACTIVE',action=action,child_pid=child.pid,epoch=time.time()))
            start=time.time()
            try:child.wait(timeout=seconds);assert child.returncode==0,(action,child.returncode)
            except BaseException:
                failure_action=action
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:child.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
            finally:elapsed[action]=time.time()-start
    try:
        until=time.time()+5
        while not (DOC/'AFFINITY_GUARD.json').exists() and time.time()<until:
            assert helper.poll() is None;time.sleep(.05)
        assert read(DOC/'AFFINITY_GUARD.json')['pid']==helper.pid
        call('collect',p['main_seconds']);call('audit',p['audit_seconds'])
        assert read(OUT/'audit.json')['state']=='PASS'
    except BaseException:error=traceback.format_exc();save(OUT/'driver_failure.json',dict(error=error,raw_preserved=True))
    if (OUT/'worker_generations.json').exists():
        for g in read(OUT/'worker_generations.json'):ids.extend([g['parent'],*g['workers']])
    gone=all(not Path('/proc',str(pid)).exists() for pid in ids)
    save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=gone,scientific_pids=sorted(set(ids)),epoch=time.time()))
    (OUT/'AFFINITY_STOP').touch()
    try:helper.wait(timeout=5)
    except subprocess.TimeoutExpired:helper.terminate();helper.wait(timeout=3)
    log.close();s=read(OUT/'summary.json') if (OUT/'summary.json').exists() else None
    report=dict(state='FAILED' if error else 'COMPLETE',error=error,failure_action=failure_action,all_scientific_processes_stopped=gone,helper_pid=helper.pid,helper_stopped=helper.poll() is not None,gate=s['gate'] if s else None,full407=None,no_model_promoted=True,
        physics=s['physics'] if s else p['max_physics'],commands=s['commands'] if s else p['max_commands'],
        science_seconds=elapsed.get('collect',0),audit_seconds=elapsed.get('audit',0),Actor_calls=actor_counts(),
        audit=read(OUT/'audit.json') if (OUT/'audit.json').exists() else None,remote='PENDING_LOCAL_RAW_RETAINED',epoch=time.time())
    save(OUT/'FINAL_REPORT.json',report);save(DOC/'FINAL_REPORT.json',report)
    ledger=read(DOC.parent/'LEDGER.json');item=next(j for j in ledger['jobs'] if j['id']=='T685');assert item['state']=='RESERVED'
    item.update(state=report['state'],actual_physics=report['physics'],actual_commands=report['commands'],science_seconds=report['science_seconds'],audit_seconds=report['audit_seconds'])
    for key,val in report['Actor_calls'].items():ledger[key]+=val
    item['Actor_calls']=report['Actor_calls']
    for dest,src in [('physics','physics'),('commands','commands'),('science_seconds','science_seconds'),('audit_seconds','audit_seconds')]:ledger[dest]+=report[src]
    save(DOC.parent/'LEDGER.json',ledger)
    state=read(DOC.parent/'CAMPAIGN_STATE.json');state.update(epoch=time.time(),state='T685_FINISHED_PROXY_VALIDITY_REVIEW' if not error else 'T685_FAILED_PRESERVE_AND_DIAGNOSE',active_run=str(OUT),unit='stage-t685.service',new_campaign_science_started=True,new_campaign_physics=ledger['physics'],new_campaign_commands=ledger['commands'],next_action='Read original/fresh IID per-action proxy signs, exact trajectory logratios/ESS/support, deterministic counterexamples and independent audit. No optimizer, candidate selection, ARlearning or automatic admission; next mechanism must address observed support/mismatch.',last_report=str(DOC/'FINAL_REPORT.json'))
    save(DOC.parent/'CAMPAIGN_STATE.json',state);save(OUT/'status.json',dict(state=report['state'],all_science_stopped=gone,epoch=time.time()))
    print(json.dumps({k:v for k,v in report.items() if k!='audit'}),flush=True)


if __name__=='__main__':{'prepare':prepare,'register':register,'launch':launch,'driver':driver}[sys.argv[1]]()
