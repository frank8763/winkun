"""Finite supervised fit/audit then original-start evaluation/audit."""
from pathlib import Path
import json,os,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t698_cross_representation_success';DOC=ROOT/'stage/docs/weekend_20261120_23/t698'
from status_io import save
import t698_wait as Wait
read=lambda f:json.loads(Path(f).read_text())
def sha(f):
    import hashlib
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def used(folder):return sum(f.stat().st_size for f in Path(folder).rglob('*') if f.is_file())


def register():
    assert not OUT.exists();p=read(HERE/'protocol.json');assert not p['authorization_approved']
    assert read(DOC/'PREFLIGHT.json')['state']=='PASS_ZERO_NATIVE_ZERO_OPTIMIZER_NO_COEFFICIENT_COMMIT'
    Wait.Raw.resources();auth=read(p['weekend_authorization']);ledger=read(DOC.parent/'LEDGER.json')
    assert auth['approved'] and not any(j['id']=='T698' for j in ledger['jobs'])
    for f,h in p['inputs'].items():assert sha(f)==h,f
    assert used(ROOT/'stage/runs')+p['max_bytes']<min(auth['archive_trigger_bytes'],p['global_bytes'])
    assert time.time()+p['root_seconds']<auth['absolute_all_stop']
    for key,amount,cap in [('physics',p['night_max_physics'],'max_new_physics'),('commands',p['night_max_commands'],'max_new_commands'),
        ('science_seconds',p['preflight_seconds']+p['night_main_seconds'],'max_science_seconds'),('audit_seconds',p['audit_total_seconds'],'max_audit_seconds')]:
        assert ledger[key]+amount<=auth[cap]
    for key in ['Actor_live','Actor_shadow','Actor_independent']:assert ledger[key]+1<=auth[key+'_cap']
    now=time.time();p.update(authorization_approved=True,registration_epoch=now,absolute_all_stop=min(now+p['root_seconds'],auth['absolute_all_stop']),
        absolute_learning_stop=min(now+p['root_seconds'],auth['absolute_learning_stop']),absolute_native_stop=min(now+p['root_seconds']-300,auth['absolute_native_stop']))
    ledger['science_seconds']+=p['preflight_seconds'];ledger['jobs'].append(dict(id='T698',state='RESERVED',out=str(OUT),
        max_physics=p['night_max_physics'],max_commands=p['night_max_commands'],main_seconds=p['night_main_seconds'],audit_seconds=p['audit_total_seconds'],
        preflight_seconds=p['preflight_seconds'],Actor_each_cap=1,Value_each_cap=0,fit_type='ONE_SUPERVISED_RADIAL_READOUT_COEFFICIENT_COMMIT',completed_children=[]))
    save(DOC.parent/'LEDGER.json',ledger);OUT.mkdir()
    blob=json.dumps(p,separators=(',',':'))+'\n'
    assert len(blob.encode())<2**21
    (HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob)
    save(DOC/'REGISTERED.json',dict(state='REGISTERED_NOT_LAUNCHED',epoch=now,absolute_all_stop=p['absolute_all_stop'],new_native=0,new_Actor=0,new_optimizer=0))
    print('T698_REGISTERED_NOT_LAUNCHED')


def launch():
    p=read(OUT/'protocol.json');assert p['authorization_approved'] and not (OUT/'launch.json').exists();Wait.Raw.resources()
    r=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert r['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time()-2);assert seconds>p['night_main_seconds']+p['audit_total_seconds']+60
    cmd=['systemd-run','--user','--unit=stage-t698','--collect','--property=Type=exec','--property=CPUQuota=800%','--property=CPUWeight=10','--property=Nice=19',
        '--property=AllowedCPUs=8 9 10 11 12 13 14 15','--property=MemoryMax=8G','--property=MemorySwapMax=0','--property=KillMode=control-group','--property=TimeoutStopSec=5',
        f'--property=RuntimeMaxSec={seconds}',f'--property=WorkingDirectory={ROOT}',f'--property=StandardOutput=append:{OUT}/driver.log',
        '--property=StandardError=inherit','--setenv=CUDA_VISIBLE_DEVICES=0',
        'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage T698','--why=Finite fullhistory compatibility',str(ROOT/'run_stage.sh'),str(HERE/'control.py'),'driver']
    c=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True);assert c.returncode==0,c.stderr
    save(OUT/'launch.json',dict(epoch=time.time(),unit='stage-t698',source_remote=r['remote_commit']));print(c.stderr.strip())


def counts():
    f=OUT/'fit/fit_calls.jsonl';rows=[json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []
    result={key:sum(r[0]==who and r[2:]==['actor_closed_form','BEFORE'] for r in rows) for who,key in
        [('live','Actor_live'),('shadow','Actor_shadow'),('independent_history','Actor_independent')]}
    assert all(v<=1 for v in result.values());return result


def driver():
    p=read(OUT/'protocol.json');assert p['authorization_approved'] and not (OUT/'DRIVER_STARTED.json').exists()
    assert sha(OUT/'protocol.json')==sha(HERE/'protocol.json')
    for f,h in p['inputs'].items():assert sha(f)==h,f
    Wait.resources();os.sched_setaffinity(0,{14});save(OUT/'DRIVER_STARTED.json',dict(pid=os.getpid(),epoch=time.time(),science_network=False))
    log=(OUT/'affinity.log').open('xb');helper=subprocess.Popen(['/usr/bin/python3',str(HERE/'affinity_guard.py')],cwd=ROOT,stdout=log,stderr=log)
    ids=[];items=[];error=None
    try:
        until=time.time()+5
        while not (DOC/'AFFINITY_GUARD.json').exists() and time.time()<until:
            assert helper.poll() is None;time.sleep(.05)
        assert read(DOC/'AFFINITY_GUARD.json')['pid']==helper.pid
        for arm in ['fit','evaluation']:
            fitting=arm=='fit';main=p['fit_main_seconds'] if fitting else p['child_main_seconds'];audit=p['fit_audit_seconds'] if fitting else p['child_audit_seconds']
            size=p['fit_bytes'] if fitting else p['child_bytes'];Wait.resources();assert time.time()+main+audit+60<p['absolute_all_stop']
            assert used(OUT)+size+p['final_storage_reserve']<p['max_bytes']
            folder=OUT/arm;folder.mkdir();(folder/'episodes').mkdir();(DOC/arm).mkdir();now=time.time()
            child=dict(out=str(folder),arm=arm,_root_sha256=sha(OUT/'protocol.json'),max_bytes=size,child_bytes=size,
                max_physics=0 if fitting else p['child_max_physics'],max_commands=0 if fitting else p['child_max_commands'],main_seconds=main,audit_seconds=audit,
                absolute_learning_stop=min(now+main,p['campaign_learning_stop']),absolute_native_stop=min(now+main,p['campaign_native_stop']),
                absolute_all_stop=min(now+main+audit+60,p['absolute_all_stop']))
            save(folder/'protocol.json',child);elapsed={};failure=None
            try:
                actions=[('fit',main),('fit_audit',audit)] if fitting else [('evaluate',main),('evaluate_audit',audit)]
                for action,seconds in actions:
                    assert helper.poll() is None
                    with (folder/(action+'.log')).open('xb') as f:
                        c=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'t698_runtime.py'),action,arm],cwd=ROOT,stdout=f,stderr=f,
                                           start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8'))
                        ids.append(c.pid);save(OUT/'status.json',dict(state='ACTIVE',child=arm,action=action,child_pid=c.pid,epoch=time.time()));started=time.time()
                        try:c.wait(timeout=seconds);assert c.returncode==0,(arm,action,c.returncode)
                        except BaseException:
                            try:os.killpg(c.pid,signal.SIGTERM)
                            except ProcessLookupError:pass
                            try:c.wait(timeout=3)
                            except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
                            raise
                        finally:elapsed['audit' if 'audit' in action else 'collect']=time.time()-started
                assert read(folder/'audit.json')['state']=='PASS'
            except BaseException:failure=traceback.format_exc()
            s=read(folder/'summary.json') if (folder/'summary.json').exists() else None
            item=dict(arm=arm,state='FAILED' if failure else 'COMPLETE',error=failure,physics=s['physics'] if s else child['max_physics'],
                commands=s['commands'] if s else child['max_commands'],science_seconds=elapsed.get('collect',0),audit_seconds=elapsed.get('audit',0),
                Actor_calls=counts() if fitting else dict(Actor_live=0,Actor_shadow=0,Actor_independent=0),
                command_gate=s.get('command_gate') if s else None,full407=s.get('full407') if s else None,
                audit=read(folder/'audit.json') if (folder/'audit.json').exists() else None)
            items.append(item);save(OUT/'LEDGER.json',dict(children=items))
            ledger=read(DOC.parent/'LEDGER.json');job=next(j for j in ledger['jobs'] if j['id']=='T698');assert arm not in job['completed_children']
            for key in ['physics','commands','science_seconds','audit_seconds']:ledger[key]+=item[key]
            for key,val in item['Actor_calls'].items():ledger[key]+=val
            job.update(state='ACTIVE',actual_physics=sum(v['physics'] for v in items),actual_commands=sum(v['commands'] for v in items));job['completed_children'].append(arm)
            save(DOC.parent/'LEDGER.json',ledger)
            if failure:raise RuntimeError(failure)
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    for f in OUT.glob('*/worker_generations.json'):
        for g in read(f):ids.extend([g['parent'],*g['workers']])
    gone=all(not Path('/proc',str(pid)).exists() for pid in set(ids));save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=gone,scientific_pids=sorted(set(ids)),epoch=time.time()))
    (OUT/'AFFINITY_STOP').touch()
    try:helper.wait(timeout=5)
    except subprocess.TimeoutExpired:helper.terminate();helper.wait(timeout=3)
    log.close();report=dict(state='FAILED' if error or not gone else 'COMPLETE',error=error,children=items,all_scientific_processes_stopped=gone,
        helper_pid=helper.pid,helper_stopped=helper.poll() is not None,remote='PENDING_LOCAL_RAW_RETAINED',epoch=time.time())
    save(OUT/'FINAL_REPORT.json',report);save(DOC/'FINAL_REPORT.json',report)
    ledger=read(DOC.parent/'LEDGER.json');next(j for j in ledger['jobs'] if j['id']=='T698')['state']=report['state'];save(DOC.parent/'LEDGER.json',ledger)
    state=read(DOC.parent/'CAMPAIGN_STATE.json');state.update(epoch=time.time(),state='T698_FINISHED_NEEDS_FULL407_AUDIT_REVIEW',active_run=str(OUT),unit='stage-t698.service',
        last_report=str(DOC/'FINAL_REPORT.json'),next_action='Read same candidate all403 retention, all7 fixed source jobs, core407 and separate T69524, independent fit and native audit; allPIDs/helper/unit exit before source/archive. Finite supervised compatibility only. Preserve T694 and T501 until complete gate.')
    save(DOC.parent/'CAMPAIGN_STATE.json',state);save(OUT/'status.json',dict(state=report['state'],all_science_stopped=gone,epoch=time.time()));print(report['state'])


if __name__=='__main__':
    if sys.argv[1]=='prepare':
        import t698_prepare
        t698_prepare.main()
    else:{'register':register,'launch':launch,'driver':driver}[sys.argv[1]]()
