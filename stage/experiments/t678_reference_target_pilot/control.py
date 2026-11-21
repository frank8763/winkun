"""Two sequential finite one-step candidates, no duplicate or unlimited relay."""
from pathlib import Path
import json,os,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t678_reference_target_pilot';DOC=ROOT/'stage/docs/weekend_20261120_23/t678'
from status_io import save
import t678_wait as Wait
def read(f):return json.loads(Path(f).read_text())
def sha(f):
    import hashlib
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()

def used(folder):return sum(f.stat().st_size for f in Path(folder).rglob('*') if f.is_file())

def prepare():
    import t678_prepare as P
    P.main()

def register():
    assert not OUT.exists();p=read(HERE/'protocol.json');assert not p['authorization_approved']
    assert read(DOC/'PREFLIGHT.json')['state']=='PASS_ZERO_NATIVE_ZERO_OPTIMIZER'
    for f,h in p['inputs'].items():assert sha(f)==h,f
    Wait.Raw.resources();a=read(p['weekend_authorization']);l=read(DOC.parent/'LEDGER.json');assert a['approved'] and not any(j['id']=='T678' for j in l['jobs'])
    assert used(ROOT/'stage/runs')+p['max_bytes']<p['global_bytes']
    for key,amount,cap in [('physics',p['night_max_physics'],'max_new_physics'),('commands',p['night_max_commands'],'max_new_commands'),('science_seconds',p['night_main_seconds']+p['preflight_seconds'],'max_science_seconds'),('audit_seconds',p['audit_total_seconds'],'max_audit_seconds')]:assert l[key]+amount<=a[cap]
    for key in ['Actor_live','Actor_shadow','Actor_independent']:assert l[key]+2<=a[key+'_cap']
    l['science_seconds']+=p['preflight_seconds'];l['jobs'].append(dict(id='T678',state='RESERVED',max_physics=p['night_max_physics'],max_commands=p['night_max_commands'],main_seconds=p['night_main_seconds'],audit_seconds=p['audit_total_seconds'],preflight_seconds=p['preflight_seconds'],Actor_each_cap=2,out=str(OUT),completed_children=[]))
    save(DOC.parent/'LEDGER.json',l);p.update(authorization_approved=True,registration_epoch=time.time());OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'REGISTERED.json',dict(state='REGISTERED_NOT_LAUNCHED',epoch=time.time(),new_optimizer=0));print('T678_REGISTERED_NOT_LAUNCHED')

def launch():
    p=read(OUT/'protocol.json');assert p['authorization_approved'] and not (OUT/'launch.json').exists();Wait.Raw.resources()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    left=int(p['absolute_all_stop']-time.time()-2);assert left>p['night_main_seconds']+p['audit_total_seconds']
    cmd=['systemd-run','--user','--unit=stage-t678','--collect','--property=Type=exec','--property=CPUQuota=400%','--property=CPUWeight=10','--property=Nice=19',
         '--property=AllowedCPUs=8 10 12 14','--property=MemoryMax=4G','--property=MemorySwapMax=0','--property=KillMode=control-group','--property=TimeoutStopSec=5',
         f'--property=RuntimeMaxSec={left}',f'--property=WorkingDirectory={ROOT}',f'--property=StandardOutput=append:{OUT}/driver.log','--property=StandardError=inherit','--setenv=CUDA_VISIBLE_DEVICES=',
         'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage T678','--why=Two finite matched reference-target candidates',str(ROOT/'run_stage.sh'),str(HERE/'control.py'),'driver']
    r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True);assert r.returncode==0,r.stderr
    save(OUT/'launch.json',dict(epoch=time.time(),unit='stage-t678',source_remote=receipt['remote_commit']));print(r.stderr.strip())

def counts(folder):
    f=folder/'optimizer_calls.jsonl';rows=[json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []
    out={key:sum(x[0]==who and x[2:]==['actor','BEFORE'] for x in rows) for who,key in [('live','Actor_live'),('shadow','Actor_shadow'),('independent_history','Actor_independent')]}
    assert all(v<=1 for v in out.values());return out

def driver():
    p=read(OUT/'protocol.json');assert p['authorization_approved'] and not (OUT/'DRIVER_STARTED.json').exists()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    Wait.resources();os.sched_setaffinity(0,{14});save(OUT/'DRIVER_STARTED.json',dict(pid=os.getpid(),epoch=time.time(),science_network=False))
    log=(OUT/'affinity.log').open('xb');helper=subprocess.Popen(['/usr/bin/python3',str(HERE/'affinity_guard.py')],stdout=log,stderr=log,cwd=ROOT)
    ids=[];items=[];error=None
    try:
        until=time.time()+5
        while not (DOC/'AFFINITY_GUARD.json').exists() and time.time()<until:assert helper.poll() is None;time.sleep(.05)
        assert read(DOC/'AFFINITY_GUARD.json')['pid']==helper.pid
        for arm in p['arms']:
            Wait.resources();assert time.time()+p['child_main_seconds']+p['child_audit_seconds']+60<p['absolute_all_stop']
            assert used(OUT)+p['child_bytes']+p['final_storage_reserve']<p['max_bytes'] and used(ROOT/'stage/runs')+p['child_bytes']<p['global_bytes']
            folder=OUT/arm;folder.mkdir();(folder/'episodes').mkdir();(DOC/arm).mkdir();now=time.time()
            child=dict(out=str(folder),arm=arm,target_arm=arm,_root_sha256=sha(OUT/'protocol.json'),max_bytes=p['child_bytes'],max_physics=p['child_max_physics'],max_commands=p['child_max_commands'],
                       main_seconds=p['child_main_seconds'],audit_seconds=p['child_audit_seconds'],absolute_learning_stop=min(now+p['learning_seconds']+30,p['campaign_learning_stop']),
                       absolute_native_stop=min(now+p['child_main_seconds'],p['campaign_native_stop']),absolute_all_stop=min(now+p['child_main_seconds']+p['child_audit_seconds']+60,p['absolute_all_stop']))
            save(folder/'protocol.json',child);elapsed={};failure=None
            try:
                for action,seconds in [('collect',p['child_main_seconds']),('audit',p['child_audit_seconds'])]:
                    assert helper.poll() is None
                    with (folder/(action+'.log')).open('xb') as f:
                        c=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'t678_runtime.py'),action,arm],cwd=ROOT,stdout=f,stderr=f,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
                        ids.append(c.pid);save(OUT/'status.json',dict(state='ACTIVE',child=arm,action=action,child_pid=c.pid,epoch=time.time()));start=time.time()
                        try:c.wait(timeout=seconds);assert c.returncode==0,(arm,action,c.returncode)
                        except BaseException:
                            try:os.killpg(c.pid,signal.SIGTERM)
                            except ProcessLookupError:pass
                            try:c.wait(timeout=3)
                            except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
                            raise
                        finally:elapsed[action]=time.time()-start
                assert read(folder/'audit.json')['state']=='PASS'
            except BaseException:failure=traceback.format_exc()
            s=read(folder/'summary.json') if (folder/'summary.json').exists() else None
            item=dict(arm=arm,state='FAILED' if failure else 'COMPLETE',error=failure,physics=s['physics'] if s else p['child_max_physics'],commands=s['commands'] if s else p['child_max_commands'],
                      science_seconds=elapsed.get('collect',0),audit_seconds=elapsed.get('audit',0),Actor_calls=counts(folder),full407=s.get('full407') if s else None,
                      screen=read(folder/'SCREEN.json') if (folder/'SCREEN.json').exists() else None,audit=read(folder/'audit.json') if (folder/'audit.json').exists() else None)
            items.append(item);save(OUT/'LEDGER.json',dict(children=items))
            l=read(DOC.parent/'LEDGER.json');job=next(j for j in l['jobs'] if j['id']=='T678');assert arm not in job['completed_children']
            for key in ['physics','commands','science_seconds','audit_seconds']:l[key]+=item[key]
            for key,val in item['Actor_calls'].items():l[key]+=val
            job['completed_children'].append(arm);job['state']='ACTIVE';job['actual_physics']=sum(x['physics'] for x in items);job['actual_commands']=sum(x['commands'] for x in items)
            save(DOC.parent/'LEDGER.json',l)
            if failure:raise RuntimeError(failure)
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    for f in OUT.glob('*/worker_generations.json'):
        for generation in read(f):ids.extend([generation['parent'],*generation['workers']])
    gone=all(not Path('/proc',str(pid)).exists() for pid in set(ids));save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=gone,scientific_pids=sorted(set(ids)),epoch=time.time()))
    (OUT/'AFFINITY_STOP').touch()
    try:helper.wait(timeout=5)
    except subprocess.TimeoutExpired:helper.terminate();helper.wait(timeout=3)
    log.close();report=dict(state='FAILED' if error else ('COMPLETE' if len(items)==2 else 'INCOMPLETE'),error=error,children=items,all_scientific_processes_stopped=gone,remote='PENDING_LOCAL_RAW_RETAINED',epoch=time.time())
    save(OUT/'FINAL_REPORT.json',report);save(DOC/'FINAL_REPORT.json',report)
    l=read(DOC.parent/'LEDGER.json');next(j for j in l['jobs'] if j['id']=='T678')['state']=report['state'];save(DOC.parent/'LEDGER.json',l)
    state=read(DOC.parent/'CAMPAIGN_STATE.json');state.update(epoch=time.time(),state='T678_FINISHED_NEEDS_MATCHED_FULL407_REVIEW',active_run=str(OUT),unit='stage-t678.service',last_report=str(DOC/'FINAL_REPORT.json'),next_action='Read paired single-factor target-only real one-step outcomes and all perID gains/losses/audits. Keep only samecandidate384+gain full407 PASS; if neither improves review/pivot, no targetweight/seed grid.')
    save(DOC.parent/'CAMPAIGN_STATE.json',state);save(OUT/'status.json',dict(state=report['state'],all_science_stopped=gone,epoch=time.time()));print(report['state'])

if __name__=='__main__':{'prepare':prepare,'register':register,'launch':launch,'driver':driver}[sys.argv[1]]()
