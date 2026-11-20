"""Eight independently audited finite batches; no optimizer calls."""
from pathlib import Path
import json,os,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t672_full_suite_onpolicy';DOC=ROOT/'stage/docs/weekend_20261120_23/t672'
import t672_wait as Wait
from status_io import save
def read(f):return json.loads(Path(f).read_text())
def sha(f):
    import hashlib
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def used(f):return sum(x.stat().st_size for x in Path(f).rglob('*') if x.is_file())


def register():
    p=read(HERE/'protocol.json');assert not OUT.exists() and not p['authorization_approved']
    assert read(DOC/'PREFLIGHT.json')['state']=='PASS_ZERO_NATIVE_ZERO_OPTIMIZER'
    auth=read(p['weekend_authorization']);assert auth['approved']
    for f,h in p['inputs'].items():assert sha(f)==h,f
    Wait.Raw.resources();l=read(DOC.parent/'LEDGER.json')
    assert 'T672' not in [j['id'] for j in l['jobs']]
    assert l['physics']+p['night_max_physics']<=auth['max_new_physics']
    assert l['commands']+p['night_max_commands']<=auth['max_new_commands']
    assert l['science_seconds']+p['night_main_seconds']<=auth['max_science_seconds']
    assert l['audit_seconds']+p['audit_total_seconds']<=auth['max_audit_seconds']
    l['jobs'].append(dict(id='T672',state='RESERVED',max_physics=p['night_max_physics'],max_commands=p['night_max_commands'],main_seconds=p['night_main_seconds'],audit_seconds=p['audit_total_seconds'],out=str(OUT),actual_physics=0,actual_commands=0,science_seconds=0,completed_children=[]))
    save(DOC.parent/'LEDGER.json',l);p.update(authorization_approved=True,registration_epoch=time.time())
    OUT.mkdir();blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob)
    save(DOC/'REGISTERED.json',dict(state='REGISTERED_NOT_LAUNCHED',epoch=time.time(),new_optimizer=0));print('T672_REGISTERED_NOT_LAUNCHED')


def launch():
    p=read(OUT/'protocol.json');assert p['authorization_approved'] and not (OUT/'launch.json').exists()
    Wait.Raw.resources();r=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert r['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    left=int(p['absolute_all_stop']-time.time()-2);assert left>sum(c['main_seconds']+c['audit_seconds'] for c in p['batch_configurations'])
    cmd=['systemd-run','--user','--unit=stage-t672','--collect','--property=Type=exec','--property=CPUQuota=400%',
        '--property=CPUWeight=10','--property=Nice=19','--property=AllowedCPUs=8 10 12 14','--property=MemoryMax=4G',
        '--property=MemorySwapMax=0','--property=KillMode=control-group','--property=TimeoutStopSec=5',
        f'--property=RuntimeMaxSec={left}',f'--property=WorkingDirectory={ROOT}',f'--property=StandardOutput=append:{OUT}/driver.log',
        '--property=StandardError=inherit','--setenv=CUDA_VISIBLE_DEVICES=','systemd-inhibit','--what=sleep:idle','--mode=block',
        '--who=Stage T672','--why=Finite systematic onpolicy task coverage',str(ROOT/'run_stage.sh'),str(HERE/'control.py'),'driver']
    r0=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True);assert r0.returncode==0,r0.stderr
    save(OUT/'launch.json',dict(epoch=time.time(),unit='stage-t672',source_remote=r['remote_commit']));print(r0.stderr.strip())


def driver():
    p=read(OUT/'protocol.json');assert p['authorization_approved'] and not (OUT/'DRIVER_STARTED.json').exists()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    Wait.resources();os.sched_setaffinity(0,{14})
    save(OUT/'DRIVER_STARTED.json',dict(pid=os.getpid(),epoch=time.time(),science_network=False,new_optimizer=0))
    log=(OUT/'affinity.log').open('xb');helper=subprocess.Popen(['/usr/bin/python3',str(HERE/'affinity_guard.py')],stdout=log,stderr=log,cwd=ROOT)
    children=[];results=[];error=None
    ledger=dict(physics=0,commands=0,main_seconds=0.,audit_seconds=0.,children=[],denied=[])
    def action(name,which,seconds):
        assert helper.poll() is None and time.time()+seconds<p['absolute_all_stop']
        with (OUT/name/(which+'.log')).open('xb') as f:
            child=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'t672_runtime.py'),which,name],cwd=ROOT,stdout=f,stderr=f,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
            children.append(child.pid);save(OUT/'status.json',dict(state='ACTIVE',child=name,action=which,child_pid=child.pid,epoch=time.time()))
            start=time.time()
            try:child.wait(timeout=seconds);assert child.returncode==0,(name,which,child.returncode)
            except BaseException:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:child.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
            finally:return_elapsed[which]=time.time()-start
    try:
        until=time.time()+5
        while not (DOC/'AFFINITY_GUARD.json').exists() and time.time()<until:
            assert helper.poll() is None;time.sleep(.05)
        assert read(DOC/'AFFINITY_GUARD.json')['pid']==helper.pid
        for config in p['batch_configurations']:
            Wait.resources();name=config['name']
            checks=dict(physics=ledger['physics']+config['max_physics']<=p['night_max_physics'],commands=ledger['commands']+config['max_commands']<=p['night_max_commands'],
                science_time=ledger['main_seconds']+config['main_seconds']<=p['night_main_seconds'],audit_time=ledger['audit_seconds']+config['audit_seconds']<=p['audit_total_seconds'],
                storage=used(OUT)+config['child_bytes']+p['final_storage_reserve']<=p['max_bytes'],
                native_deadline=time.time()+config['main_seconds']<p['absolute_native_stop'],all_deadline=time.time()+config['main_seconds']+config['audit_seconds']+60<p['absolute_all_stop'])
            if not all(checks.values()):ledger['denied'].append(dict(name=name,checks=checks));break
            folder=OUT/name;folder.mkdir();(folder/'episodes').mkdir();(DOC/name).mkdir()
            child=dict(config,out=str(folder),_root_sha256=sha(OUT/'protocol.json'));save(folder/'protocol.json',child)
            return_elapsed={};failure=None
            try:
                action(name,'collect',config['main_seconds']);assert read(folder/'summary.json')['state']=='COMPLETE'
                action(name,'audit',config['audit_seconds']);assert read(folder/'audit.json')['state']=='PASS'
            except BaseException:failure=traceback.format_exc()
            s=read(folder/'summary.json') if (folder/'summary.json').exists() else None
            item=dict(name=name,state='FAILED' if failure else 'COMPLETE',error=failure,
                physics=s['physics'] if s else config['max_physics'],commands=s['commands'] if s else config['max_commands'],
                science_seconds=return_elapsed.get('collect',0),audit_seconds=return_elapsed.get('audit',0),
                measured=s['measured'] if s else None,safe=s['safe'] if s else None,
                audit=read(folder/'audit.json') if (folder/'audit.json').exists() else None)
            results.append(item);ledger.update(physics=sum(x['physics'] for x in results),commands=sum(x['commands'] for x in results),
                main_seconds=sum(x['science_seconds'] for x in results),audit_seconds=sum(x['audit_seconds'] for x in results),children=results)
            save(OUT/'LEDGER.json',ledger)
            global_ledger=read(DOC.parent/'LEDGER.json');job=next(j for j in global_ledger['jobs'] if j['id']=='T672')
            assert name not in job['completed_children']
            for dst,value in [('physics',item['physics']),('commands',item['commands']),('science_seconds',item['science_seconds']),('audit_seconds',item['audit_seconds'])]:global_ledger[dst]+=value
            job.update(state='ACTIVE',actual_physics=ledger['physics'],actual_commands=ledger['commands'],science_seconds=ledger['main_seconds'],audit_seconds=ledger['audit_seconds'])
            job['completed_children'].append(name);save(DOC.parent/'LEDGER.json',global_ledger)
            state=read(DOC.parent/'CAMPAIGN_STATE.json');state.update(epoch=time.time(),state='T672_SYSTEMATIC_COLLECTION_ACTIVE',active_run=str(OUT),unit='stage-t672.service',completed_collection_batches=len(results),planned_collection_batches=8,next_action='Finish all preregistered systematic tasks and audits before any learning; sampled GOAL is not deterministic acceptance')
            save(DOC.parent/'CAMPAIGN_STATE.json',state)
            if failure:raise RuntimeError(failure)
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    for f in OUT.glob('*/worker_generations.json'):
        for group in read(f):children.extend([group['parent'],*group['workers']])
    gone=all(not Path('/proc',str(pid)).exists() for pid in set(children))
    save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=gone,scientific_pids=sorted(set(children)),epoch=time.time()))
    (OUT/'AFFINITY_STOP').touch()
    try:helper.wait(timeout=5)
    except subprocess.TimeoutExpired:helper.terminate();helper.wait(timeout=3)
    log.close()
    report=dict(state='FAILED' if error else ('COMPLETE' if len(results)==8 else 'INCOMPLETE_BUDGET_ADMISSION_STOP'),error=error,
        ledger=ledger,all_scientific_processes_stopped=gone,completed_batches=len(results),new_optimizer=0,remote='PENDING_LOCAL_RAW_RETAINED',epoch=time.time())
    save(OUT/'FINAL_REPORT.json',report);save(DOC/'FINAL_REPORT.json',report)
    l=read(DOC.parent/'LEDGER.json');job=next(j for j in l['jobs'] if j['id']=='T672');job['state']=report['state'];save(DOC.parent/'LEDGER.json',l)
    state=read(DOC.parent/'CAMPAIGN_STATE.json');state.update(epoch=time.time(),state='T672_FINISHED_NEEDS_ALL_TASK_RESULT_REVIEW',last_report=str(DOC/'FINAL_REPORT.json'),next_action='Aggregate all407 onpolicy support, hash/exact audits and task-balanced gradient geometry; only then preregister one joint finite learning job, no stale-batch unlimited fitting')
    save(DOC.parent/'CAMPAIGN_STATE.json',state);save(OUT/'status.json',dict(state=report['state'],all_science_stopped=gone,epoch=time.time()))
    print(json.dumps({k:v for k,v in report.items() if k!='ledger'}),flush=True)


if __name__=='__main__':{'register':register,'launch':launch,'driver':driver}[sys.argv[1]]()
