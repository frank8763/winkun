"""Independent authorized window registration and single offline driver."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t647_joint_success_coverage';DOC=ROOT/'stage/docs/day_20261119/t647'
sys.path.insert(0,str(ROOT/'stage/experiments/t645_iterative_recovery_relay'))
import control as Resource

def read(f):return json.loads(Path(f).read_text())
def save(f,v):Path(f).write_text(json.dumps(v,indent=2)+'\n')
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()

def register():
    assert not OUT.exists();p=read(HERE/'protocol.json');a=read(DOC/'AUTHORIZATION.json')
    assert a['approved'] and not p['authorization_approved']
    for k in ['total_max_physics','total_max_commands','total_main_seconds','Actor_each_cap','max_bytes','global_bytes','all_seconds']:assert a[k]==p[k],k
    assert read(DOC/'PREFLIGHT.json')['state']=='PASS_ZERO_NATIVE_ZERO_OPTIMIZER'
    for f,h in p['inputs'].items():assert sha(f)==h,f
    Resource.resources();now=time.time()
    p.update(authorization_approved=True,registration_epoch=now,absolute_learning_stop=now+7200,absolute_native_stop=now+9000,absolute_all_stop=now+10800)
    p['inputs'].update({str(DOC/'AUTHORIZATION.json'):sha(DOC/'AUTHORIZATION.json'),str(DOC/'PREFLIGHT.json'):sha(DOC/'PREFLIGHT.json')})
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    for name in p['arms']:
        (OUT/name).mkdir();(DOC/name).mkdir()
        save(OUT/name/'protocol.json',dict(arm=name,_root_sha256=sha(OUT/'protocol.json')))
    save(DOC/'REGISTERED.json',dict(state='REGISTERED_NOT_LAUNCHED',epoch=now,all_stop=p['absolute_all_stop']))
    print('T647_REGISTERED_NOT_LAUNCHED',p['absolute_all_stop'])


def run(action,name,seconds,p,children):
    assert time.time()+seconds+30<p['absolute_all_stop']
    folder=OUT/name if name else OUT
    with (folder/(action+'.log')).open('xb') as log:
        cmd=[str(ROOT/'run_stage.sh'),str(HERE/'t647_runtime.py'),action,name] if name else [str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),action]
        c=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CUBLAS_WORKSPACE_CONFIG=':4096:8'))
        children.append(c.pid)
        try:c.wait(timeout=seconds);assert c.returncode==0,(action,name,c.returncode)
        except BaseException:
            try:os.killpg(c.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:c.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
            raise


def driver():
    p=read(OUT/'protocol.json');assert p['authorization_approved'] and sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert not (OUT/'DRIVER_STARTED.json').exists()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    Resource.resources();os.sched_setaffinity(0,{14})
    save(OUT/'DRIVER_STARTED.json',dict(pid=os.getpid(),epoch=time.time(),source_remote=receipt['remote_commit']))
    # Helper shares this exact cgroup and all external bounds; no network or
    # science mutation. Check the helper's own error file before final success.
    with (OUT/'affinity.log').open('xb') as log:
        helper=subprocess.Popen(['/usr/bin/python3',str(HERE/'affinity_guard.py')],cwd=ROOT,stdout=log,stderr=log,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    children=[];results=[];error=None;old_remote='PENDING_LOCAL_RAW_RETAINED'
    try:
        try:run('old',None,p['old_backup_seconds'],p,children);old_remote='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
        except BaseException:save(DOC/'OLD_BACKUP_PENDING.json',dict(error=traceback.format_exc(),raw_retained=True))
        main_used=0.
        for name in p['arms']:
            assert time.time()+p['main_seconds']+p['audit_seconds']+p['backup_seconds']+p['source_seconds']+60<p['absolute_all_stop']
            Resource.resources();save(OUT/'status.json',dict(state='ACTIVE',arm=name,epoch=time.time()))
            started=time.time();run('main',name,p['main_seconds'],p,children);main_used+=time.time()-started
            run('audit',name,p['audit_seconds'],p,children)
            s=read(OUT/name/'summary.json');assert read(OUT/name/'audit.json')['state']=='PASS'
            results.append(dict(arm=name,state=s['state'],physics=s['physics'],commands=s['commands'],field24_success=s['screen']['field24_success'],old_screen_kept=s['screen']['old_sentinels_and_field5_retained'],new_field_success_ids=s['screen']['new_field_success_ids'],final407=s['final407'],retained=s['retained']))
            assert sum(v['physics'] for v in results)<=p['total_max_physics'] and sum(v['commands'] for v in results)<=p['total_max_commands'] and main_used<=p['total_main_seconds']
            save(OUT/'LEDGER.json',dict(results=results,physics=sum(v['physics'] for v in results),commands=sum(v['commands'] for v in results),main_seconds=main_used,Actor_live_each=len(results)*p['fit_steps']))
    except BaseException:error=traceback.format_exc()
    ids=list(children)
    for f in OUT.glob('*/worker_generations.json'):
        for g in read(f):ids.extend([g['parent'],*g['workers']])
    ids=sorted(set(ids));gone=all(not Path('/proc',str(i)).exists() for i in ids)
    save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=gone,scientific_pids=ids,epoch=time.time()));assert gone
    report=dict(state='FAILED' if error else 'COMPLETE',error=error,results=results,all_scientific_processes_stopped=gone,old_T646_remote=old_remote,remote='PENDING_LOCAL_RAW_RETAINED')
    save(OUT/'FINAL_REPORT.json',report)
    if time.time()+p['backup_seconds']+p['source_seconds']+60<p['absolute_all_stop']:
        try:run('current',None,p['backup_seconds'],p,[]);report['remote']='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
        except BaseException:report['backup_error']=traceback.format_exc()
    save(DOC/'FINAL_REPORT.json',report);save(OUT/'FINAL_REPORT_FINAL.json',report)
    save(OUT/'status.json',dict(state=report['state'],all_science_stopped=True,epoch=time.time()))
    save(OUT/'local_manifest.json',dict(files=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name not in ['driver.log','local_manifest.json','affinity.log']]))
    status=ROOT/'stage/docs/RESEARCH_STATUS.md';status.write_text('2026-11-19 T647成功轨迹覆盖对照结束：'+report['state']+'，remote='+report['remote']+'；field24/旧384/条件407见day_20261119/t647/FINAL_REPORT，未测null、不称泛化。\n\n'+status.read_text())
    try:
        paths=[str(DOC.relative_to(ROOT)),'stage/docs/RESEARCH_STATUS.md',str(Path('stage/docs/overnight_20261118/SOURCE_PENDING.json'))]
        for pattern in ['*T646_post*.json','*T647_post*.json']:
            paths.extend(str(f.relative_to(ROOT)) for f in (ROOT/'stage/docs/long_distance_20261101').glob(pattern))
        subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True,capture_output=True)
        subprocess.run(['git','commit','-m','Report matched original-start success coverage and exact retention gates'],cwd=ROOT,check=True,capture_output=True)
        if time.time()+p['source_seconds']+15<p['absolute_all_stop']:
            subprocess.run([str(ROOT/'run_stage.sh'),str(ROOT/'stage/experiments/t629_offline_night_relay/t629_transport.py'),'source_sync'],cwd=ROOT,check=True,capture_output=True,timeout=p['source_seconds'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    except BaseException:save(DOC/'SOURCE_PENDING.json',dict(error=traceback.format_exc()))
    helper.terminate();helper.wait(timeout=3);print(json.dumps(report),flush=True)


def launch():
    p=read(OUT/'protocol.json');assert p['authorization_approved'] and not (OUT/'launch.json').exists()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    Resource.resources();left=int(p['absolute_all_stop']-time.time()-2);assert left>1800
    cmd=['systemd-run','--user','--unit=stage-t647','--collect','--property=Type=exec','--property=CPUQuota=400%','--property=CPUWeight=10','--property=Nice=19','--property=AllowedCPUs=8 10 12 14','--property=MemoryMax=4G','--property=MemorySwapMax=0','--property=KillMode=control-group','--property=TimeoutStopSec=5',f'--property=RuntimeMaxSec={left}',f'--property=WorkingDirectory={ROOT}',f'--property=StandardOutput=append:{OUT}/driver.log','--property=StandardError=inherit','--setenv=CUDA_VISIBLE_DEVICES=',
       'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage T647','--why=Bounded successful trajectory coverage',str(ROOT/'run_stage.sh'),str(HERE/'control.py'),'driver']
    c=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True);assert c.returncode==0,c.stderr
    with (OUT/'launch.json').open('x') as f:json.dump(dict(unit='stage-t647',epoch=time.time(),absolute_all_stop=p['absolute_all_stop'],source_remote=receipt['remote_commit']),f,indent=2)
    print(c.stderr.strip())

if __name__=='__main__':globals()[sys.argv[1]]()
