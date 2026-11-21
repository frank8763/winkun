"""Finite zero-native response diagnostic and independent reconstruction."""
from pathlib import Path
import ast,json,os,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t677_reference_target_contract';DOC=ROOT/'stage/docs/weekend_20261120_23/t677'
sys.path.insert(0,str(ROOT/'stage/experiments/t666_failure_full_history'))
from status_io import save
def read(f):return json.loads(Path(f).read_text())
def sha(f):
    import hashlib
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def prepare():
    import t677_prepare as P
    P.main()


def register():
    assert not OUT.exists();p=read(HERE/'protocol.json');assert not p['authorization_approved']
    assert read(DOC/'PREFLIGHT.json')['state']=='PASS_ZERO_NATIVE_ZERO_OPTIMIZER'
    for f,h in p['inputs'].items():assert sha(f)==h,f
    l=read(DOC.parent/'LEDGER.json');auth=read(DOC.parent/'AUTHORIZATION.json')
    assert 'T677' not in [j['id'] for j in l['jobs']]
    assert l['science_seconds']+p['preflight_seconds']+600<=auth['max_science_seconds'] and l['audit_seconds']+600<=auth['max_audit_seconds']
    l['science_seconds']+=p['preflight_seconds']
    l['jobs'].append(dict(id='T677',state='RESERVED',max_physics=0,max_commands=0,main_seconds=600,audit_seconds=600,preflight_seconds=p['preflight_seconds'],out=str(OUT)))
    save(DOC.parent/'LEDGER.json',l);p.update(authorization_approved=True,registration_epoch=time.time())
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);save(DOC/'REGISTERED.json',dict(state='REGISTERED_NOT_LAUNCHED',epoch=time.time()))
    print('T677_REGISTERED_NOT_LAUNCHED')


def launch():
    p=read(OUT/'protocol.json');assert p['authorization_approved'] and not (OUT/'launch.json').exists()
    import t677_analysis as A
    A.B.T.Wait.Raw.resources()
    r=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert r['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    left=int(p['absolute_all_stop']-time.time()-2);assert left>p['main_seconds']+p['audit_seconds']
    args=['systemd-run','--user','--unit=stage-t677','--collect','--property=Type=exec','--property=CPUQuota=400%',
        '--property=CPUWeight=10','--property=Nice=19','--property=AllowedCPUs=14','--property=MemoryMax=4G',
        '--property=MemorySwapMax=0','--property=KillMode=control-group','--property=TimeoutStopSec=5',
        f'--property=RuntimeMaxSec={left}',f'--property=WorkingDirectory={ROOT}',f'--property=StandardOutput=append:{OUT}/driver.log',
        '--property=StandardError=inherit','--setenv=CUDA_VISIBLE_DEVICES=',str(ROOT/'run_stage.sh'),str(HERE/'control.py'),'driver']
    c=subprocess.run(args,cwd=ROOT,capture_output=True,text=True);assert c.returncode==0,c.stderr
    save(OUT/'launch.json',dict(epoch=time.time(),unit='stage-t677',source_remote=r['remote_commit']));print(c.stderr.strip())


def driver():
    p=read(OUT/'protocol.json');assert p['authorization_approved'] and not (OUT/'DRIVER_STARTED.json').exists()
    os.sched_setaffinity(0,{14});save(OUT/'DRIVER_STARTED.json',dict(pid=os.getpid(),epoch=time.time(),new_native=0,new_optimizer=0))
    error=None;elapsed={};ids=[]
    try:
        for action in ['main','audit']:
            for f,h in p['inputs'].items():assert sha(f)==h,f
            assert time.time()+p[action+'_seconds']<p['absolute_all_stop']
            with (OUT/(action+'.log')).open('xb') as f:
                c=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'t677_analysis.py'),action],cwd=ROOT,stdout=f,stderr=f,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
                ids.append(c.pid);start=time.time();save(OUT/'status.json',dict(state='ACTIVE',action=action,child_pid=c.pid,epoch=time.time()))
                try:c.wait(timeout=p[action+'_seconds']);assert c.returncode==0,(action,c.returncode)
                except BaseException:
                    try:os.killpg(c.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:c.wait(timeout=3)
                    except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
                    raise
                finally:elapsed[action]=time.time()-start
    except BaseException:error=traceback.format_exc()
    gone=all(not Path('/proc',str(pid)).exists() for pid in ids)
    result=read(OUT/'RESULTS.json') if (OUT/'RESULTS.json').exists() else None
    report=dict(state='FAILED' if error else 'COMPLETE',error=error,gate=result['state'] if result else None,
        checks=result['checks'] if result else None,new_native=0,new_optimizer=0,all_scientific_processes_stopped=gone,
        science_seconds=elapsed.get('main',0),audit_seconds=elapsed.get('audit',0),epoch=time.time())
    save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=gone,scientific_pids=ids,epoch=time.time()))
    save(OUT/'FINAL_REPORT.json',report);save(DOC/'FINAL_REPORT.json',report)
    l=read(DOC.parent/'LEDGER.json');item=next(j for j in l['jobs'] if j['id']=='T677');assert item['state']=='RESERVED'
    item.update(state=report['state'],gate=report['gate'],actual_physics=0,actual_commands=0,science_seconds=report['science_seconds'],audit_seconds=report['audit_seconds'])
    l['science_seconds']+=report['science_seconds'];l['audit_seconds']+=report['audit_seconds'];save(DOC.parent/'LEDGER.json',l)
    v=read(DOC.parent/'CAMPAIGN_STATE.json');v.update(epoch=time.time(),state='T677_FINISHED',active_run=str(OUT),unit='stage-t677.service',last_report=str(DOC/'FINAL_REPORT.json'),response_gate=report['gate'],next_action='Read frozen reference-target semantics, provenance decomposition/zero gradient, original44 and field nonlinear directions and independent audit. Zero-point PASS not navigation or automatic learning; next one main-variable matched pilot must be separately registered without failed boundary cost.')
    save(DOC.parent/'CAMPAIGN_STATE.json',v);save(OUT/'status.json',dict(state=report['state'],gate=report['gate'],all_science_stopped=gone,epoch=time.time()))
    print(json.dumps(report),flush=True)


if __name__=='__main__':{'prepare':prepare,'register':register,'launch':launch,'driver':driver}[sys.argv[1]]()
