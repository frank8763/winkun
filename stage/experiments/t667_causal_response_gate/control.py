"""Finite zero-native response diagnostic and independent reconstruction."""
from pathlib import Path
import ast,json,os,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t667_causal_response_gate';DOC=ROOT/'stage/docs/weekend_20261120_23/t667'
sys.path.insert(0,str(ROOT/'stage/experiments/t666_failure_full_history'))
from status_io import save
def read(f):return json.loads(Path(f).read_text())
def sha(f):
    import hashlib
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def prepare():
    assert not OUT.exists()
    import t667_analysis as A
    src=ROOT/'stage/runs/t666_failure_full_history';p0=read(src/'protocol.json')
    assert read(src/'FINAL_REPORT.json')['state']=='COMPLETE' and read(src/'audit.json')['state']=='PASS'
    assert not Path('/proc',str(read(src/'DRIVER_STARTED.json')['pid'])).exists()
    for pid in read(src/'SEALED.json')['scientific_pids']:assert not Path('/proc',str(pid)).exists()
    auth=read(DOC.parent/'AUTHORIZATION.json');assert auth['approved']
    now=time.time();p=dict(id='T667_CAUSAL_RESPONSE_FALSIFICATION_GATE',out=str(OUT),source_run=str(src),authorization_approved=False,
        main_seconds=600,audit_seconds=600,absolute_all_stop=min(now+1800,auth['absolute_all_stop']),
        max_bytes=4*2**20,global_bytes=10*2**30,finite_difference_epsilon=.001,numeric_atol=1e-8,numeric_rtol=1e-4,
        nonzero_tolerance=1e-12,clip_kink_tolerance=1e-6,max_relative_secant_residual=.25,
        max_physics=0,max_commands=0,new_Actor=0,new_Value=0,new_optimizer=0)
    os.sched_setaffinity(0,{14});A.torch.set_num_threads(1);res=A.S.Wait.Raw.resources()
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    # Numerical/dispatch fixture only; not a measured physical neighborhood.
    A.T.setup(p0);A.Native.P.frozen=A.Native.frozen
    q=A.Native.frozen(p0,p0['pairs'][0]['reference']['path']);before=q.phash()
    row=read(src/'summary.json')['rows'][0];a=A.T.R.E.N.S.arrays(row['path'])
    x=A.torch.tensor(a['states'][:4],dtype=A.torch.float64);d=A.torch.zeros_like(x);d[:,0]=1e-4
    def fun(z):return q.sequence(z)[-1].clamp(-1,1)
    y,j=A.torch.autograd.functional.jvp(fun,x,d)
    fd=(fun(x+.001*d)-fun(x-.001*d))/.002
    assert A.torch.isfinite(j).all() and A.torch.allclose(j,fd,atol=1e-8,rtol=1e-4)
    assert q.phash()==before and all(v.grad is None for v in q.parameters())
    pre=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',full_sequence_JVP_dispatch_numerical_fixture=True,
        fixture_not_physical_neighborhood=True,model_unchanged=True,all_source_science_exited=True,new_native=0,new_optimizer=0,resources=res)
    save(DOC/'PREFLIGHT.json',pre)
    deps=set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC.parent/'AUTHORIZATION.json',DOC.parent/'RETENTION_PRIMARY_SOURCES.md'}
    deps.update(Path(f) for f in p0['inputs'])
    deps.update(f for f in src.rglob('*') if f.is_file() and f.name not in ['driver.log','affinity.log'])
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):deps.add(Path(f).resolve())
    p['inputs']={str(f):sha(f) for f in sorted(deps)};save(HERE/'protocol.json',p)
    save(DOC/'BUDGET.json',{k:v for k,v in p.items() if k!='inputs'})
    print(json.dumps(pre))


def register():
    assert not OUT.exists();p=read(HERE/'protocol.json');assert not p['authorization_approved']
    assert read(DOC/'PREFLIGHT.json')['state']=='PASS_ZERO_NATIVE_ZERO_OPTIMIZER'
    for f,h in p['inputs'].items():assert sha(f)==h,f
    l=read(DOC.parent/'LEDGER.json');auth=read(DOC.parent/'AUTHORIZATION.json')
    assert 'T667' not in [j['id'] for j in l['jobs']]
    assert l['science_seconds']+600<=auth['max_science_seconds'] and l['audit_seconds']+600<=auth['max_audit_seconds']
    l['jobs'].append(dict(id='T667',state='RESERVED',max_physics=0,max_commands=0,main_seconds=600,audit_seconds=600,out=str(OUT)))
    save(DOC.parent/'LEDGER.json',l);p.update(authorization_approved=True,registration_epoch=time.time())
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);save(DOC/'REGISTERED.json',dict(state='REGISTERED_NOT_LAUNCHED',epoch=time.time()))
    print('T667_REGISTERED_NOT_LAUNCHED')


def launch():
    p=read(OUT/'protocol.json');assert p['authorization_approved'] and not (OUT/'launch.json').exists()
    r=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert r['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    left=int(p['absolute_all_stop']-time.time()-2);assert left>p['main_seconds']+p['audit_seconds']
    args=['systemd-run','--user','--unit=stage-t667','--collect','--property=Type=exec','--property=CPUQuota=400%',
        '--property=CPUWeight=10','--property=Nice=19','--property=AllowedCPUs=14','--property=MemoryMax=4G',
        '--property=MemorySwapMax=0','--property=KillMode=control-group','--property=TimeoutStopSec=5',
        f'--property=RuntimeMaxSec={left}',f'--property=WorkingDirectory={ROOT}',f'--property=StandardOutput=append:{OUT}/driver.log',
        '--property=StandardError=inherit','--setenv=CUDA_VISIBLE_DEVICES=',str(ROOT/'run_stage.sh'),str(HERE/'control.py'),'driver']
    c=subprocess.run(args,cwd=ROOT,capture_output=True,text=True);assert c.returncode==0,c.stderr
    save(OUT/'launch.json',dict(epoch=time.time(),unit='stage-t667',source_remote=r['remote_commit']));print(c.stderr.strip())


def driver():
    p=read(OUT/'protocol.json');assert p['authorization_approved'] and not (OUT/'DRIVER_STARTED.json').exists()
    os.sched_setaffinity(0,{14});save(OUT/'DRIVER_STARTED.json',dict(pid=os.getpid(),epoch=time.time(),new_native=0,new_optimizer=0))
    error=None;elapsed={};ids=[]
    try:
        for action in ['main','audit']:
            for f,h in p['inputs'].items():assert sha(f)==h,f
            assert time.time()+p[action+'_seconds']<p['absolute_all_stop']
            with (OUT/(action+'.log')).open('xb') as f:
                c=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'t667_analysis.py'),action],cwd=ROOT,stdout=f,stderr=f,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
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
    l=read(DOC.parent/'LEDGER.json');item=next(j for j in l['jobs'] if j['id']=='T667');assert item['state']=='RESERVED'
    item.update(state=report['state'],gate=report['gate'],actual_physics=0,actual_commands=0,science_seconds=report['science_seconds'],audit_seconds=report['audit_seconds'])
    l['science_seconds']+=report['science_seconds'];l['audit_seconds']+=report['audit_seconds'];save(DOC.parent/'LEDGER.json',l)
    v=read(DOC.parent/'CAMPAIGN_STATE.json');v.update(epoch=time.time(),state='T667_FINISHED',active_run=str(OUT),unit='stage-t667.service',last_report=str(DOC/'FINAL_REPORT.json'),response_gate=report['gate'],next_action='If numerical and secant gate PASS prepare single matched causal-response retention pilot; otherwise reject this mechanism without hyperparameter rescue, review actual occupancy correction')
    save(DOC.parent/'CAMPAIGN_STATE.json',v);save(OUT/'status.json',dict(state=report['state'],gate=report['gate'],all_science_stopped=gone,epoch=time.time()))
    print(json.dumps(report),flush=True)


if __name__=='__main__':{'prepare':prepare,'register':register,'launch':launch,'driver':driver}[sys.argv[1]]()
