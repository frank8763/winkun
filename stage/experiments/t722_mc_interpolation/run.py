"""One bounded fixed-capacity successful-behavior augmentation experiment."""
from pathlib import Path
import fcntl, importlib.util, json, os, shutil, signal, subprocess, sys, time, traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t722_mc_interpolation';DOC=ROOT/'stage/docs/day_20261125/t722';UNIT='stage-t722.service'
if __name__=='__main__' and sys.argv[1]=='helper':
    spec=importlib.util.spec_from_file_location('t722_resource_helper',ROOT/'stage/experiments/t699_unexecuted_continuation/run.py')
    H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
    test=len(sys.argv)>2 and sys.argv[2]=='preflight'
    H.OUT=OUT/'helper_preflight' if test else OUT;H.DOC=DOC/'helper_preflight' if test else DOC
    H.UNIT='stage-t722-helper-test.service' if test else UNIT
    H.helper();sys.exit(0)
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'stage/experiments/t701_compact_retention'))
spec=importlib.util.spec_from_file_location('t722_native_controller',ROOT/'stage/experiments/t701_compact_retention/run.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
sys.path.insert(0,str(ROOT/'stage/experiments/t703_projected_local'))
import t703_actor as A
C=M.C;read,save,sha,used=C.read,C.save,C.sha,C.used
M.OUT,M.DOC,M.HERE,M.UNIT,M.A=OUT,DOC,HERE,UNIT,A
R=M.R;R.OUT,R.DOC,R.HERE,R.UNIT=OUT,DOC,HERE,UNIT
_worker=M.worker
sys.path.insert(0,str(ROOT/'stage/experiments/t713_unseen_layouts'))
import geometry as Geo

def worker(pipe,index,p,counts):
    sys.path.insert(0,str(HERE));import acquisition_native as RN
    RN.worker(pipe,index,p,counts)
M.worker=worker
_root=None;_outside=None;_resource_time=0.

def unit_stopped(unit):
    s=subprocess.run(['systemctl','--user','show',unit,'-p','MainPID','-p','ActiveState'],capture_output=True,text=True)
    assert 'MainPID=0' in s.stdout and ('ActiveState=inactive' in s.stdout or 'ActiveState=failed' in s.stdout),(unit,s.stdout,s.stderr)

def old_stopped():
    spec=importlib.util.spec_from_file_location('t722_previous_guard',ROOT/'stage/experiments/t721_credit_assignment/run.py')
    G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G);G.stopped()
    unit_stopped('stage-t721-offline-finalizer.service')
    assert read(ROOT/'stage/docs/overnight_20261124_25/t721/FINALIZER_DONE.json')['state']=='STOPPED'

def stopped():
    old_stopped()
    for unit in [UNIT,'stage-t722-helper-test.service','stage-t722-preflight.service','stage-t722-preflight-recovery.service','stage-t722-preflight-recovery2.service']:unit_stopped(unit)
    if (OUT/'DRIVER_STARTED.json').exists():
        s=read(OUT/'SEALED.json');assert s['all_scientific_processes_stopped'] and s['helper_stopped']
        assert all(not Path('/proc',str(i)).exists() for i in s['scientific_pids']+[read(OUT/'DRIVER_STARTED.json')['pid']])
    else:assert not (OUT/'launch.json').exists()

def root_plan():
    global _root
    if _root is None:
        _root=read(OUT/'protocol.json')
        if '_study_data' in _root:
            ref=_root['_study_data'];assert sha(ref['path'])==ref['sha256'];extra=read(ref['path']);assert not (set(extra)&set(_root));_root.update(extra)
    return _root

def verify(p,*unused,force=False):
    global _outside,_resource_time
    root=root_plan();now=time.time()
    assert now<min(root['absolute_all_stop'],p['absolute_all_stop']),'Absolute stop'
    assert sha(OUT/'protocol.json')==sha(HERE/'protocol.json'),'Root protocol changed'
    own=used(OUT);assert own<root['max_bytes']
    if _outside is None or force:
        outside=used(ROOT/'stage/runs')-own
        if _outside is not None:assert outside==_outside,'Other Stage output changed during exclusive science'
        _outside=outside
    assert _outside+own<root['admission_bytes'] and shutil.disk_usage(ROOT).free>40*2**30
    if p.get('out')!=str(OUT):assert used(p['out'])<root['child_bytes']
    if now-_resource_time>1 or force:
        M.resource_wait(min(root['absolute_all_stop'],p['absolute_all_stop']));_resource_time=time.time()
    for f,h in root['inputs'].items():
        q=Path(f);assert [q.stat().st_size,q.stat().st_mtime_ns]==root['input_stats'][f],f
        if force:assert sha(q)==h,f
M.verify=R.verify=verify

def native(action,name):
    import acquisition_native as RN
    import t722_runtime as T;getattr(T,action)(name)

def schedule(p):
    result=[]
    for panel,main,audit in [('old403',3000,360),('exposed174',2700,300),('recovery18',600,150),('prospective24',750,180)]:
        x=p['panels'][panel];result.append(dict(name=panel,kind='evaluate',panel=panel,main_seconds=main,audit_seconds=audit,max_physics=sum(j['case']['step_cap'] for j in x['jobs']+x['exact_jobs']),max_commands=len(x['jobs'])+len(x['exact_jobs'])+14))
    return result

def admission(p,remaining,presource=0):
    cursor=time.time()+presource
    for x in remaining:
        assert cursor+x['main_seconds']<p['absolute_native_stop'];cursor+=x['main_seconds']+x['audit_seconds']+60
    assert cursor+300<p['absolute_all_stop']
    return cursor

def account(items,state):
    totals=dict(physics=sum(x.get('physics',0) for x in items),commands=sum(x.get('commands',0) for x in items),science_seconds=sum(x['elapsed_seconds'] for x in items if not x['audit']),audit_seconds=sum(x['elapsed_seconds'] for x in items if x['audit']),new_Actor_Adam=0,new_Value_Adam=0,new_RL=0)
    p=root_plan();assert totals['physics']<=p['max_physics'] and totals['commands']<=p['max_commands'];save(OUT/'LEDGER.json',dict(state=state,totals=totals,children=items))
    l=read(C.WEEKEND/'LEDGER.json');job=next(x for x in l['new_wednesday_experiments'] if x['id']=='T722');old=job.get('accounted',{})
    for k in ['physics','commands','science_seconds','audit_seconds']:assert totals[k]>=old.get(k,0);l[k]+=totals[k]-old.get(k,0)
    for k,cap in [('physics',1500000000),('commands',45000),('science_seconds',180000),('audit_seconds',18000)]:assert l[k]<=cap
    job.update(state=state,accounted=totals);save(C.WEEKEND/'LEDGER.json',l)

def register():
    stopped();assert read(DOC/'PREFLIGHT.json')['state']=='PASS' and not (OUT/'protocol.json').exists();p=read(OUT/'prepared.json');p.update(id='T722_FIXED_HALF_MC_B0_PARAMETER_INTERPOLATION',out=str(OUT),registration_epoch=time.time(),retained='T698');p['schedule']=schedule(p);p['max_physics']=sum(x['max_physics'] for x in p['schedule']);p['max_commands']=sum(x['max_commands'] for x in p['schedule']);p['worst_end']=admission(p,p['schedule'],300)
    l=read(C.WEEKEND/'LEDGER.json');assert not any(j['id']=='T722' for j in l.get('new_wednesday_experiments',[]))
    for k,x,cap in [('physics',p['max_physics'],1500000000),('commands',p['max_commands'],45000),('science_seconds',7350+300,180000),('audit_seconds',990,18000)]:assert l[k]+x<=cap
    data={k:p.pop(k) for k in list(p) if isinstance(p[k],(dict,list)) and len(json.dumps(p[k]))>20000};study=HERE/'study_data.json';study.write_text(json.dumps(data,separators=(',',':'))+'\n');p['_study_data']=dict(path=str(study),sha256=sha(study))
    old=read(ROOT/'stage/runs/t721_credit_assignment/protocol.json');inputs=dict(old['inputs']);extra=set(HERE.glob('*.py'))|{study}|set(DOC.glob('*.md'))|{DOC/f for f in ['AUTHORIZATION.json','MODEL_CONSTRUCTION.json','PREFLIGHT.json']}|{ROOT/'stage/runs/t721_credit_assignment/SEALED.json',ROOT/'stage/docs/overnight_20261124_25/t721/FINALIZER_DONE.json',ROOT/'stage/docs/overnight_20261124_25/t721/PAIRED_RESULTS.json',Path(p['candidate_model']['path'])}
    for panel in data['panels'].values():
        for row in panel.get('baseline_rows',[]):extra.add(Path(row['path']))
    inputs.update({str(f):sha(f) for f in extra});entries=[[f,h,Path(f).stat().st_size,Path(f).stat().st_mtime_ns] for f,h in sorted(inputs.items())];(HERE/'inputs.json').write_text(json.dumps(entries,separators=(',',':'))+'\n');p['_input_manifest']=dict(path=str(HERE/'inputs.json'),sha256=sha(HERE/'inputs.json'));blob=json.dumps(p,separators=(',',':'))+'\n';assert len(blob.encode())<2**21;(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob)
    l['science_seconds']+=300;l.setdefault('new_wednesday_experiments',[]).append(dict(id='T722',state='RESERVED',out=str(OUT),max_physics=p['max_physics'],max_commands=p['max_commands'],new_optimizer=0,preparation_science_seconds=300));save(C.WEEKEND/'LEDGER.json',l)
    save(DOC/'TRANSPORT_LEDGER.json',dict(cap=900,charged_seconds=0,source_commits=[],attempts=[]));save(DOC/'REGISTERED.json',dict(state='REGISTERED_NOT_LAUNCHED',inputs=len(inputs),max_physics=p['max_physics'],max_commands=p['max_commands'],science_episodes=692,new_learning=0,absolute_native_stop=p['absolute_native_stop'],absolute_all_stop=p['absolute_all_stop'],worst_end=p['worst_end']));verify(root_plan(),force=True);print('T722_REGISTERED',len(inputs),flush=True)

def sync():
    stopped();assert time.time()+300<root_plan()['absolute_all_stop'];signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T722 source300')));signal.alarm(295)
    sys.path.insert(0,str(ROOT/'stage/experiments/weekend_20261120_23'));import archive_after_t695 as G
    G.stopped=G.Client.A.stopped=stopped;G.Client.client()
    original=ROOT/'stage/experiments/t626_multitask_closed_loop/t626_sync_source.py';import types
    sys.path.insert(0,str(original.parent));module=types.ModuleType('T722_metadata_source');exec(compile(original.read_text().replace('assert len(b)<2*1024**2','assert len(b)<8*1024**2'),str(original),'exec'),module.__dict__);module.main();stopped();signal.alarm(0)

def launch():
    stopped();p=root_plan();verify(p,force=True);assert not (OUT/'launch.json').exists();admission(p,p['schedule']);r=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip();assert tree==r['root_tree']
    cmd=['systemd-run','--user','--unit=stage-t722','--collect','--property=Type=exec','--property=CPUQuota=800%','--property=CPUWeight=10','--property=AllowedCPUs=8 9 10 11 12 13 14 15','--property=Nice=19','--property=MemoryMax=8G','--property=MemorySwapMax=0','--property=KillMode=control-group','--property=TimeoutStopSec=5',f'--property=RuntimeMaxSec={int(p["absolute_all_stop"]-time.time()-3)}',f'--property=WorkingDirectory={ROOT}',f'--property=StandardOutput=append:{OUT}/driver.log','--property=StandardError=inherit','--setenv=CUDA_VISIBLE_DEVICES=','--setenv=OMP_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1','systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage T722','--why=Fixed midpoint frozen validation',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'driver'];c=subprocess.run(cmd,capture_output=True,text=True);assert c.returncode==0,c.stderr;save(OUT/'launch.json',dict(epoch=time.time(),unit=UNIT,remote_commit=r['remote_commit'],root_tree=tree));print(c.stderr,flush=True)

def driver():
    p=root_plan();verify(p,force=True);assert not (OUT/'DRIVER_STARTED.json').exists();os.sched_setaffinity(0,{15});save(OUT/'DRIVER_STARTED.json',dict(pid=os.getpid(),epoch=time.time()));log=(OUT/'helper.log').open('xb');helper=subprocess.Popen(['/usr/bin/python3',str(HERE/'run.py'),'helper'],stdout=log,stderr=log);ids=[helper.pid];items=[];error=None
    def action(mode,x,audit):
        assert helper.poll() is None;verify(p,force=True);folder=OUT/x['name'];seconds=x['audit_seconds'] if audit else x['main_seconds'];child=None;started=time.time()
        with (folder/(mode+'.log')).open('xb') as f:
            child=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'run.py'),mode,x['name']],cwd=ROOT,stdout=f,stderr=f,start_new_session=True);ids.append(child.pid);save(OUT/'status.json',dict(state='ACTIVE',child=x['name'],action=mode,child_pid=child.pid,driver_pid=os.getpid(),epoch=started,absolute_action_stop=started+seconds))
            try:child.wait(timeout=seconds);assert child.returncode==0,(mode,child.returncode)
            except BaseException:
                R.scoped_signal(child.pid,signal.SIGCONT);R.scoped_signal(child.pid,signal.SIGTERM)
                try:child.wait(timeout=3)
                except subprocess.TimeoutExpired:R.scoped_signal(child.pid,signal.SIGKILL);child.wait()
                raise
            finally:
                it=dict(name=x['name'],mode=mode,pid=child.pid,audit=audit,returncode=child.returncode,elapsed_seconds=time.time()-started)
                if not audit:
                    sm=read(folder/'summary.json') if (folder/'summary.json').exists() else {};it.update(physics=sm.get('physics',x['max_physics']),commands=sm.get('commands',x['max_commands']),conservative_upper_if_summary_missing=not bool(sm))
                items.append(it);account(items,'ACTIVE')
    try:
        until=time.time()+5
        while not (DOC/'AFFINITY_GUARD.json').exists() and time.time()<until:time.sleep(.05)
        assert read(DOC/'AFFINITY_GUARD.json')['pid']==helper.pid
        for index,x in enumerate(p['schedule']):
            admission(p,p['schedule'][index:]);folder=OUT/x['name'];folder.mkdir();(folder/'episodes').mkdir();now=time.time();child=dict(x,**p['panels'][x['panel']]);child.update(out=str(folder),_root_sha256=sha(OUT/'protocol.json'),evaluation_model=p['candidate_model'],absolute_native_stop=min(now+x['main_seconds'],p['absolute_native_stop']),absolute_all_stop=min(now+x['main_seconds']+x['audit_seconds']+60,p['absolute_all_stop']));save(folder/'protocol.json',child)
            action('evaluate',x,False);action('evaluate_audit',x,True);assert read(folder/'audit.json')['state']=='PASS'
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        (OUT/'HELPER_STOP').touch()
        try:helper.wait(timeout=5)
        except subprocess.TimeoutExpired:helper.terminate();helper.wait(timeout=3)
        log.close()
        for f in OUT.glob('*/worker_generations.json'):
            for g in read(f):ids.extend([g['parent'],*g['workers']])
        alive=[i for i in set(ids) if Path('/proc',str(i)).exists()];state='FAILED' if error or alive else 'COMPLETE';account(items,state);report=dict(state=state,error=error,scientific_pids=sorted(set(ids)),all_scientific_processes_stopped=not alive,helper_stopped=helper.poll() is not None,retained='T698',epoch=time.time())
        for f in [OUT/'SEALED.json',OUT/'FINAL_REPORT.json',OUT/'status.json',DOC/'FINAL_REPORT.json']:save(f,report)
if __name__=='__main__':
    mode=sys.argv[1]
    if mode in ['evaluate','evaluate_audit']:
        import t722_runtime as T;getattr(T,mode)(sys.argv[2])
    else:{'register':register,'sync':sync,'launch':launch,'driver':driver}[mode]()
