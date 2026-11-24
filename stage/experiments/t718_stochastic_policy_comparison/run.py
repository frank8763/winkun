"""One bounded fixed-capacity successful-behavior augmentation experiment."""
from pathlib import Path
import fcntl, importlib.util, json, os, shutil, signal, subprocess, sys, time, traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t718_stochastic_policy_comparison';DOC=ROOT/'stage/docs/day_20261124/t718';UNIT='stage-t718.service'
if __name__=='__main__' and sys.argv[1]=='helper':
    spec=importlib.util.spec_from_file_location('t718_resource_helper',ROOT/'stage/experiments/t699_unexecuted_continuation/run.py')
    H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
    test=len(sys.argv)>2 and sys.argv[2]=='preflight'
    H.OUT=OUT/'helper_preflight' if test else OUT;H.DOC=DOC/'helper_preflight' if test else DOC
    H.UNIT='stage-t718-helper-test.service' if test else UNIT
    H.helper();sys.exit(0)
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'stage/experiments/t701_compact_retention'))
spec=importlib.util.spec_from_file_location('t718_native_controller',ROOT/'stage/experiments/t701_compact_retention/run.py')
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
    sys.path.insert(0,str(HERE));import t718_native as RN
    RN.worker(pipe,index,p,counts)
M.worker=worker
_root=None;_outside=None;_resource_time=0.

def unit_stopped(unit):
    s=subprocess.run(['systemctl','--user','show',unit,'-p','MainPID','-p','ActiveState'],capture_output=True,text=True)
    assert 'MainPID=0' in s.stdout and ('ActiveState=inactive' in s.stdout or 'ActiveState=failed' in s.stdout),(unit,s.stdout,s.stderr)

def old_stopped():
    spec=importlib.util.spec_from_file_location('t718_previous_guard',ROOT/'stage/experiments/t717_onpolicy_recovery/run.py')
    G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G);G.stopped();unit_stopped('stage-t717-finalizer.service')
    assert read(ROOT/'stage/docs/day_20261124/t717/FINALIZER_DONE.json')['state']=='STOPPED'

def stopped():
    old_stopped()
    for unit in [UNIT,'stage-t718-helper-test.service','stage-t718-preflight.service','stage-t718-preflight-recovery.service','stage-t718-preflight-recovery2.service']:unit_stopped(unit)
    if (OUT/'DRIVER_STARTED.json').exists():
        s=read(OUT/'SEALED.json');assert s['all_scientific_processes_stopped'] and s['helper_stopped']
        assert all(not Path('/proc',str(i)).exists() for i in s['scientific_pids']+[read(OUT/'DRIVER_STARTED.json')['pid']])
    else:assert not (OUT/'launch.json').exists()

def root_plan():
    global _root
    if _root is None:_root=read(OUT/'protocol.json')
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
    assert _outside+own<root['global_bytes'] and shutil.disk_usage(ROOT).free>40*2**30
    if p.get('out')!=str(OUT):assert used(p['out'])<root['child_bytes']
    if now-_resource_time>1 or force:
        M.resource_wait(min(root['absolute_all_stop'],p['absolute_all_stop']));_resource_time=time.time()
    for f,h in root['inputs'].items():
        q=Path(f);assert [q.stat().st_size,q.stat().st_mtime_ns]==root['input_stats'][f],f
        if force:assert sha(q)==h,f
M.verify=R.verify=verify

def native(action,name):
    import t718_native as RN
    RN.install();M.native_child(action,name)

def call_counts():
    totals={k:0 for k in ['Gradient_live','Gradient_shadow','Gradient_independent','Optimizer_live','Optimizer_shadow','Optimizer_independent','Radius_live','Radius_shadow','Radius_independent']}
    f=OUT/'fit/fit_calls.jsonl'
    if f.exists():
        for line in f.read_text().splitlines():
            r=json.loads(line)
            if r['stage']=='BEFORE':totals[dict(gradient='Gradient_',optimizer='Optimizer_',radius='Radius_')[r['kind']]+r['who']]+=1
    return totals

def account(items,state):
    totals=dict(physics=sum(i.get('physics',0) for i in items),commands=sum(i.get('commands',0) for i in items),
        science_seconds=sum(i['elapsed_seconds'] for i in items if not i['audit']),audit_seconds=sum(i['elapsed_seconds'] for i in items if i['audit']),**call_counts())
    p=root_plan()
    assert totals['physics']<=p['max_physics'] and totals['commands']<=p['max_commands']
    for who in ['live','shadow','independent']:
        assert totals['Gradient_'+who]==0 and totals['Optimizer_'+who]==0 and totals['Radius_'+who]==0
    save(OUT/'LEDGER.json',dict(state=state,children=items,totals=totals,new_Adam=0,new_RL=0,new_Value=0))
    ledger=read(C.WEEKEND/'LEDGER.json');job=next(j for j in ledger['new_tuesday_experiments'] if j['id']=='T718');before=job.get('accounted',{})
    for key in ['physics','commands','science_seconds','audit_seconds']:
        value=totals[key];assert value>=before.get(key,0);ledger[key]+=value-before.get(key,0)
    assert ledger['physics']<=1500000000 and ledger['commands']<=45000 and ledger['science_seconds']<=180000 and ledger['audit_seconds']<=18000
    job.update(state=state,accounted=totals);save(C.WEEKEND/'LEDGER.json',ledger)

def register():
    stopped();assert read(DOC/'PREFLIGHT.json')['state']=='PASS' and read(DOC/'NATIVE_PREFLIGHT.json')['state']=='PASS' and not (OUT/'protocol.json').exists()
    old=read(ROOT/'stage/runs/t717_onpolicy_recovery/protocol.json');p=read(OUT/'prepared.json');auth=read(DOC/'AUTHORIZATION.json');now=time.time()
    p.update(id='T718_FROZEN_STOCHASTIC_VS_EXPLORATION_CONTROL',out=str(OUT),registration_epoch=now,absolute_learning_stop=auth['absolute_learning_stop'],absolute_native_stop=auth['absolute_native_stop'],absolute_all_stop=auth['absolute_all_stop'],max_bytes=512*2**20,child_bytes=256*2**20,global_bytes=10*2**30,new_Actor=0,new_RL=0,new_Value=0,new_optimizer=0,arms=[],schedule=[['recovery','baseline'],['recovery','trained']],new_main_seconds=900,new_audit_seconds=180,max_physics=12*sum(j['case']['step_cap'] for j in p['recovery_jobs']),max_commands=244,retained='T698',conditional_stochastic_comparison_only=True)
    assert now+300+2*(900+180+60)+180+300+120<auth['absolute_all_stop'] and now+300+2*(900+180)+60<auth['absolute_native_stop']
    assert used(ROOT/'stage/runs')+p['max_bytes']<8*2**30
    ledger=read(C.WEEKEND/'LEDGER.json');assert not any(j['id']=='T718' for j in ledger.get('new_tuesday_experiments',[]))
    for k,x,cap in [('physics',p['max_physics'],1500000000),('commands',244,45000),('science_seconds',1800,180000),('audit_seconds',360,18000)]:assert ledger[k]+x<=cap
    inputs=dict(old['inputs']);extra=set(HERE.glob('*.py'))|{DOC/f for f in ['PLAN.md','AUTHORIZATION.json','PREFLIGHT.json','NATIVE_PREFLIGHT.json','FIXED_SELECTION.json','T717_RATIONALE.json','PRIMARY_REVIEW.md']}|{ROOT/'stage/runs/t717_onpolicy_recovery/SEALED.json',ROOT/'stage/docs/day_20261124/t717/FINALIZER_DONE.json',ROOT/'stage/runs/t717_onpolicy_recovery/training/audit.json',ROOT/'stage/runs/t717_onpolicy_recovery/validation/audit.json'}
    for model in p['models'].values():extra.add(Path(model['path']))
    inputs.update({str(f):sha(f) for f in extra});entries=[[f,h,Path(f).stat().st_size,Path(f).stat().st_mtime_ns] for f,h in sorted(inputs.items())];(HERE/'inputs.json').write_text(json.dumps(entries,separators=(',',':'))+'\n');assert (HERE/'inputs.json').stat().st_size<2**21
    p['_input_manifest']=dict(path=str(HERE/'inputs.json'),sha256=sha(HERE/'inputs.json'));p.pop('inputs',None);p.pop('input_stats',None);blob=json.dumps(p,separators=(',',':'))+'\n';assert len(blob.encode())<2**21;(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob)
    ledger.setdefault('new_tuesday_experiments',[]).append(dict(id='T718',state='RESERVED',out=str(OUT),max_physics=p['max_physics'],max_commands=244,new_learning=0));save(C.WEEKEND/'LEDGER.json',ledger);save(DOC/'TRANSPORT_LEDGER.json',dict(cap=900,charged_seconds=0,source_commits=[],attempts=[]))
    save(DOC/'REGISTERED.json',dict(state='REGISTERED_NOT_LAUNCHED',inputs=len(inputs),max_physics=p['max_physics'],max_commands=244,absolute_native_stop=p['absolute_native_stop'],absolute_all_stop=p['absolute_all_stop'],new_learning=0));verify(read(OUT/'protocol.json'),force=True);print('T718_REGISTERED',len(inputs),flush=True)

def sync(final=False):
    stopped();p=root_plan();assert time.time()+300<p['absolute_all_stop']
    l=read(DOC/'TRANSPORT_LEDGER.json');assert l['charged_seconds']+300+(0 if final else 300)<=l['cap']
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T718 source300')));signal.alarm(295)
    sys.path.insert(0,str(ROOT/'stage/experiments/weekend_20261120_23'));import archive_after_t695 as G
    G.stopped=G.Client.A.stopped=stopped;G.Client.client()
    sys.path.insert(0,str(ROOT/'stage/experiments/t626_multitask_closed_loop'));import t626_sync_source
    t626_sync_source.main();stopped();signal.alarm(0)

def launch():
    stopped();p=root_plan();verify(p,force=True);assert not (OUT/'launch.json').exists()
    r=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip();assert r['root_tree']==tree
    # Recheck the full worst path after transport; no rolling absolute deadline.
    fit=0;native=2*(900+180+60)
    assert time.time()+fit+native+600<p['absolute_all_stop'] 
    cmd=['systemd-run','--user','--unit=stage-t718','--collect','--property=Type=exec','--property=CPUQuota=800%','--property=CPUWeight=10','--property=AllowedCPUs=8 9 10 11 12 13 14 15','--property=Nice=19','--property=MemoryMax=8G','--property=MemorySwapMax=0','--property=KillMode=control-group','--property=TimeoutStopSec=5',f'--property=RuntimeMaxSec={int(p["absolute_all_stop"]-time.time()-3)}',f'--property=WorkingDirectory={ROOT}',f'--property=StandardOutput=append:{OUT}/driver.log','--property=StandardError=inherit','--setenv=CUDA_VISIBLE_DEVICES=','--setenv=OMP_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1','systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage T718','--why=Bounded retention tradeoff',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'driver']
    c=subprocess.run(cmd,capture_output=True,text=True);assert c.returncode==0,c.stderr
    save(OUT/'launch.json',dict(epoch=time.time(),unit=UNIT,remote_commit=r['remote_commit'],root_tree=tree));print(c.stderr,flush=True)

def driver():
    p=root_plan();verify(p,force=True);assert not (OUT/'DRIVER_STARTED.json').exists();os.sched_setaffinity(0,{15})
    save(OUT/'DRIVER_STARTED.json',dict(pid=os.getpid(),epoch=time.time()))
    log=(OUT/'helper.log').open('xb');helper=subprocess.Popen(['/usr/bin/python3',str(HERE/'run.py'),'helper'],stdout=log,stderr=log)
    ids=[helper.pid];items=[];error=None
    def action(command,folder,name,seconds,audit,physical=None):
        assert helper.poll() is None and time.time()+seconds+10<p['absolute_all_stop'];verify(p,force=True)
        with (folder/(name+'.log')).open('xb') as f:
            child=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=f,start_new_session=True);ids.append(child.pid);start=time.time()
            save(OUT/'status.json',dict(state='ACTIVE',child=name,action='audit' if audit else 'main',child_pid=child.pid,driver_pid=os.getpid(),epoch=start,absolute_action_stop=start+seconds))
            try:child.wait(timeout=seconds);assert child.returncode==0,(name,child.returncode)
            except BaseException:
                R.scoped_signal(child.pid,signal.SIGCONT);R.scoped_signal(child.pid,signal.SIGTERM)
                try:child.wait(timeout=3)
                except subprocess.TimeoutExpired:R.scoped_signal(child.pid,signal.SIGKILL);child.wait()
                raise
            finally:
                item=dict(name=name,pid=child.pid,audit=audit,returncode=child.returncode,elapsed_seconds=time.time()-start)
                if physical:
                    summary=read(folder/'summary.json') if (folder/'summary.json').exists() else {}
                    item.update(physics=summary.get('physics',physical['max_physics']),commands=summary.get('commands',physical['max_commands']),conservative_upper_if_summary_missing=not bool(summary))
                items.append(item);account(items,'ACTIVE')
    try:
        until=time.time()+5
        while not (DOC/'AFFINITY_GUARD.json').exists() and time.time()<until:time.sleep(.05)
        assert read(DOC/'AFFINITY_GUARD.json')['pid']==helper.pid
        for arm in ['baseline','trained']:
            name='recovery_'+arm;folder=OUT/name;folder.mkdir();(folder/'episodes').mkdir()
            jobs=p['replicated_jobs'];maximum=2*sum(j['case']['step_cap'] for j in jobs)
            assert time.time()+900+180+60<p['absolute_all_stop'] and time.time()+900<p['absolute_native_stop']
            child=dict(arm=name,suite='new',model_name=arm,out=str(folder),_root_sha256=sha(OUT/'protocol.json'),main_seconds=900,audit_seconds=180,child_bytes=p['child_bytes'],max_physics=maximum,max_commands=2*len(jobs)+14,new_jobs=jobs,new_exact_jobs=jobs,absolute_native_stop=min(time.time()+900,p['absolute_native_stop']),absolute_all_stop=min(time.time()+1140,p['absolute_all_stop']))
            save(folder/'protocol.json',child)
            for mode,seconds in [('evaluate',900),('audit',180)]:action([str(ROOT/'run_stage.sh'),str(HERE/'run.py'),mode,name],folder,mode,seconds,mode=='audit',child if mode=='evaluate' else None)
            assert read(folder/'audit.json')['state']=='PASS'
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        (OUT/'HELPER_STOP').touch()
        try:helper.wait(timeout=5)
        except subprocess.TimeoutExpired:helper.terminate();helper.wait(timeout=3)
        log.close()
        for f in OUT.glob('*/worker_generations.json'):
            for generation in read(f):ids.extend([generation['parent'],*generation['workers']])
        alive=[pid for pid in set(ids) if Path('/proc',str(pid)).exists()];state='FAILED' if error or alive else 'COMPLETE'
        account(items,state);report=dict(state=state,error=error,scientific_pids=sorted(set(ids)),all_scientific_processes_stopped=not alive,helper_stopped=helper.poll() is not None,retained='T698',new_Adam=0,new_RL=0,new_Value=0,epoch=time.time())
        for path in [OUT/'SEALED.json',OUT/'FINAL_REPORT.json',OUT/'status.json',DOC/'FINAL_REPORT.json']:save(path,report)

if __name__=='__main__':
    mode=sys.argv[1]
    if mode in ['evaluate','audit']:native(mode,sys.argv[2])
    else:{'register':register,'sync':sync,'launch':launch,'driver':driver}[mode]()
