"""One bounded fixed-capacity successful-behavior augmentation experiment."""
from pathlib import Path
import fcntl, importlib.util, json, os, shutil, signal, subprocess, sys, time, traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t721_credit_assignment';DOC=ROOT/'stage/docs/overnight_20261124_25/t721';UNIT='stage-t721.service'
if __name__=='__main__' and sys.argv[1]=='helper':
    spec=importlib.util.spec_from_file_location('t721_resource_helper',ROOT/'stage/experiments/t699_unexecuted_continuation/run.py')
    H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
    test=len(sys.argv)>2 and sys.argv[2]=='preflight'
    H.OUT=OUT/'helper_preflight' if test else OUT;H.DOC=DOC/'helper_preflight' if test else DOC
    H.UNIT='stage-t721-helper-test.service' if test else UNIT
    H.helper();sys.exit(0)
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'stage/experiments/t701_compact_retention'))
spec=importlib.util.spec_from_file_location('t721_native_controller',ROOT/'stage/experiments/t701_compact_retention/run.py')
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
    spec=importlib.util.spec_from_file_location('t721_previous_guard',ROOT/'stage/experiments/t720_complete_exposed_panels/run.py')
    G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G);G.stopped()
    unit_stopped('stage-t720-finalizer.service')
    assert read(ROOT/'stage/docs/day_20261124/t720/FINALIZER_DONE.json')['state']=='STOPPED'

def stopped():
    old_stopped()
    for unit in [UNIT,'stage-t721-helper-test.service','stage-t721-preflight.service','stage-t721-preflight-recovery.service','stage-t721-preflight-recovery2.service']:unit_stopped(unit)
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
    import t721_runtime as T;getattr(T,action)(name)

def counts():
    values={k+'_'+who:0 for k in ['Actor_Adam','Value_Adam'] for who in ['live','shadow','independent']}
    for f in [DOC/'PREFLIGHT_CALLS.jsonl',OUT/'fit_calls.jsonl']:
        if f.exists():
            for line in f.read_text().splitlines():
                r=json.loads(line)
                if r['stage']=='BEFORE':values[('Actor_Adam' if r['kind']=='actor' else 'Value_Adam')+'_'+r['who']]+=1
    return values

def account(items,state):
    p=root_plan();totals=dict(physics=sum(x.get('physics',0) for x in items),commands=sum(x.get('commands',0) for x in items),science_seconds=sum(x['elapsed_seconds'] for x in items if not x['audit']),audit_seconds=sum(x['elapsed_seconds'] for x in items if x['audit']),**counts())
    assert totals['physics']<=p['max_physics'] and totals['commands']<=p['max_commands'];assert all(v<=98 for k,v in totals.items() if '_Adam_' in k)
    save(OUT/'LEDGER.json',dict(state=state,children=items,totals=totals,old_4096_counters_unchanged=True))
    l=read(C.WEEKEND/'LEDGER.json');j=next(j for j in l['new_tuesday_experiments'] if j['id']=='T721');before=j.get('accounted',{})
    for k in ['physics','commands','science_seconds','audit_seconds']:assert totals[k]>=before.get(k,0);l[k]+=totals[k]-before.get(k,0)
    j.update(state=state,accounted=totals)
    for k,cap in [('physics',1500000000),('commands',45000),('science_seconds',180000),('audit_seconds',18000)]:assert l[k]<=cap
    save(C.WEEKEND/'LEDGER.json',l)

def schedule(p):
    steps=[]
    # Alternate complete four-round blocks; state continuity is per arm.
    for segment in range(3):
        for arm,lam in [('gae95',.95),('mc1',1.)]:
            jobs=[j for pair in zip(p['training_original_jobs'],p['training_jobs']) for j in pair]
            physics=4*sum(j['case']['step_cap'] for j in jobs)+2*sum(j['case']['step_cap'] for j in p['training_original_jobs'])+2*sum(j['case']['step_cap'] for j in p['training_jobs'])
            steps.append(dict(name=f'{arm}_train{segment}',kind='train',arm=arm,gae_lambda=lam,round_start=segment*4,rounds=4,previous_child=f'{arm}_train{segment-1}' if segment else None,main_seconds=2400,audit_seconds=600,max_physics=physics,max_commands=200))
    for panel,main,audit in [('old403',3000,360),('exposed174',2700,300),('recovery18',600,150),('prospective24',750,180)]:
        for arm in (['B0','gae95','mc1'] if panel=='prospective24' else ['gae95','mc1']):
            x=p['panels'][panel];steps.append(dict(name=f'{arm}_{panel}',kind='evaluate',arm=arm,panel=panel,main_seconds=main,audit_seconds=audit,max_physics=sum(j['case']['step_cap'] for j in x['jobs']+x['exact_jobs']),max_commands=len(x['jobs'])+len(x['exact_jobs'])+14))
    return steps

def full_time_admission(p,remaining,pre_source=0):
    now=time.time()+pre_source;cursor=now
    for x in remaining:
        if x['kind']=='train':assert cursor+x['main_seconds']<p['absolute_learning_stop']
        assert cursor+x['main_seconds']<p['absolute_native_stop']
        cursor+=x['main_seconds']+x['audit_seconds']+60
    assert cursor+780<p['absolute_all_stop'],'Full final audits/source/backup reserve missing'
    return dict(start=now,worst_scientific_end=cursor,worst_all_end=cursor+780)

def register():
    stopped();p=read(OUT/'prepared.json');a=read(DOC/'AUTHORIZATION.json');assert read(DOC/'PREFLIGHT.json')['state']=='PASS' and read(DOC/'NATIVE_PREFLIGHT.json')['state']=='PASS';assert not (OUT/'protocol.json').exists()
    p.update(a);p.update(id='T721_MATCHED_ACTOR_CREDIT_GAE95_VS_MC1',out=str(OUT),registration_epoch=time.time(),retained='T698',new_RL=True,max_Actor_Adam_each=98,max_Value_Adam_each=98)
    p['schedule']=schedule(p);p['max_physics']=sum(x['max_physics'] for x in p['schedule']);p['max_commands']=sum(x['max_commands'] for x in p['schedule']);p['full_time_admission']=full_time_admission(p,p['schedule'],300)
    assert p['max_physics']<=92794924 and p['max_commands']<=2800
    assert used(ROOT/'stage/runs')+p['max_bytes']+128*2**20<8*2**30
    ledger=read(C.WEEKEND/'LEDGER.json');assert not any(j['id']=='T721' for j in ledger.get('new_tuesday_experiments',[]))
    for k,x,cap in [('physics',p['max_physics'],1500000000),('commands',p['max_commands'],45000),('science_seconds',29250+3600,180000),('audit_seconds',5760,18000)]:assert ledger[k]+x<=cap
    data={k:p.pop(k) for k in list(p) if isinstance(p[k],(dict,list)) and len(json.dumps(p[k]))>20000}
    study=HERE/'study_data.json';study.write_text(json.dumps(data,separators=(',',':'))+'\n');p['_study_data']=dict(path=str(study),sha256=sha(study))
    inputs=dict(read(ROOT/'stage/runs/t720_complete_exposed_panels/protocol.json')['inputs']);extra=set(HERE.glob('*.py'))|{study}|{DOC/f for f in ['PLAN.md','AUTHORIZATION.json','PRIMARY_REVIEW.md','PREFLIGHT.json','PREFLIGHT_CALLS.jsonl','NATIVE_PREFLIGHT.json','PROSPECTIVE_INITIALS.json','STORAGE_BOUND.json','AFFINITY_PREPARATION_RECOVERY.json','PREPARATION_ACCOUNTING.json']}|{Path(p['learner_initial']['path']),ROOT/'stage/runs/t701r_native_entry/known_replay_refined108/summary.json',ROOT/'stage/runs/t720_complete_exposed_panels/SEALED.json',ROOT/'stage/docs/day_20261124/t720/FINALIZER_DONE.json'}
    for panel in data.get('panels',p.get('panels',{})).values():
        for r in panel['baseline_rows']:extra.add(Path(r['path']))
    for j in data.get('recovery_jobs',p.get('recovery_jobs',[])):extra.add(Path(j['source_path']))
    inputs.update({str(f):sha(f) for f in extra});entries=[[f,h,Path(f).stat().st_size,Path(f).stat().st_mtime_ns] for f,h in sorted(inputs.items())];(HERE/'inputs.json').write_text(json.dumps(entries,separators=(',',':'))+'\n')
    p['_input_manifest']=dict(path=str(HERE/'inputs.json'),sha256=sha(HERE/'inputs.json'));p.pop('inputs',None);p.pop('input_stats',None);blob=json.dumps(p,separators=(',',':'))+'\n';assert len(blob.encode())<2**21;(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob)
    # Preparation is conservatively charged once; optimizer counts remain distinct new-study counters.
    ledger['science_seconds']+=3600;ledger.setdefault('new_tuesday_experiments',[]).append(dict(id='T721',state='RESERVED',out=str(OUT),max_physics=p['max_physics'],max_commands=p['max_commands'],new_independent_Actor_Adam_each=98,new_independent_Value_Adam_each=98,preparation_science_seconds=3600));save(C.WEEKEND/'LEDGER.json',ledger)
    save(DOC/'TRANSPORT_LEDGER.json',dict(cap=900,charged_seconds=0,source_commits=[],attempts=[]));save(DOC/'REGISTERED.json',dict(state='REGISTERED_NOT_LAUNCHED',inputs=len(inputs),physics=p['max_physics'],commands=p['max_commands'],source_pending=True,full_time_admission=p['full_time_admission'],absolute_learning_stop=p['absolute_learning_stop'],absolute_native_stop=p['absolute_native_stop'],absolute_all_stop=p['absolute_all_stop']))
    verify(read(OUT/'protocol.json'),force=True);print('T721_REGISTERED',len(inputs),flush=True)

def sync(final=False):
    stopped();p=root_plan();assert time.time()+300<p['absolute_all_stop'];l=read(DOC/'TRANSPORT_LEDGER.json');assert l['charged_seconds']+300+(0 if final else 300)<=900
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T721 source300')));signal.alarm(295)
    sys.path.insert(0,str(ROOT/'stage/experiments/weekend_20261120_23'));import archive_after_t695 as G
    G.stopped=G.Client.A.stopped=stopped;G.Client.client();sys.path.insert(0,str(ROOT/'stage/experiments/t626_multitask_closed_loop'));import t626_sync_source;t626_sync_source.main();stopped();signal.alarm(0)

def launch():
    stopped();p=root_plan();verify(p,force=True);assert not (OUT/'launch.json').exists();full_time_admission(p,p['schedule']);receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip();assert receipt['root_tree']==tree
    cmd=['systemd-run','--user','--unit=stage-t721','--collect','--property=Type=exec','--property=CPUQuota=800%','--property=CPUWeight=10','--property=AllowedCPUs=8 9 10 11 12 13 14 15','--property=Nice=19','--property=MemoryMax=8G','--property=MemorySwapMax=0','--property=KillMode=control-group','--property=TimeoutStopSec=5',f'--property=RuntimeMaxSec={int(p["absolute_all_stop"]-time.time()-3)}',f'--property=WorkingDirectory={ROOT}',f'--property=StandardOutput=append:{OUT}/driver.log','--property=StandardError=inherit','--setenv=CUDA_VISIBLE_DEVICES=','--setenv=OMP_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1','systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage T721','--why=Bounded overnight credit comparison',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'driver']
    c=subprocess.run(cmd,capture_output=True,text=True);assert c.returncode==0,c.stderr;save(OUT/'launch.json',dict(epoch=time.time(),unit=UNIT,remote_commit=receipt['remote_commit'],root_tree=tree));print(c.stderr,flush=True)

def driver():
    p=root_plan();verify(p,force=True);assert not (OUT/'DRIVER_STARTED.json').exists();os.sched_setaffinity(0,{15});save(OUT/'DRIVER_STARTED.json',dict(pid=os.getpid(),epoch=time.time()))
    log=(OUT/'helper.log').open('xb');helper=subprocess.Popen(['/usr/bin/python3',str(HERE/'run.py'),'helper'],stdout=log,stderr=log);ids=[helper.pid];items=[];error=None
    def action(mode,name,seconds,audit,physical=None):
        assert helper.poll() is None;verify(p,force=True);folder=OUT/name
        with (folder/(mode+'.log')).open('xb') as f:
            child=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'run.py'),mode,name],cwd=ROOT,stdout=f,stderr=f,start_new_session=True);ids.append(child.pid);started=time.time();save(OUT/'status.json',dict(state='ACTIVE',child=name,action=mode,child_pid=child.pid,driver_pid=os.getpid(),epoch=started,absolute_action_stop=started+seconds))
            try:child.wait(timeout=seconds);assert child.returncode==0,(mode,name,child.returncode)
            except BaseException:
                R.scoped_signal(child.pid,signal.SIGCONT);R.scoped_signal(child.pid,signal.SIGTERM)
                try:child.wait(timeout=3)
                except subprocess.TimeoutExpired:R.scoped_signal(child.pid,signal.SIGKILL);child.wait()
                raise
            finally:
                x=dict(name=name,mode=mode,pid=child.pid,audit=audit,returncode=child.returncode,elapsed_seconds=time.time()-started)
                if physical:
                    sm=read(folder/'summary.json') if (folder/'summary.json').exists() else {};x.update(physics=sm.get('physics',physical['max_physics']),commands=sm.get('commands',physical['max_commands']),conservative_upper_if_summary_missing=not bool(sm))
                items.append(x);account(items,'ACTIVE')
    try:
        until=time.time()+5
        while not (DOC/'AFFINITY_GUARD.json').exists() and time.time()<until:time.sleep(.05)
        assert read(DOC/'AFFINITY_GUARD.json')['pid']==helper.pid
        for index,specification in enumerate(p['schedule']):
            full_time_admission(p,p['schedule'][index:]);x=dict(specification);folder=OUT/x['name'];folder.mkdir();(folder/'episodes').mkdir();now=time.time();x.update(out=str(folder),_root_sha256=sha(OUT/'protocol.json'),absolute_native_stop=min(now+x['main_seconds'],p['absolute_native_stop']),absolute_all_stop=min(now+x['main_seconds']+x['audit_seconds']+60,p['absolute_all_stop']))
            if x['kind']=='evaluate':
                x.update(p['panels'][x['panel']]);x['evaluation_model']=p['learner_initial'] if x['arm']=='B0' else read(OUT/f'{x["arm"]}_train2/summary.json')['model'];assert x['arm']=='B0' or read(OUT/f'{x["arm"]}_train2/audit.json')['state']=='PASS'
            save(folder/'protocol.json',x);main='train' if x['kind']=='train' else 'evaluate';audit='train_audit' if x['kind']=='train' else 'evaluate_audit'
            action(main,x['name'],x['main_seconds'],False,x);action(audit,x['name'],x['audit_seconds'],True);assert read(folder/'audit.json')['state']=='PASS'
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
    if mode in ['train','train_audit','evaluate','evaluate_audit']:
        import t721_runtime as T;getattr(T,mode)(sys.argv[2])
    else:{'register':register,'sync':sync,'launch':launch,'driver':driver}[mode]()
