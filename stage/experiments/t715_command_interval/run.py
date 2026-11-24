"""One bounded fixed-capacity successful-behavior augmentation experiment."""
from pathlib import Path
import fcntl, importlib.util, json, os, shutil, signal, subprocess, sys, time, traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t715_command_interval';DOC=ROOT/'stage/docs/day_20261124/t715';UNIT='stage-t715.service'
if __name__=='__main__' and sys.argv[1]=='helper':
    spec=importlib.util.spec_from_file_location('t715_resource_helper',ROOT/'stage/experiments/t699_unexecuted_continuation/run.py')
    H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
    test=len(sys.argv)>2 and sys.argv[2]=='preflight'
    H.OUT=OUT/'helper_preflight' if test else OUT;H.DOC=DOC/'helper_preflight' if test else DOC
    H.UNIT='stage-t715-helper-test.service' if test else UNIT
    H.helper();sys.exit(0)
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'stage/experiments/t701_compact_retention'))
spec=importlib.util.spec_from_file_location('t715_native_controller',ROOT/'stage/experiments/t701_compact_retention/run.py')
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
    N=M.N.N;N.C.ACTIVE=p;real=N.G._real_affinity
    def pin(pid,mask):return real(pid,{p['stage_cpus'][index]}) if pid==0 and set(mask)=={index} else real(pid,mask)
    os.sched_setaffinity=pin;N.G.T.Session=Geo.Session if p.get('layout_test') else N.C.Session
    N.G.B.native_setup=(lambda folder,pc:Geo.setup(N.F,N.C.C,folder,pc)) if p.get('layout_test') else (lambda folder,pc:N.C.C.consistent_reset.setup(N.F,folder,pc,N.C.C.CANONICAL_SETUP))
    N.P.frozen=A.frozen;N.G.T.worker(pipe,index,p,counts)
M.worker=worker
_root=None;_outside=None;_resource_time=0.

def unit_stopped(unit):
    s=subprocess.run(['systemctl','--user','show',unit,'-p','MainPID','-p','ActiveState'],capture_output=True,text=True)
    assert 'MainPID=0' in s.stdout and ('ActiveState=inactive' in s.stdout or 'ActiveState=failed' in s.stdout),(unit,s.stdout,s.stderr)

def old_stopped():
    spec=importlib.util.spec_from_file_location('t715_previous_guard',ROOT/'stage/experiments/t714_success_data/run.py')
    G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)
    identity=read(DOC/'HISTORICAL_PID_IDENTITIES.json');exists=Path.exists
    def historical_identity_exists(path):
        if path.parent==Path('/proc') and path.name in identity['bounds'] and exists(path):
            try:
                values=(path/'stat').read_text().split(') ',1)[1].split()
                born=time.time()-float(Path('/proc/uptime').read_text().split()[0])+int(values[19])/os.sysconf('SC_CLK_TCK')
                if born>identity['bounds'][path.name]+5 and '/stage-' not in (path/'cgroup').read_text():return False
            except (FileNotFoundError,ProcessLookupError):return False
        return exists(path)
    try:
        Path.exists=historical_identity_exists;G.stopped()
    finally:Path.exists=exists
    for unit in ['stage-t714-finalizer.service','stage-t714-handoff-relay.service']:unit_stopped(unit)
    assert read(ROOT/'stage/runs/t714_success_data/FINAL_REPORT.json')['state']=='COMPLETE'

def stopped():
    old_stopped()
    for unit in [UNIT,'stage-t715-helper-test.service','stage-t715-preflight.service','stage-t715-preflight-recovery.service','stage-t715-preflight-recovery2.service']:unit_stopped(unit)
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
    p=root_plan();arm=read(OUT/name/'protocol.json')['model_name']
    assert read(OUT/'fit/audit.json')['finals'][arm]['state']=='PASS'
    M.native_child(action,name)

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
        assert totals['Gradient_'+who]<=80 and totals['Optimizer_'+who]<=64 and totals['Radius_'+who]==0
    save(OUT/'LEDGER.json',dict(state=state,children=items,totals=totals,new_Adam=0,new_RL=0,new_Value=0))
    ledger=read(C.WEEKEND/'LEDGER.json');job=next(j for j in ledger['new_tuesday_experiments'] if j['id']=='T715');before=job.get('accounted',{})
    for key in ['physics','commands','science_seconds','audit_seconds']:
        value=totals[key];assert value>=before.get(key,0);ledger[key]+=value-before.get(key,0)
    assert ledger['physics']<=1500000000 and ledger['commands']<=45000 and ledger['science_seconds']<=180000 and ledger['audit_seconds']<=18000
    job.update(state=state,accounted=totals);save(C.WEEKEND/'LEDGER.json',ledger)

def register():
    stopped();assert read(DOC/'ADMISSION_CHECK.json')['state']=='PASS';assert read(DOC/'PREFLIGHT.json')['state']=='PASS' and not (OUT/'protocol.json').exists()
    old=read(ROOT/'stage/runs/t714_success_data/protocol.json');p=json.loads((ROOT/'stage/runs/t714_success_data/protocol.json').read_text())
    acquisition=read(DOC/'ACQUISITION.json')
    auth=read(DOC/'AUTHORIZATION.json');test=read(OUT/'prepared_test.json');arms=['interval'];models=['B0','raw','interval']
    p.update(id='T715_FIXED_CAPACITY_EXECUTED_COMMAND_INTERVAL_SUPERVISION',out=str(OUT),registration_epoch=time.time(),
        absolute_learning_stop=auth['absolute_learning_stop'],absolute_native_stop=auth['absolute_native_stop'],absolute_all_stop=auth['absolute_all_stop'],
        max_bytes=int(2.5*2**30),child_bytes=512*2**20,global_bytes=10*2**30,arms=arms,transfer_models=models,
        gradient_cap_per_arm=80,optimizer_cap_per_arm=64,fit_main_seconds=1200,fit_audit_seconds=600,
        known_main_seconds=2800,known_audit_seconds=600,new_main_seconds=1200,new_audit_seconds=300,
        source_selection=read(ROOT/'stage/docs/day_20261124/t714/SOURCE_SELECTION.json'),layout_rules=test['layout_rules'],test_jobs=test['new_jobs'],test_exact_jobs=test['new_exact_jobs'],test_contracts=test['new_initial_contracts'],acquisition_jobs=acquisition['jobs'],acquisition_exact_jobs=acquisition['exact_jobs'],acquisition_contracts=acquisition['contracts'],
        max_Actor_live=0,max_Actor_shadow=0,max_Actor_independent=0,new_Adam=0,new_RL=0,new_Value=0,
        separate_gradient_each_cap=80,separate_optimizer_each_cap=64,separate_radius_each_cap=0,
        schedule=[[suite,arm] for arm in arms for suite in ['known','new','acquisition','exposed']]+[['test',name] for name in models],retained='T698',development_only=False,
        reused_controls={'B0':{'known':'stage/runs/t701r_native_entry/known_replay_refined108','new':'stage/runs/t701r_native_entry/new_replay_refined108'},
        'raw':{s:'stage/runs/t714_success_data/'+s+'_success' for s in ['known','new','acquisition']}})
    for key in ['replay_alphas','compact_reference','selection_order','selection_old_loss_limits']:p.pop(key,None)
    p['practical_finite_gate']=dict(max_old403_lost=20,require_test_safe_strictly_greater_than=['B0','raw'],require_test_collisions_not_greater_than=['B0','raw'],not_automatic_retention=True,not_significance_or_novelty=True,require_acquisition_safe_greater_than_raw=True)
    p['exposed_jobs']=old['test_jobs'];p['exposed_exact_jobs']=old['test_exact_jobs'];p['exposed_contracts']=old['test_contracts'];p['exposed_layout_rules']=old['layout_rules']
    p['reused_controls']['B0']['exposed']='stage/runs/t714_success_data/test_B0'
    p['reused_controls']['raw']['exposed']='stage/runs/t714_success_data/test_success'
    assert time.time()+1800+3460+1560+840+4*1560+900<p['absolute_all_stop'],'Full finite path must fit'
    assert time.time()+1800+60<p['absolute_learning_stop']
    assert time.time()+1800+3460+1560+840+4*1560-300<p['absolute_native_stop']
    known=p['evaluation_jobs']+p['screen_jobs'];dev=p['new_jobs']+p['new_exact_jobs'];fresh=p['test_jobs']+p['test_exact_jobs'];acq=p['acquisition_jobs']+p['acquisition_exact_jobs']
    p['max_physics']=sum(j['case']['step_cap'] for j in known+dev+acq+p['exposed_jobs']+p['exposed_exact_jobs'])+3*sum(j['case']['step_cap'] for j in fresh)
    p['max_commands']=len(known)+len(dev)+len(acq)+len(p['exposed_jobs'])+len(p['exposed_exact_jobs'])+56+3*(len(fresh)+14)
    p['storage_worst_bytes']=(147*p['fresh_bytes']+431*p['compact_bytes']+64*2**20)+4*(72*p['fresh_bytes']+24*p['compact_bytes']+32*2**20)+(18*p['fresh_bytes']+18*p['compact_bytes']+8*2**20)+128*2**20
    assert p['storage_worst_bytes']<p['max_bytes'] and used(ROOT/'stage/runs')+p['max_bytes']<8*2**30
    ledger=read(C.WEEKEND/'LEDGER.json')
    for key,extra,cap in [('physics',p['max_physics'],1500000000),('commands',p['max_commands'],45000),('science_seconds',1200+2800+1200+600+4800,180000),('audit_seconds',600+600+300+180+1200,18000)]:assert ledger[key]+extra<=cap,(key,ledger[key],extra)
    assert not any(j['id']=='T715' for j in ledger.get('new_tuesday_experiments',[]))
    inputs=dict(old['inputs']);extra=set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'AUTHORIZATION.json',DOC/'PREFLIGHT.json',DOC/'ADMISSION_CHECK.json',DOC/'LOSS_PREFLIGHT.json',DOC/'HISTORICAL_PID_IDENTITIES.json',DOC/'PID_IDENTITY_RECOVERY.json',DOC/'PID_IDENTITY_RECOVERY_2.json',ROOT/'stage/docs/day_20261124/t714/SOURCE_SELECTION.json',ROOT/'stage/docs/day_20261124/t714/SOURCE_QUALIFICATION.json',DOC/'ACQUISITION.json',DOC/'LAYOUTS.json',DOC/'TEST_INITIALS.json',DOC.parent/'T715_PRIMARY_REVIEW.md',ROOT/'stage/runs/t699_matched_value/DATA.pt.gz',OUT/'fit/summary.json',OUT/'fit/audit.json',ROOT/'stage/runs/t713_unseen_layouts/FINAL_REPORT.json',ROOT/'stage/experiments/t713_unseen_layouts/geometry.py'}
    # Frozen fit references are copied into mutable summary after registration;
    # immutable reference bytes are held separately in REFERENCES.json.
    extra.discard(OUT/'fit/summary.json');extra.discard(OUT/'fit/audit.json');extra.add(DOC/'REFERENCES.json')
    for row in p['source_selection']['rows']:
        for side in ['source','baseline']:
            extra.add(Path(row[side]['path']));extra.add(Path(row[side]['request']['model']))
            block=Path(row[side]['path']).parent.parent
            for file in ['summary.json','audit.json','cases.jsonl.gz']:extra.add(block/file)
    for ref in read(DOC/'REFERENCES.json')['finals'].values():extra.add(Path(ref['model']['path']))
    extra.update((ROOT/'stage/experiments/t713_unseen_layouts').glob('*.py'))
    for paths in p['reused_controls'].values():
        for folder in paths.values():
            for file in ['summary.json','audit.json','cases.jsonl.gz']:extra.add(ROOT/folder/file)
    extra.update(Path(r['path']) for r in read(DOC/'HISTORICAL_PID_IDENTITIES.json')['markers'])
    inputs.update({str(f):sha(f) for f in extra});entries=[[f,h,Path(f).stat().st_size,Path(f).stat().st_mtime_ns] for f,h in sorted(inputs.items())]
    (HERE/'inputs.json').write_text(json.dumps(entries,separators=(',',':'))+'\n');assert (HERE/'inputs.json').stat().st_size<2**21
    p['_input_manifest']=dict(path=str(HERE/'inputs.json'),sha256=sha(HERE/'inputs.json'));p.pop('inputs',None);p.pop('input_stats',None)
    blob=json.dumps(p,separators=(',',':'))+'\n';assert len(blob.encode())<2**21;(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob)
    ledger.setdefault('new_tuesday_experiments',[]).append(dict(id='T715',state='RESERVED',out=str(OUT),max_physics=p['max_physics'],max_commands=p['max_commands'],separate_gradient_each_cap=80,separate_optimizer_each_cap=64,Actor_old_cap_unchanged=True));save(C.WEEKEND/'LEDGER.json',ledger)
    save(DOC/'TRANSPORT_LEDGER.json',dict(cap=900,charged_seconds=0,source_commits=[],attempts=[]))
    save(DOC/'REGISTERED.json',dict(state='REGISTERED_NOT_LAUNCHED',arms=arms,transfer_models=models,inputs=len(inputs),max_physics=p['max_physics'],max_commands=p['max_commands'],storage_worst_bytes=p['storage_worst_bytes'],absolute_all_stop=p['absolute_all_stop'],new_authorization_not_old_extension=True))
    verify(read(OUT/'protocol.json'),force=True);print('T715_REGISTERED',len(inputs),flush=True)

def sync(final=False):
    stopped();p=root_plan();assert time.time()+300<p['absolute_all_stop']
    l=read(DOC/'TRANSPORT_LEDGER.json');assert l['charged_seconds']+300+(0 if final else 300)<=l['cap']
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T715 source300')));signal.alarm(295)
    sys.path.insert(0,str(ROOT/'stage/experiments/weekend_20261120_23'));import archive_after_t695 as G
    G.stopped=G.Client.A.stopped=stopped;G.Client.client()
    sys.path.insert(0,str(ROOT/'stage/experiments/t626_multitask_closed_loop'));import t626_sync_source
    t626_sync_source.main();stopped();signal.alarm(0)

def launch():
    stopped();p=root_plan();verify(p,force=True);assert not (OUT/'launch.json').exists()
    r=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip();assert r['root_tree']==tree
    # Recheck the full worst path after transport; no rolling absolute deadline.
    fit=1800;native=3460+1560+840+4*1560
    assert time.time()+fit+native+600<p['absolute_all_stop'] and time.time()+fit+30<p['absolute_learning_stop']
    cmd=['systemd-run','--user','--unit=stage-t715','--collect','--property=Type=exec','--property=CPUQuota=800%','--property=CPUWeight=10','--property=AllowedCPUs=8 9 10 11 12 13 14 15','--property=Nice=19','--property=MemoryMax=8G','--property=MemorySwapMax=0','--property=KillMode=control-group','--property=TimeoutStopSec=5',f'--property=RuntimeMaxSec={int(p["absolute_all_stop"]-time.time()-3)}',f'--property=WorkingDirectory={ROOT}',f'--property=StandardOutput=append:{OUT}/driver.log','--property=StandardError=inherit','--setenv=CUDA_VISIBLE_DEVICES=','--setenv=OMP_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1','systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage T715','--why=Bounded retention tradeoff',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'driver']
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
        # Each model is saved and independently audited before any later model.
        for arm in p['arms']:
            folder=OUT/'fit'/arm;folder.mkdir()
            for mode,seconds in [('main',1200),('audit',600)]:
                action([str(ROOT/'run_stage.sh'),str(HERE/'fit.py'),mode,arm],folder,arm+'_fit_'+mode,seconds,mode=='audit')
            assert read(OUT/'fit/audit.json')['finals'][arm]['state']=='PASS'
        for suite,arm in p['schedule']:
            name=suite+'_'+arm;folder=OUT/name;folder.mkdir();(folder/'episodes').mkdir()
            main=p['known_main_seconds'] if suite=='known' else (600 if suite=='acquisition' else p['new_main_seconds']);audit=p['known_audit_seconds'] if suite=='known' else (180 if suite=='acquisition' else p['new_audit_seconds'])
            jobs=p['evaluation_jobs']+p['screen_jobs'] if suite=='known' else (p['test_jobs']+p['test_exact_jobs'] if suite=='test' else (p['exposed_jobs']+p['exposed_exact_jobs'] if suite=='exposed' else (p['acquisition_jobs']+p['acquisition_exact_jobs'] if suite=='acquisition' else p['new_jobs']+p['new_exact_jobs'])))
            assert time.time()+main+audit+60<p['absolute_all_stop'] and time.time()+main+10<p['absolute_native_stop']
            child=dict(arm=name,suite=suite,model_name=arm,out=str(folder),_root_sha256=sha(OUT/'protocol.json'),main_seconds=main,audit_seconds=audit,child_bytes=p['child_bytes'],max_physics=sum(j['case']['step_cap'] for j in jobs),max_commands=len(jobs)+14,absolute_native_stop=min(time.time()+main,p['absolute_native_stop']),absolute_all_stop=min(time.time()+main+audit+60,p['absolute_all_stop']))
            if suite=='acquisition':child.update(suite='new',new_jobs=p['acquisition_jobs'],new_exact_jobs=p['acquisition_exact_jobs'],new_initial_contracts=p['acquisition_contracts'])
            if suite=='exposed':child.update(suite='new',layout_test=True,layout_rules=p['exposed_layout_rules'],new_jobs=p['exposed_jobs'],new_exact_jobs=p['exposed_exact_jobs'],new_initial_contracts=p['exposed_contracts'])
            if suite=='test':child.update(suite='new',layout_test=True,new_jobs=p['test_jobs'],new_exact_jobs=p['test_exact_jobs'],new_initial_contracts=p['test_contracts'])
            save(folder/'protocol.json',child)
            for mode,seconds in [('evaluate',main),('audit',audit)]:action([str(ROOT/'run_stage.sh'),str(HERE/'run.py'),mode,name],folder,mode,seconds,mode=='audit',child if mode=='evaluate' else None)
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
