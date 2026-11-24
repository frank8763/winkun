"""One bounded fixed-capacity successful-behavior augmentation experiment."""
from pathlib import Path
import fcntl, importlib.util, json, os, shutil, signal, subprocess, sys, time, traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t719_outcome_weighted_replay';DOC=ROOT/'stage/docs/day_20261124/t719';UNIT='stage-t719.service'
if __name__=='__main__' and sys.argv[1]=='helper':
    spec=importlib.util.spec_from_file_location('t719_resource_helper',ROOT/'stage/experiments/t699_unexecuted_continuation/run.py')
    H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
    test=len(sys.argv)>2 and sys.argv[2]=='preflight'
    H.OUT=OUT/'helper_preflight' if test else OUT;H.DOC=DOC/'helper_preflight' if test else DOC
    H.UNIT='stage-t719-helper-test.service' if test else UNIT
    H.helper();sys.exit(0)
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'stage/experiments/t701_compact_retention'))
spec=importlib.util.spec_from_file_location('t719_native_controller',ROOT/'stage/experiments/t701_compact_retention/run.py')
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
    spec=importlib.util.spec_from_file_location('t719_previous_guard',ROOT/'stage/experiments/t718_stochastic_policy_comparison/run.py')
    G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G);G.stopped();unit_stopped('stage-t718-finalizer.service')
    assert read(ROOT/'stage/docs/day_20261124/t718/FINALIZER_DONE.json')['state']=='STOPPED'

def stopped():
    old_stopped()
    for unit in [UNIT,'stage-t719-helper-test.service','stage-t719-preflight.service','stage-t719-preflight-recovery.service','stage-t719-preflight-recovery2.service']:unit_stopped(unit)
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
    import acquisition_native as RN
    RN.install();M.native_child(action,name)

def call_counts():
    values={'Actor_Adam_'+w:0 for w in ['live','shadow','independent']}
    for f in [DOC/'PREFLIGHT_CALLS.jsonl',OUT/'fit/fit_calls.jsonl']:
        if f.exists():
            for line in f.read_text().splitlines():
                v=json.loads(line)
                if v['stage']=='BEFORE':values['Actor_Adam_'+v['who']]+=1
    return values

def account(items,state):
    totals=dict(physics=sum(i.get('physics',0) for i in items),commands=sum(i.get('commands',0) for i in items),science_seconds=sum(i['elapsed_seconds'] for i in items if not i['audit']),audit_seconds=sum(i['elapsed_seconds'] for i in items if i['audit']),**call_counts())
    p=root_plan();assert totals['physics']<=p['max_physics'] and totals['commands']<=p['max_commands'];assert all(v<=17 for k,v in totals.items() if '_Adam_' in k)
    save(OUT/'LEDGER.json',dict(state=state,children=items,totals=totals,old_4096_counters_unchanged=True))
    ledger=read(C.WEEKEND/'LEDGER.json');job=next(j for j in ledger['new_tuesday_experiments'] if j['id']=='T719');before=job.get('accounted',{})
    for k in ['physics','commands','science_seconds','audit_seconds']:
        assert totals[k]>=before.get(k,0);ledger[k]+=totals[k]-before.get(k,0)
    job.update(state=state,accounted=totals)
    for key,cap in [('physics',1500000000),('commands',45000),('science_seconds',180000),('audit_seconds',18000)]:assert ledger[key]<=cap
    save(C.WEEKEND/'LEDGER.json',ledger)

def register():
    stopped();auth=read(DOC/'AUTHORIZATION.json');assert read(DOC/'PREFLIGHT.json')['state']=='PASS' and read(DOC/'NATIVE_PREFLIGHT.json')['state']=='PASS' and not (OUT/'protocol.json').exists()
    p=read(OUT/'prepared.json');now=time.time();p.update(id='T719_ALL_OUTCOME_WEIGHTED_RECOVERY_REPLAY',out=str(OUT),registration_epoch=now,absolute_learning_stop=auth['absolute_learning_stop'],absolute_native_stop=auth['absolute_native_stop'],absolute_all_stop=auth['absolute_all_stop'],max_bytes=512*2**20,child_bytes=384*2**20,global_bytes=10*2**30,retained='T698',new_RL=False,new_offpolicy_weighted_fit=True,max_Actor_Adam_each=17,new_Value=0)
    p['max_physics']=2*sum(j['case']['step_cap'] for j in p['recovery_jobs']+p['original_jobs']);p['max_commands']=86
    assert now+300+600+300<auth['absolute_learning_stop'] and now+300+600+300+1200<auth['absolute_native_stop'] and now+300+600+300+1200+180+120+480<auth['absolute_all_stop'];assert used(ROOT/'stage/runs')+p['max_bytes']<8*2**30
    ledger=read(C.WEEKEND/'LEDGER.json');assert not any(j['id']=='T719' for j in ledger.get('new_tuesday_experiments',[]))
    for k,x,cap in [('physics',p['max_physics'],1500000000),('commands',86,45000),('science_seconds',1800,180000),('audit_seconds',480,18000)]:assert ledger[k]+x<=cap
    old=read(ROOT/'stage/runs/t718_stochastic_policy_comparison/protocol.json');inputs=dict(old['inputs']);extra=set(HERE.glob('*.py'))|{DOC/f for f in ['PLAN.md','AUTHORIZATION.json','FIXED_DATA.json','PRIMARY_REVIEW.md','PREFLIGHT.json','PREFLIGHT_CALLS.jsonl','NATIVE_PREFLIGHT.json']}|{ROOT/'stage/runs/t718_stochastic_policy_comparison/SEALED.json',ROOT/'stage/docs/day_20261124/t718/FINALIZER_DONE.json',ROOT/'stage/runs/t717_onpolicy_recovery/training/audit.json'}
    for row in p['replay_rows']:extra.add(Path(row['path']))
    inputs.update({str(f):sha(f) for f in extra});entries=[[f,h,Path(f).stat().st_size,Path(f).stat().st_mtime_ns] for f,h in sorted(inputs.items())];(HERE/'inputs.json').write_text(json.dumps(entries,separators=(',',':'))+'\n')
    p['_input_manifest']=dict(path=str(HERE/'inputs.json'),sha256=sha(HERE/'inputs.json'));p.pop('inputs',None);p.pop('input_stats',None);blob=json.dumps(p,separators=(',',':'))+'\n';assert len(blob.encode())<2**21;(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob)
    ledger.setdefault('new_tuesday_experiments',[]).append(dict(id='T719',state='RESERVED',out=str(OUT),max_physics=p['max_physics'],max_commands=86,independent_Actor_Adam_each=17,new_Value=0,old4096_counters_unchanged=True));save(C.WEEKEND/'LEDGER.json',ledger);save(DOC/'TRANSPORT_LEDGER.json',dict(cap=900,charged_seconds=0,source_commits=[],attempts=[]));save(DOC/'REGISTERED.json',dict(state='REGISTERED_NOT_LAUNCHED',inputs=len(inputs),max_physics=p['max_physics'],max_commands=86,absolute_learning_stop=p['absolute_learning_stop'],absolute_native_stop=p['absolute_native_stop'],absolute_all_stop=p['absolute_all_stop']))
    verify(read(OUT/'protocol.json'),force=True);print('T719_REGISTERED',len(inputs),flush=True)

def sync(final=False):
    stopped();p=root_plan();assert time.time()+300<p['absolute_all_stop'];l=read(DOC/'TRANSPORT_LEDGER.json');assert l['charged_seconds']+300+(0 if final else 300)<=900
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T719 source300')));signal.alarm(295)
    sys.path.insert(0,str(ROOT/'stage/experiments/weekend_20261120_23'));import archive_after_t695 as G
    G.stopped=G.Client.A.stopped=stopped;G.Client.client();sys.path.insert(0,str(ROOT/'stage/experiments/t626_multitask_closed_loop'));import t626_sync_source;t626_sync_source.main();stopped();signal.alarm(0)

def launch():
    stopped();p=root_plan();verify(p,force=True);assert not (OUT/'launch.json').exists();receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip();assert receipt['root_tree']==tree
    now=time.time();assert now+600+300<p['absolute_learning_stop'] and now+600+300+1200<p['absolute_native_stop'] and now+600+300+1200+180+120+480<p['absolute_all_stop']
    command=['systemd-run','--user','--unit=stage-t719','--collect','--property=Type=exec','--property=CPUQuota=800%','--property=CPUWeight=10','--property=AllowedCPUs=8 9 10 11 12 13 14 15','--property=Nice=19','--property=MemoryMax=8G','--property=MemorySwapMax=0','--property=KillMode=control-group','--property=TimeoutStopSec=5',f'--property=RuntimeMaxSec={int(p["absolute_all_stop"]-now-3)}',f'--property=WorkingDirectory={ROOT}',f'--property=StandardOutput=append:{OUT}/driver.log','--property=StandardError=inherit','--setenv=CUDA_VISIBLE_DEVICES=','--setenv=OMP_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1','systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage T719','--why=Bounded onpolicy recovery',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'driver']
    c=subprocess.run(command,capture_output=True,text=True);assert c.returncode==0,c.stderr
    save(OUT/'launch.json',dict(epoch=now,unit=UNIT,remote_commit=receipt['remote_commit'],root_tree=tree));print(c.stderr,flush=True)

def driver():
    p=root_plan();verify(p,force=True);assert not (OUT/'DRIVER_STARTED.json').exists();os.sched_setaffinity(0,{15});save(OUT/'DRIVER_STARTED.json',dict(pid=os.getpid(),epoch=time.time()))
    log=(OUT/'helper.log').open('xb');helper=subprocess.Popen(['/usr/bin/python3',str(HERE/'run.py'),'helper'],stdout=log,stderr=log);ids=[helper.pid];items=[];error=None
    def action(mode,name,seconds,audit,physical=None):
        assert helper.poll() is None;folder=OUT/name;verify(p,force=True);assert time.time()+seconds+5<p['absolute_all_stop']
        with (folder/(mode+'.log')).open('xb') as f:
            child=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'run.py'),mode,name],cwd=ROOT,stdout=f,stderr=f,start_new_session=True);ids.append(child.pid);start=time.time();save(OUT/'status.json',dict(state='ACTIVE',child=name,action=mode,child_pid=child.pid,driver_pid=os.getpid(),epoch=start,absolute_action_stop=start+seconds))
            try:child.wait(timeout=seconds);assert child.returncode==0,(mode,child.returncode)
            except BaseException:
                R.scoped_signal(child.pid,signal.SIGCONT);R.scoped_signal(child.pid,signal.SIGTERM)
                try:child.wait(timeout=3)
                except subprocess.TimeoutExpired:R.scoped_signal(child.pid,signal.SIGKILL);child.wait()
                raise
            finally:
                item=dict(name=name,mode=mode,pid=child.pid,audit=audit,returncode=child.returncode,elapsed_seconds=time.time()-start)
                if physical:
                    sm=read(folder/'summary.json') if (folder/'summary.json').exists() else {};item.update(physics=sm.get('physics',physical['max_physics']),commands=sm.get('commands',physical['max_commands']),conservative_upper_if_summary_missing=not bool(sm))
                items.append(item);account(items,'ACTIVE')
    try:
        until=time.time()+5
        while not (DOC/'AFFINITY_GUARD.json').exists() and time.time()<until:time.sleep(.05)
        assert read(DOC/'AFFINITY_GUARD.json')['pid']==helper.pid
        fitdir=OUT/'fit';assert time.time()+600+300<p['absolute_learning_stop']
        action('fit','fit',600,False,dict(max_physics=0,max_commands=0));action('fit_audit','fit',300,True);assert read(fitdir/'audit.json')['state']=='PASS'
        folder=OUT/'validation';folder.mkdir();(folder/'episodes').mkdir();now=time.time();assert now+1200+180+60<p['absolute_all_stop'] and now+1200<p['absolute_native_stop']
        child=dict(out=str(folder),arm='validation',_root_sha256=sha(OUT/'protocol.json'),main_seconds=1200,audit_seconds=180,max_physics=p['max_physics'],max_commands=86,absolute_native_stop=min(now+1200,p['absolute_native_stop']),absolute_all_stop=min(now+1440,p['absolute_all_stop']));save(folder/'protocol.json',child)
        action('evaluate','validation',1200,False,child);action('evaluate_audit','validation',180,True);assert read(folder/'audit.json')['state']=='PASS'
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        (OUT/'HELPER_STOP').touch()
        try:helper.wait(timeout=5)
        except subprocess.TimeoutExpired:helper.terminate();helper.wait(timeout=3)
        log.close()
        for f in OUT.glob('*/worker_generations.json'):
            for g in read(f):ids.extend([g['parent'],*g['workers']])
        alive=[i for i in set(ids) if Path('/proc',str(i)).exists()];state='FAILED' if error or alive else 'COMPLETE';account(items,state);report=dict(state=state,error=error,scientific_pids=sorted(set(ids)),all_scientific_processes_stopped=not alive,helper_stopped=helper.poll() is not None,retained='T698',new_RL=False,new_offpolicy_Actor_fit=True,epoch=time.time())
        for f in [OUT/'SEALED.json',OUT/'FINAL_REPORT.json',OUT/'status.json',DOC/'FINAL_REPORT.json']:save(f,report)

if __name__=='__main__':
    mode=sys.argv[1]
    if mode in ['fit','fit_audit']:
        import t719_fit as F;F.fit() if mode=='fit' else F.audit()
    elif mode in ['evaluate','evaluate_audit']:
        import t719_runtime as T;getattr(T,mode)(sys.argv[2])
    else:{'register':register,'sync':sync,'launch':launch,'driver':driver}[mode]()
