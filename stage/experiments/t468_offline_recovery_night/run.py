"""Bounded offline Stage night registration, checks, and conditional gate."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import copy,hashlib,json,os,shutil,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t468_offline_recovery_night';DOC=ROOT/'stage/docs/overnight_20261110'
sys.path.insert(0,str(HERE))
import night_learning as N
P,E,M,R,H,torch,np=N.P,N.E,N.M,N.R,N.H,N.torch,N.np
read,save,sha,digest=N.read,N.save,N.sha,N.digest

def rootplan():return read(OUT/'protocol.json')
def storage_module():
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location("stage468_local_reversible_storage",HERE/"storage.py")
    module=module_from_spec(spec);spec.loader.exec_module(module)
    assert Path(module.__file__).resolve()==HERE/"storage.py"
    return module
def verify(full=False,ready=False):
    p=rootplan();assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert M.used(OUT)<=p['max_bytes'] and M.used()<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    if ready:assert M.used()-M.used(OUT)+p['max_bytes']<=2**30
    for f,h in (p['inputs'] if full else p['protected_files']).items():assert sha(f)==h,f

def prepare():
    assert not OUT.exists(),'Neverrestart'
    old=ROOT/'stage/runs/t467_actual_occupancy_audit';p0=read(old/'protocol.json');ss=read(old/'summary.json');diagnosis=read(old/'occupancy_diagnosis.json')
    assert ss['state']=='COMPLETE' and read(old/'audit.json')['state']=='PASS'
    assert read(ROOT/'stage/docs/review_20261110/T467_FINAL_REPORT.json')['remote']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert all(not Path('/proc',str(pid)).exists() for pid in read(old/'SEALED.json')['scientific_pids'])
    assert not Path('/proc',str(read(old/'driver_started.json')['pid'])).exists()
    base=read(ROOT/'stage/runs/t463_evening_2000/protocol.json');science=copy.deepcopy(base['science'])
    initial=read(ROOT/'stage/runs/t463_evening_2000/ppo/summary.json')['candidate']
    previous=read(ROOT/'stage/runs/t464_two_failure_recovery/summary.json');safe00=read(ROOT/'stage/runs/t465_matched_safe_recovery/summary.json')['query_rows'][0]
    carry21=next(v for v in previous['query_rows'] if v['job_id']=='dev_task6_21')
    science.update(recovery_teacher_model=p0['recovery_teacher_model'],onset_min_decision=20,onset_consecutive=3,onset_abs_steering_error=.02)
    stop=datetime(2026,11,11,8,30,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
    p=dict(id='T468_OFFLINE_CURRENT_OCCUPANCY_AND_FULL_DURATION_RELAY',out=str(OUT),science=science,initial_candidate=initial,
        modes=['occupancy','duration'],max_rounds_per_mode=2,max_bytes=80*2**20,root_metadata_bytes=6*2**20,
        round_partition_bytes=12*2**20,gate_partition_bytes=43*2**20,
        max_physics=40000000,max_commands=2000,max_main_seconds=14400,preflight_commands=24,
        absolute_learning_stop=stop-4500,absolute_native_stop=stop-2700,absolute_all_stop=stop,
        child_main_seconds=2400,child_learning_seconds=1200,child_audit_seconds=600,
        gate_main_seconds=1800,gate_audit_seconds=600,backup_seconds=900,
        seed_occupancy_rows=ss['source_rows'],seed_occupancy_diagnosis=diagnosis['labels'],
        validated_initial_recoveries=[safe00,carry21],seed_policy_hash=initial['model']['policy_hash'],
        user_authorization='Nov10 continue next mechanism and prepare unattended no-network night; user confirmsNov11 08:30allstop; review6mresultsbefore8m.',
        budget_scope='New bounded night window, not extending or restarting old endedT463/464/465/467. Separate40M/2000/14400mainseconds; disk1GiB/free40GiB unchanged.',
        transfer='Only auditPASS model with strict increase in successIDs and preserves ALL current successIDs is retained. Otherwise stop that mechanism; never pass rejected proposal.',
        admission='At most2rounds per2fixedmechanisms, each whole next12MiB plus complete43MiBgate andmetadata admitted from actualremaining80MiB. Not guarantee4rounds will be admitted.',
        onset_rule='occupancy first3consecutive steerabs>.02 forindices>=20; detectiondecision handover; ifnohit fixed20decisions fallback. No thresholds/time/teachergrid.',
        duration_rule='Full validated recovery physical-durationweighted actionloss, no arbitraryearly1/tail.1; otherretentiontargets unchanged.',
        same_policy_cache='Exact samepolicy andtask seed6 occupancy and initialvalid00/21 recovery explicitcache; no addedindependentdata orphysicalrepeatclaimed.',
        numerical_failure='Sharedsource/array/model/ActorAdam/density/rollback failure stops wholechain; recoveryfeasibilityfail stops only its fixedmechanism, may go to distinctothermechanism.',
        fresh_gate='Same329allsafe -> stopalltraining ->16fresh then78newworkers/newmodelpathfull exact ->audit/fullbinarybackup. Neveradapttofresh.',
        eight_started=False,eight_implemented=False,user_review_required_before8=True,
        protected_files={**p0['protected_files'],initial['model']['path']:initial['model']['sha256']})
    files=set(p0['inputs'])|{str(f) for f in HERE.glob('*.py')}|set(p['protected_files'])
    files|={v['path'] for v in ss['source_rows']+[safe00,carry21]}
    files|={str(old/'protocol.json'),str(old/'summary.json'),str(old/'audit.json'),str(old/'occupancy_diagnosis.json'),str(old/'cases.jsonl'),
        str(ROOT/'stage/docs/review_20261110/T467_FINAL_REPORT.json'),str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T467.json'),
        str(DOC/'PLAN.md'),str(DOC/'MECHANISMS.md')}
    S=storage_module()
    import t468_cache as C
    need=max(1,M.used()+p['max_bytes']+2*2**20-2**30)
    encoding=S.plan(files,need)
    p['storage_rows']=encoding['rows'];p['storage_expected_net_release']=encoding['selected_net_release_bytes']
    required_cache=max(1,need-encoding['selected_net_release_bytes'])
    cache=C.plan(files,required_cache);cachefile=HERE/'cache_plan.json'
    cachefile.write_bytes(C.encoded(cache));assert cachefile.stat().st_size<2*2**20
    p['remote_cache_path']=str(cachefile);p['remote_cache_sha256']=sha(cachefile)
    p['cache_total_reserved_bytes']=8*2**20
    p['cache_hard_stop_epoch']=time.time()+1200
    files.add(str(cachefile))
    for group in cache['groups']:
        for key in ['receipt','manifest']:files.add(str(ROOT/group[key]))
    for row in encoding['rows']:
        for key in ['receipt','manifest']:files.add(str(ROOT/row[key]))
    p['inputs']={f:sha(f) for f in sorted(files)};p['constructor_reads']=sorted(files)
    assert M.used()-cache['conservative_released_bytes']-encoding['selected_net_release_bytes']+p['max_bytes']+2*2**20<=2**30
    OUT.mkdir();DOC.mkdir(exist_ok=True,parents=True);save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'BUDGET.json',dict(state='REGISTERED_BEFORE_STORAGE_PREFLIGHT_LAUNCH',night40M2000main14400=True,max_physics=p['max_physics'],max_commands=p['max_commands'],
        max_main_seconds=p['max_main_seconds'],total_reserved_bytes=p['max_bytes'],root_metadata=p['root_metadata_bytes'],per_round=p['round_partition_bytes'],gate=p['gate_partition_bytes'],
        maximum_conditionally_admitted_rounds=4,full_four_rounds_NOT_guaranteed=True,cache_required=required_cache,cache_net=cache['conservative_released_bytes'],
        gzip_net=encoding['selected_net_release_bytes'],full_reserved_global_after_planned= M.used()-M.used(OUT)-cache['conservative_released_bytes']-encoding['selected_net_release_bytes']+p['max_bytes'],
        absolute_all_stop=stop,learning_stop=p['absolute_learning_stop'],native_stop=p['absolute_native_stop'],eight_started=False))
    print('T468_REGISTERED',flush=True)

def cache_evict():
    import t468_cache as C
    p=rootplan();assert sha(p['remote_cache_path'])==p['remote_cache_sha256'];p['remote_cache']=read(p['remote_cache_path']);C.evict(p)
def cache_audit():
    import t468_cache as C
    p=rootplan();assert sha(p['remote_cache_path'])==p['remote_cache_sha256'];p['remote_cache']=read(p['remote_cache_path']);C.audit(p)
def compress():
    S=storage_module()
    p=rootplan()
    if p['storage_rows']:S.execute(p)
    else:save(OUT/'compression_receipt.json',dict(state='NO_ENCODING_REQUIRED',files=0,native=0,optimizer=0))
def compression_audit():
    S=storage_module()
    p=rootplan()
    if p['storage_rows']:S.audit(p)
    else:save(OUT/'compression_audit.json',dict(state='PASS_NO_ENCODING_REQUIRED',files=0,native=0,optimizer=0))
    verify(True,True)

def child_config(root,mode,index,current):
    p=copy.deepcopy(root['science']);folder=OUT/f'{mode}_{index}'
    p.update(id=f'T468_{mode.upper()}_{index}',out=str(folder),mode=mode,round_index=index,learning_initial=current['model'],current_success_ids=current['success_ids'],
        diagnostic_jobs=[j for key in ['dev_task6_00','dev_task6_21','dev_task6_17','dev_task6_22','dev_task6_26','wide117_80_dev6_LEFT_13'] for j in p['main_jobs']+p['old275_jobs'] if j['id']==key],
        max_bytes=root['round_partition_bytes'],max_physics=8000000,max_commands=400,preflight_commands=0,
        main_seconds=2400,learning_seconds=1200,model_bytes=768*1024,full_bytes=448*1024,compact_bytes=4096,
        seed_policy_hash=root['seed_policy_hash'],seed_occupancy_rows=root['seed_occupancy_rows'],validated_initial_recoveries=root['validated_initial_recoveries'],
        absolute_learning_stop=root['absolute_learning_stop'],absolute_native_stop=root['absolute_native_stop'],absolute_all_stop=root['absolute_all_stop'],
        warmup_steps=128,protected_files=root['protected_files'])
    return p

def preflight():
    verify(True,True);root=rootplan();assert read(OUT/'cache_audit.json')['state'].startswith('PASS') and read(OUT/'compression_audit.json')['state'].startswith('PASS')
    assert not (OUT/'preflight.json').exists()
    results={}
    for mode in root['modes']:
        p=child_config(root,mode,0,root['initial_candidate']);a,b=N.make(p),N.make(p);initial=N.state(*a)
        data=N.data(p,root['seed_occupancy_rows'],root['validated_initial_recoveries'],mode)
        va=N.bc_step(a,data,mode);vb=N.bc_step(b,data,mode)
        assert va==vb and digest(N.state(*a))==digest(N.state(*b))
        assert va['reference218']<1e-24 and va['own10']<1e-24
        model=N.export(a,OUT/f'preflight_{mode}.pt.gz',p['model_bytes']);assert N.frozen(p,model['path']).phash()==a[0].phash()
        N.load(a,initial);N.load(b,initial);assert digest(N.state(*a))==digest(initial)==digest(N.state(*b))
        results[mode]=dict(step=va,model=model,reference_count=data['reference_count'],own_count=data['own_count'])
    import night_runtime as A
    p=child_config(root,'occupancy',0,root['initial_candidate']);folder=OUT/'preflight_native';folder.mkdir();p['out']=str(folder)
    # Exercise the actual CPU query branch without any native steps, including
    # independent teacher histories, dynamic prefix length and old full prefix.
    from importlib.util import spec_from_file_location,module_from_spec
    sp=spec_from_file_location('stage468_pref467',ROOT/'stage/experiments/t467_actual_occupancy_audit/run.py');U=module_from_spec(sp);sp.loader.exec_module(U)
    expected,raw=U.label_statistics(p,root['seed_occupancy_rows'])
    for key in ['dev_task6_00','dev_task6_21']:
        row=next(v for v in root['seed_occupancy_rows'] if v['job_id']==key);a=E.N.S.arrays(row['path'])
        job=next(v for v in p['diagnostic_jobs'] if v['id']==key);teacher,kind,hsh=A.teacher(p,job);hidden=None;labels=[]
        for x in a['states']:
            action,hidden=A.teacher_command(teacher,kind,x,hidden);labels.append(action[0].numpy().copy())
        assert np.array_equal(np.asarray(labels,np.float32),raw[key+'_teacher_actions'])
        old=next(v for v in root['validated_initial_recoveries'] if v['job_id']==key);A.prefix_exact(old,row,20)
    counts=R.M.mp.get_context('spawn').RawArray('q',9);pool=A.Pool(dict(p,native_stop_epoch=time.time()+180,max_physics=1),counts,lambda:None)
    try:
        replies=pool.request([dict(model=p['learning_initial']['path'],job=j,seed=None,cancel_probe=True) for j in p['old8_jobs']])
        assert all(v['state']=='CANCEL_PASS' for v in replies) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',native=0,commands=24,discarded_BC_Adam=4,live_shadow_fullActorAdam_rollback_exact=True,
        variable_handover_original_history_masks=True,duration_nativeweighted_fullsequence=True,eightworkers_cancelled=True,fixture_NOT_actualnewrecovery=True,
        fresh16_unconsumed=True,reload78_not_executed=True))
    print('T468_PREFLIGHT_PASS',flush=True)

def roundmain(mode,index):
    import night_runtime as A
    A.mainround(mode,int(index))
def roundaudit(mode,index):
    import night_runtime as A
    A.auditround(mode,int(index))
def install_gate():
    X=N.X;X.OUT,X.HERE,X.DOC=OUT,HERE,DOC
    return X
def gatemain():install_gate().gatemain()
def gateaudit():install_gate().gateaudit()

def launch():
    verify(True,True);p=rootplan();assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    now=time.time();assert now+2400+600+1800+600+900<p['absolute_all_stop']
    with (OUT/'driver_console.log').open('xb') as f:
        c=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T468','--why=Offline bounded Stage overnight relay until08:30',
            'timeout','--signal=TERM','--kill-after=2',str(int(p['absolute_all_stop']-now)),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=c.pid,epoch=now,learning_stop=p['absolute_learning_stop'],native_stop=p['absolute_native_stop'],all_stop=p['absolute_all_stop'],
        source_remote=src['remote_commit'],protocol_sha256=sha(OUT/'protocol.json'),no_network_required_for_scientific_chain=True))
    print('T468_ACTUALLY_LAUNCHED',c.pid,flush=True)

if __name__=='__main__':
    try:globals()[sys.argv[1]](*sys.argv[2:])
    except BaseException:
        if OUT.exists():save(OUT/f'error_{sys.argv[1]}_{os.getpid()}.json',dict(error=traceback.format_exc(),epoch=time.time()))
        raise
