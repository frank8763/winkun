"""Paired T443 online task-allocation control; no runtime task routing."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,os,shutil,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stage/experiments/t441_joint_actor_critic'))
import t441_learning as L
spec=importlib.util.spec_from_file_location('stage443_pinned441_native',ROOT/'stage/experiments/t441_joint_actor_critic/run.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
import reserve as Z
E,torch,np=L.E,L.torch,L.np
read,save,sha,digest=L.read,L.save,L.sha,L.digest
OUT=ROOT/'stage/runs/t443_sampling_control';DOC=ROOT/'stage/docs/research_rescue_20261109'


def verify(p,full=False):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    assert Z.used(OUT)<=p['max_bytes'] and Z.used()<=2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30
    for f,h in p['protected_files'].items():assert sha(f)==h,f
    if full:
        for f,h in p['inputs'].items():assert sha(f)==h,f


def objects(p,device='cuda'):
    obj=L.make(p,device);L.load(obj,L.read_state(p['joint_initial_model']))
    assert digest(L.state(*obj))==p['joint_initial_hash']
    assert obj[0].phash()==p['expected_policy_hash']
    return obj


def prepare():
    assert not OUT.exists(),'Never restart an existing run'
    old=ROOT/'stage/runs/t441_joint_actor_critic';base=read(old/'protocol.json');s=read(old/'summary.json')
    assert read(old/'audit.json')['state']=='PASS'
    for group in ['T441','T442']:
        receipt=read(ROOT/f'stage/docs/long_distance_20261101/release_receipt_{group}.json')
        assert receipt['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    for folder in [old,ROOT/'stage/runs/t442_value_warmup']:
        for name in ['launch.json','driver_started.json']:
            assert not Path('/proc',str(read(folder/name)['pid'])).exists()
    p={k:copy.deepcopy(base[k]) for k in ['source_protocol','initial_model','student_start_model','student_start_sha256',
        'raw_sigma','phase','old8_jobs','old275_jobs','main_jobs','actor_lr','value_lr','value_seed','epochs','ppo_clip',
        'target_kl','gamma_native','lambda_native','centre_jobs','relay_stage']}
    model=s['development']['model'];p.update(joint_initial_model=model['path'],joint_initial_hash=model['joint_hash'],expected_policy_hash=model['policy_hash'])
    p.update(id='T443_TASK_ALLOCATION_CONTROL',out=str(OUT),seed=202611094430,
        max_bytes=36*2**20,full_bytes=352*1024,compact_bytes=4*1024,
        max_physics=21000000,max_commands=900,arm_physics=10500000,arm_commands=450,
        main_seconds=5400,learning_seconds=3600,audit_seconds=1200,backup_seconds=600,all_seconds=8100,
        arms=['uniform','left_coverage'],rounds_per_arm=2,preflight_zero_native_commands=16,
        inherited_Actor_V_and_both_Adam=True,T442_value_heads_NOT_used=True,
        full329_each_fixed_final=True,fresh16=None,reload78=None,six_gate=False,eight_started=False,
        maximum_live_Actor_Adam=16,maximum_live_V_Adam=16,maximum_shadow_Actor_Adam=16,maximum_shadow_V_Adam=16,
        maximum_audit_Actor_plus_V_Adam=32,preflight_discarded_Actor_Adam=4,preflight_discarded_V_Adam=4,
        maximum_new_training_episodes=64,maximum_exact_training_replays=64,
        no_teacher_no_auxiliary_no_PCGrad=True,no_old_success_per_round_rollback=True,
        comparison='Equal32 episode/8Actor+8V maximum updates per arm, equal physical cap. Actual decisions/physical durations can differ because tasks differ.',
        user_authorization='Nov9 next step after reviewed task-allocation comparison; independent finite protocol, not continuation of T441/T442.')
    byid={j['id']:j for j in p['old275_jobs']+p['main_jobs']}
    hard=['dev_task6_00','dev_task6_20','dev_task6_21','dev_task6_23',
          'dev_task6_24','dev_task6_26','hold_task6_00','dev_task6_13']
    schedule={k:[] for k in p['arms']}
    for rnd in range(2):
        uniform=[]
        for task in range(8):uniform.append(byid[f'dev_task{task}_{[0,20][rnd]:02d}'])
        focused=[byid[x] for x in hard[rnd*4:rnd*4+4]]
        focused.extend(byid[f'dev_task{t}_{[0,20][rnd]:02d}'] for t in ([0,1,2,3] if rnd==0 else [4,5,7,0]))
        wide=[byid[f'wide117_80_dev6_{side}_{index:02d}'] for side in ['LEFT','RIGHT'] for index in ([3,10,17,24] if rnd==0 else [0,7,14,21])]
        schedule['uniform'].append(uniform+wide);schedule['left_coverage'].append(focused+wide)
    p['training_jobs_by_arm']=schedule
    p['baseline_success_ids']=sorted(r['job_id'] for r in map(json.loads,(old/'cases.jsonl').read_text().splitlines()) if r['role']=='FINAL329' and r['success'] and not r['collision'])
    assert len(p['baseline_success_ids'])==321
    worst={}
    for arm,batches in schedule.items():
        cap=2*sum(j['case']['step_cap'] for batch in batches for j in batch)
        cap+=2*sum(j['case']['step_cap'] for j in p['centre_jobs'])+sum(j['case']['step_cap'] for j in p['main_jobs']+p['old275_jobs'])
        worst[arm]=cap;assert cap<=p['arm_physics']
    assert sum(worst.values())<=p['max_physics'];p['worst_physics_by_arm']=worst
    # Discover actual constructor reads before selecting storage candidates.
    # No Actor forward or optimizer step is performed in this preparation.
    seen=set();active=[True]
    def hook(event,args):
        if active[0] and event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            try:path=Path(os.fsdecode(args[0])).resolve()
            except (ValueError,OSError):return
            if path.is_relative_to(ROOT):seen.add(str(path))
    sys.addaudithook(hook)
    obj=objects(p,'cpu');assert obj[0].phash()==model['policy_hash'];del obj;active[0]=False
    protected=set(base['inputs'])|seen|{model['path'],str(old/'summary.json'),str(old/'audit.json')}
    # Existing native/science constructors consume some historical JSON; all
    # observed reads and declared input hashes are excluded from cold encoding.
    p['constructor_reads']=sorted(seen)
    required=max(0,Z.used()+p['max_bytes']-2**30+1024**2)
    cold,saving=Z.candidates(protected,required);p['cold_storage']=cold
    protected_models={f:h for f,h in base['inputs'].items() if 't255_overnight' in f and f.endswith('.gz')}
    protected_models.update({str(old/f'round_{i}.pt.gz'):sha(old/f'round_{i}.pt.gz') for i in range(4)})
    protected_models[model['path']]=model['sha256'];p['protected_files']=protected_models
    files=set(base['inputs'])|seen|{str(f) for f in HERE.glob('*.py')}|{
        str(DOC/'T443_PLAN.md'),str(old/'summary.json'),str(old/'audit.json'),model['path'],
        str(ROOT/'stage/experiments/t441_joint_actor_critic/run.py'),str(ROOT/'stage/experiments/t441_joint_actor_critic/t441_learning.py')}
    for r in cold:files|={str(ROOT/r['receipt']),str(ROOT/r['manifest'])}
    assert not {str(ROOT/r['original']) for r in cold}&files
    files={f for f in files if Path(f).is_file()}
    p['inputs']={f:sha(f) for f in sorted(files)}
    before=Z.used();assert before-saving+p['max_bytes']+512*1024<=2**30
    p['storage_before']=before;p['storage_saving_planned']=saving
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    storage=dict(full65=65*p['full_bytes'],four_models=4*1024**2,compact698=698*p['compact_bytes'],metadata=6.5*1024**2)
    assert sum(storage.values())<=p['max_bytes']
    save(DOC/'T443_BUDGET.json',dict(state='REGISTERED_ENCODING_AND_PREFLIGHT_PENDING',max_bytes=p['max_bytes'],
        global_before=before,planned_cold_files=len(cold),planned_saving=saving,global_full_after_cold=before-saving+p['max_bytes'],
        storage=storage,worst_physics=worst,total_worst_physics=sum(worst.values()),max_physics=p['max_physics'],
        maximum_all_commands=858,max_commands=900,training_episodes=64,exact_replays=64,
        actual_constructor_reads_protected=len(seen),native_during_prepare=0,Actor_forwards_during_prepare=0,Adam_steps_during_prepare=0))
    verify(p,True);print('T443_REGISTERED',sum(worst.values()),len(cold),saving,flush=True)


def encode():
    p=read(OUT/'protocol.json');verify(p,True);Z.encode(p);verify(p,True);print('T443_STORAGE_ENCODED',flush=True)
def storage_audit():
    p=read(OUT/'protocol.json');Z.audit(p);verify(p,True);print('T443_STORAGE_AUDIT_PASS',flush=True)


def preflight():
    p=read(OUT/'protocol.json');verify(p,True);assert read(OUT/'storage_audit.json')['state']=='PASS'
    assert not (OUT/'preflight.json').exists();assert torch.cuda.is_available()
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    cpu=objects(p,'cpu');data=M.synthetic(cpu[0],p);del cpu
    pp=dict(p,epochs=2);a=objects(p);b=objects(p)
    before=L.state(*a);assert digest(before)==digest(L.state(*b))
    live=L.fit(a,data,pp,lambda:None,lambda *x:None);repeat=L.fit(b,data,pp,lambda:None,lambda *x:None)
    assert live==repeat and digest(L.state(*a))==digest(L.state(*b))
    after=L.state(*a);L.load(a,before);assert digest(L.state(*a))==digest(before)
    L.load(a,after);assert digest(L.state(*a))==digest(after)
    assert live['Actor_Adam']==repeat['Actor_Adam']==2
    del a,b;torch.cuda.empty_cache()
    counts=M.mp.get_context('spawn').RawArray('q',9);folder=OUT/'preflight_native';folder.mkdir()
    pool=M.Pool(dict(p,out=str(folder),native_stop_epoch=time.time()+180,max_physics=1),counts,lambda:None)
    try:
        req=[dict(model=p['joint_initial_model'],job=j,mode='student',seed=None,cancel_probe=True) for j in p['training_jobs_by_arm']['left_coverage'][0][:8]]
        assert all(r['state']=='CANCEL_PASS' for r in pool.request(req)) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',native=0,discarded_Actor_Adam=4,discarded_V_Adam=4,
        zero_physics_commands=16,eight_workers_cancelled_and_stopped=True,
        inherited_mature_Actor_V_both_Adam_exact=True,independent_gpu_shadow_and_restore_exact=True,
        original_Actor_and_Value_source_checkpoint_not_modified=True,fit=live))
    print('T443_PREFLIGHT_PASS',flush=True)


def arm_run(p,arm,root_epoch):
    folder=OUT/arm;assert not folder.exists();folder.mkdir()
    # Reuse the locked native runtime, with its globals set only in this sole
    # parent process; children use explicit per-arm paths in their configuration.
    M.OUT=folder;M.verify=lambda *_args,**_kw:verify(p)
    save(folder/'launch.json',dict(epoch=root_epoch,root_launch_path=str(OUT/'launch.json'),not_a_separate_launch=True))
    pp=dict(p,out=str(folder),max_physics=p['arm_physics'],max_commands=p['arm_commands'])
    rt=M.Runtime(pp);obj=None;rounds=[];error=None;status='FAILED';development=None
    source=dict(path=p['joint_initial_model'],sha256=sha(p['joint_initial_model']),policy_hash=p['expected_policy_hash'],joint_hash=p['joint_initial_hash'])
    model=source
    try:
        obj=objects(p);rt.start_pool()
        for rnd,jobs in enumerate(p['training_jobs_by_arm'][arm]):
            rt.guard(True);before=L.state(*obj);seeds=[p['seed']+rnd*16+i for i in range(16)]
            items=rt.request(jobs,model,f'TRAIN_{rnd}',seeds,True)
            rt.request(jobs,model,f'EXACT_{rnd}',seeds,references=[r for a,r in items])
            arrays=[]
            for i,(a,r) in enumerate(items):a['_group']=i//8;arrays.append(a)
            shadow=objects(p);L.load(shadow,before)
            def event(which):
                def emit(where,k):
                    with (folder/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([rnd,which,where,k])+'\n')
                return emit
            try:
                live=L.fit(obj,arrays,p,lambda:rt.guard(True),event('live'))
                other=L.fit(shadow,arrays,p,lambda:rt.guard(True),event('shadow'))
                assert live==other and digest(L.state(*obj))==digest(L.state(*shadow))
            except BaseException:
                L.load(obj,before);save(folder/f'interrupted_round_{rnd}.json',dict(whole_Actor_V_Adam_rollback_exact=digest(L.state(*obj))==digest(before)));raise
            del shadow;torch.cuda.empty_cache()
            model=L.export(obj,folder/f'round_{rnd}.pt.gz')
            centres=rt.request(p['centre_jobs'],model,f'CENTRES_{rnd}')
            record=dict(round=rnd,fit=live,model=model,training_paths=[r['path'] for a,r in items],
                training_old=M.tally([r for a,r in items[:8]]),training_wide=M.tally([r for a,r in items[8:]]),
                old8=M.tally([r for a,r in centres[:8]]),wide2=M.tally([r for a,r in centres[8:]]),research_updates_retained=True)
            rounds.append(record);save(folder/'rounds.json',rounds)
        final=rt.request(p['main_jobs']+p['old275_jobs'],model,'FINAL329')
        ids=sorted(r['job_id'] for a,r in final if r['success'] and not r['collision'])
        previous=set(p['baseline_success_ids']);actual=set(ids)
        development=dict(wide54=M.tally([r for a,r in final[:54]]),old275=M.tally([r for a,r in final[54:]]),
            success_ids=ids,all329_safe=len(ids)==329,repaired=sorted(actual-previous),lost=sorted(previous-actual),model=model)
        save(folder/'development.json',development);status='COMPLETE'
    except BaseException:
        error=traceback.format_exc();save(folder/'failure.json',dict(error=error,physics=sum(rt.counts),commands=rt.commands,role=rt.role))
    finally:
        if rt.pool:rt.pool.close()
        summary=dict(state=status,pid=os.getpid(),arm=arm,elapsed_s=time.time()-rt.start,physics=sum(rt.counts),commands=rt.commands,
            rounds=rounds,development=development,error=error,all_native_workers_stopped=True,fresh16=None,reload78=None,six_gate=False,eight_started=False)
        save(folder/'summary.json',summary);save(folder/'status.json',dict(state=status,epoch=time.time()))
        M.fcntl.flock(rt.lock,M.fcntl.LOCK_UN);rt.lock.close()
    assert status=='COMPLETE',error
    return summary


def main():
    p=read(OUT/'protocol.json');verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS'
    E.N.offline();E.L.install_stop_handlers();launch=read(OUT/'launch.json');start=time.time();results={};status='FAILED';error=None
    save(OUT/'main_started.json',dict(pid=os.getpid(),epoch=start,native_stop=launch['epoch']+p['main_seconds']))
    try:
        for arm in p['arms']:
            save(OUT/'status.json',dict(state=arm,epoch=time.time(),pid=os.getpid()))
            results[arm]=arm_run(p,arm,launch['epoch'])
        assert sum(x['physics'] for x in results.values())<=p['max_physics']
        assert sum(x['commands'] for x in results.values())+16<=p['max_commands'];status='COMPLETE'
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True))
    finally:
        summary=dict(state=status,pid=os.getpid(),elapsed_s=time.time()-start,results=results,error=error,
            physics=sum(r['physics'] for r in results.values()),commands=sum(r['commands'] for r in results.values())+16,
            fresh16=None,reload78=None,six_gate=False,eight_started=False,all_native_workers_stopped=True)
        save(OUT/'summary.json',summary);save(OUT/'status.json',dict(state=status,epoch=time.time()))
    assert status=='COMPLETE',error


def audit():
    p=read(OUT/'protocol.json');verify(p,True);s=read(OUT/'summary.json');assert s['state']=='COMPLETE'
    assert not Path('/proc',str(s['pid'])).exists() and not (OUT/'audit.json').exists()
    records={};calls=0;start=time.time()
    for arm in p['arms']:
        folder=OUT/arm;ss=read(folder/'summary.json');assert ss['state']=='COMPLETE'
        assert all(not Path('/proc',str(pid)).exists() for pid in read(folder/'workers.json')['workers'])
        rows=[json.loads(v) for v in (folder/'cases.jsonl').read_text().splitlines()]
        reps=[json.loads(v) for v in (folder/'replays.jsonl').read_text().splitlines()]
        events=[json.loads(v) for v in (folder/'optimizer_calls.jsonl').read_text().splitlines()]
        obj=objects(p);training=[]
        for row in rows:
            assert sha(row['path'])==row['sha256'];a=E.N.S.arrays(row['path'])
            assert int(a['native_counts'].sum())==row['physics']
            if row['full']:
                assert digest(a)==row['array_digest']
                for key,h in row['array_hashes'].items():assert digest(a[key])==h
        for rr in ss['rounds']:
            assert time.time()-start<p['audit_seconds']
            mirror=E.safe_initial(E.N.science(p),p).eval().requires_grad_(False);mirror.load_learned(L.state(*obj)['actor'])
            arrays=[]
            for i,path in enumerate(rr['training_paths']):
                row=next(x for x in rows if x['path']==path);a=E.N.S.arrays(path)
                E.NATIVE_AUDIT.audit_array(mirror,a,p,seed=row['seed'])
                a['_group']=i//8;arrays.append(a);training.append(row)
            fit=L.fit(obj,arrays,p,lambda:None,lambda *a:None)
            assert fit==rr['fit'] and digest(L.state(*obj))==rr['model']['joint_hash']
            assert sha(rr['model']['path'])==rr['model']['sha256'];calls+=fit['Actor_Adam']+fit['V_Adam']
            for which in ['live','shadow']:
                assert [e[2:] for e in events if e[:2]==[rr['round'],which]]==[[w,k] for k in range(fit['Actor_Adam']) for w in ['BEFORE','AFTER']]
        assert ss['physics']==sum(r['physics'] for r in rows+reps)
        assert ss['commands']==8+len(rows)+len(reps)
        for r in reps:
            previous=next(x for x in rows if x['path']==r['reference_path'])
            assert r['array_digest']==previous['array_digest'] and r['integration_sha256']==previous['integration_sha256']
        final=[r for r in rows if r['role']=='FINAL329'];assert len(final)==329
        ids={r['job_id'] for r in final if r['success'] and not r['collision']}
        assert sorted(ids)==ss['development']['success_ids']
        records[arm]=training
    same=[]
    for a in records['uniform']:
        for b in records['left_coverage']:
            if a['array_digest']==b['array_digest']:same.append(dict(uniform=a['path'],left_coverage=b['path'],job_id=a['job_id']))
    verify(p,True)
    save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,new_RL=0,discarded_historical_Actor_plus_V_Adam=calls,
        all_Actor_V_Adam_raw_history_density_rewards_exact=True,all_native_workers_stopped=True,
        physics=s['physics'],commands=s['commands'],cross_arm_identical_training=same,
        cross_arm_identical_training_count=len(same),physical_repeats_billed_not_independent=True,elapsed_s=time.time()-start))
    print('T443_AUDIT_PASS',flush=True)


def launch():
    p=read(OUT/'protocol.json');verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS'
    assert read(OUT/'storage_audit.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    assert Z.used()-Z.used(OUT)+p['max_bytes']<=2**30
    now=time.time()
    with (OUT/'driver_console.log').open('xb') as f:
        proc=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T443','--why=Bounded task allocation control',
            'timeout','--signal=TERM','--kill-after=2',str(p['all_seconds']),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=proc.pid,epoch=now,learning_stop=now+p['learning_seconds'],native_stop=now+p['main_seconds'],
        all_stop=now+p['all_seconds'],source_remote=receipt['remote_commit'],protocol_sha256=sha(OUT/'protocol.json'),new_finite_window=True))
    print('T443_ACTUALLY_LAUNCHED',proc.pid,flush=True)


if __name__=='__main__':globals()[sys.argv[1]]()
