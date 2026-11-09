"""Register, preflight and run a finite value-only target-control experiment."""
from pathlib import Path
import ast,collections,fcntl,hashlib,json,math,os,shutil,signal,subprocess,time,traceback,sys
import t442_core as C
ROOT,HERE,OUT,DOC=C.ROOT,C.HERE,C.OUT,C.DOC
read,save,sha,digest,np,torch=C.read,C.save,C.sha,C.digest,C.np,C.torch


def used(path):return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
def verify(p,full=False):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    assert used(OUT)<=p['max_bytes'] and used(ROOT/'stage/runs')<=2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30
    for f,h in p['protected_files'].items():assert sha(f)==h,f
    if full:
        for f,h in p['inputs'].items():assert sha(f)==h,f


def prepare():
    assert not OUT.exists(),'Never restart an existing experiment directory'
    source=ROOT/'stage/runs/t441_joint_actor_critic';sp=read(source/'protocol.json')
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T441.json'
    assert read(source/'audit.json')['state']=='PASS'
    assert read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert all(not Path('/proc',str(pid)).exists() for pid in read(source/'SEALED.json')['scientific_pids'])
    assert all(not Path('/proc',str(read(source/f)['pid'])).exists() for f in ['launch.json','driver_started.json'])
    rows=[json.loads(s) for s in (source/'cases.jsonl').read_text().splitlines()]
    train=[r for r in rows if r['role'].startswith('TRAIN_')];assert len(train)==64
    records=[];groups=collections.defaultdict(set)
    for index,r in enumerate(train):
        assert r['full'] and sha(r['path'])==r['sha256'];a=C.arrays(r['path'])
        id=r['job_id']
        cat='old_left5' if id.startswith('dev_task6_') else 'wide_left' if id.startswith('wide117_80_dev6_LEFT') else 'wide_right' if id.startswith('wide117_80_dev6_RIGHT') else 'old_other'
        key=hashlib.sha256(b''.join(a[k].tobytes() for k in ['initial_native_integration','task_goal_xy','scene_sha256'])).hexdigest()
        groups[cat].add(key)
        records.append(dict(key=f'e{index:02d}',path=r['path'],sha256=r['sha256'],job_id=id,category=cat,
            reset_group=key,sampling_policy_hash=r['policy_hash'],sampling_model_sha256=r['loaded_model_sha256'],
            decisions=len(a['rewards']),success=r['success'],contact=r['collision'],timeout=r['timeout']))
    categories=['old_left5','old_other','wide_left','wide_right']
    seed='T442_GROUP_SPLIT_20261109'
    val={}
    for cat in categories:
        keys=sorted(groups[cat],key=lambda k:hashlib.sha256((seed+cat+k).encode()).hexdigest())
        n=max(1,math.ceil(len(keys)/4));assert len(keys)>n
        val[cat]=keys[:n]
    for r in records:r['split']='validation' if r['reset_group'] in val[r['category']] else 'train'
    train_groups={r['reset_group'] for r in records if r['split']=='train'}
    validation_groups={r['reset_group'] for r in records if r['split']=='validation'}
    assert train_groups.isdisjoint(validation_groups)
    # Same 32 minibatches for both arms, two complete histories per category.
    schedule=[]
    for batch in range(32):
        keys=[]
        for cat in categories:
            pool=sorted(r['key'] for r in records if r['category']==cat and r['split']=='train')
            assert len(pool)>=2
            keys += [pool[(batch*2+j)%len(pool)] for j in range(2)]
        assert len(keys)==len(set(keys))==8;schedule.append(keys)
    initial=source/'initial.pt.gz'
    protected={f:sha(f) for f in [str(initial),*[str(source/f'round_{i}.pt.gz') for i in range(4)],
        str(ROOT/'stage/runs/t427_fixed_representation/learn/proposed.pt.gz'),
        str(ROOT/'stage/runs/t255_overnight/s2_reset_final_checkpoint.pt.gz')]}
    p=dict(id='T442_VALUE_ONLY_GAE_VS_MONTE_CARLO_WARMUP',records=records,categories=categories,
        split_rule='Per category first ceil(number_of_actual_physical_reset_groups/4) by fixed SHA256 rank; no outcome-based selection.',
        split_seed=seed,validation_reset_groups=val,batch_schedule=schedule,
        initial_checkpoint=str(initial),initial_checkpoint_sha256=sha(initial),protected_files=protected,
        gamma_native=sp['gamma_native'],lambda_native=sp['lambda_native'],lr=sp['value_lr'],
        epochs_per_batch=4,steps_per_arm=128,arms=['GAE','MC'],seed=20261109442,
        overall_ratio=.75,category_ratio=1.02,
        max_bytes=8*2**20,model_bytes=1024**2,predictions_bytes=1024**2,
        main_seconds=1800,audit_seconds=1200,backup_seconds=600,source_seconds=300,all_seconds=4200,
        live_V_Adam=256,shadow_V_Adam=256,audit_discarded_historical_V_Adam=256,
        preflight_discarded_synthetic_V_Adam=8,native_steps=0,Actor_forwards=0,Actor_updates=0,
        gate='On grouped validation MC overall MSE<=.75*GAE and<=.75*zero; each category<=1.02*GAE and<=zero; overall start MSE no worse than GAE and zero.',
        no_best_checkpoint_selection=True,no_early_validation=True,no_grid_or_extra_seed=True,
        no_native_no_Actor_no_teacher=True,online_actor_followup_NOT_registered=True,
        caveats=['All64 trajectories are already consumed development data from4 behavior policies; this is grouped diagnostic validation, not navigation fresh16.',
                 'This unconditioned V estimates a pooled behavior-return regression, not proven value of the final online policy.',
                 'GAE arm refreshes detached lambda targets at each batch, fixed within4 updates, versus complete MC targets in MC arm.',
                 'Calibration pass is only eligibility for a separate future online test, not six-metre navigation success.'],
        authorization='Nov9 user next step after explicit reviewed zero-Actor value-calibration proposal; independent finite protocol, no T441 continuation')
    files={str(f) for f in HERE.glob('*.py')}
    files|={str(ROOT/'stage/experiments/joint_recurrent_actor_critic/physical_gae.py'),str(receipt),str(source/'protocol.json'),str(source/'audit.json'),str(source/'summary.json'),str(source/'cases.jsonl'),str(initial)}
    files|={r['path'] for r in records};files|=set(protected)
    p['inputs']={f:sha(f) for f in sorted(files)}
    before=used(ROOT/'stage/runs');assert before+p['max_bytes']<=2**30
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T442_BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',global_before=before,
        global_complete_reserved=before+p['max_bytes'],max_bytes=p['max_bytes'],
        native=0,Actor_forwards=0,live_V_Adam=256,shadow_V_Adam=256,audit_V_Adam=256,preflight_V_Adam=8,
        rows_by_category_split={cat:{split:sum(r['category']==cat and r['split']==split for r in records) for split in ['train','validation']} for cat in categories},
        reset_groups_by_category={cat:len(groups[cat]) for cat in categories},
        storage=dict(two_models=2*1024**2,three_prediction_files=3*1024**2,metadata_and_logs=3*1024**2),
        main_seconds=p['main_seconds'],audit_seconds=p['audit_seconds'],all_seconds=p['all_seconds']))
    verify(p,True);print('T442_REGISTERED',flush=True)


def preflight():
    p=read(OUT/'protocol.json');verify(p,True);assert not (OUT/'preflight.json').exists()
    assert torch.cuda.is_available()
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    initial=C.source_initial(p);seed=np.random.default_rng(p['seed']);batch={};keys=[]
    for i in range(8):
        key=f's{i}';keys.append(key)
        x=np.repeat(initial['mean'].numpy()[None],32,0)+seed.normal(size=(32,120))*.01
        x[:,112]=np.linspace(1.,.5,32);reward=seed.normal(size=32)*.001;reward[-1]=(-1 if i%2 else 1)
        counts=np.full(32,100,np.int64);g=0.;mc=np.zeros(32)
        for k in range(31,-1,-1):g=reward[k]+p['gamma_native']**int(counts[k])*g;mc[k]=g
        batch[key]=dict(x=torch.tensor(x,device='cuda'),extended=torch.tensor(np.concatenate([x,x[-1:]]),device='cuda'),
            mc=torch.tensor(mc,device='cuda'),mc_numpy=mc,counts=counts,rewards=reward)
    pp=dict(p,batch_schedule=[keys],epochs_per_batch=2)
    calls=0;results={};start=time.monotonic()
    for arm in p['arms']:
        a=C.make(p);b=C.make(p);assert digest(C.state(*a))==digest(C.state(*b))
        before=C.state(*a)
        ra=C.fit(a,batch,pp,arm,lambda:None,lambda *a:None)
        rb=C.fit(b,batch,pp,arm,lambda:None,lambda *a:None)
        assert ra==rb and digest(C.state(*a))==digest(C.state(*b))
        mature=C.state(*a);C.restore(a,before);assert digest(C.state(*a))==digest(before)
        C.restore(a,mature);assert digest(C.state(*a))==digest(mature)
        calls+=ra['steps']+rb['steps'];results[arm]=ra
        del a,b
    assert calls==8
    # Full episode graph and initial CPU/GPU equivalence, no extra Adam.
    v,opt=C.make(p);cpu=C.Value(initial).eval()
    x=torch.tensor(np.repeat(initial['mean'].numpy()[None],400,0),device='cuda')
    pred=v(x);err=float((pred.detach().cpu()-cpu(x.cpu())).abs().max());assert err<1e-10
    pred.square().mean().backward();assert all(torch.isfinite(t.grad).all() for t in v.parameters())
    torch.cuda.synchronize()
    save(OUT/'preflight.json',dict(state='PASS',native=0,Actor_forward=0,discarded_synthetic_V_Adam=calls,
        split_no_physical_reset_overlap=True,full400_graph_finite=True,CPU_GPU_initial_max_error=err,
        independent_fresh_and_mature_optimizer_shadow_exact=True,results=results,
        GPU=torch.cuda.get_device_name(0),peak_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_s=time.monotonic()-start))
    print('T442_PREFLIGHT_PASS',flush=True)


def main():
    p=read(OUT/'protocol.json');verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS'
    start=time.time();launch=read(OUT/'launch.json');deadline=launch['epoch']+p['main_seconds'];last=0.
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(OUT/'main_started.json',dict(pid=os.getpid(),epoch=start,deadline=deadline))
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('T442 external stop')))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T442 main deadline')));signal.alarm(max(1,int(deadline-time.time())))
    arm_name='SETUP';state='FAILED';result={};error=None
    def guard():
        nonlocal last
        assert time.time()<deadline-2
        if time.time()-last>5:
            last=time.time();verify(p);save(OUT/'status.json',dict(state=arm_name,epoch=last,pid=os.getpid()))
    def event(which):
        def emit(where,k):
            with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([arm_name,which,where,k])+'\n')
        return emit
    try:
        dataset=C.data(p);zero_v,_=C.make(p)
        zero,zero_pred=C.evaluate(zero_v,dataset,p);del zero_v
        np.savez_compressed(OUT/'zero_predictions.npz',**zero_pred);assert (OUT/'zero_predictions.npz').stat().st_size<=p['predictions_bytes']
        for arm_name in p['arms']:
            guard();objects=C.make(p);shadow=C.make(p)
            assert digest(C.state(*objects))==digest(C.state(*shadow))
            before=C.state(*objects)
            try:
                live=C.fit(objects,dataset,p,arm_name,guard,event('live'))
                repeat=C.fit(shadow,dataset,p,arm_name,guard,event('shadow'))
                assert live==repeat and digest(C.state(*objects))==digest(C.state(*shadow))
            except BaseException:
                C.restore(objects,before);save(OUT/f'{arm_name}_interruption.json',dict(whole_V_and_Adam_restored=digest(C.state(*objects))==digest(before)));raise
            assert live['steps']==128
            model=C.export(objects,OUT/f'{arm_name}.pt.gz',p['model_bytes'])
            # Only the fixed final head is evaluated; validation is never used
            # to stop, select a checkpoint or change fitting.
            metrics,pred=C.evaluate(objects[0],dataset,p)
            np.savez_compressed(OUT/f'{arm_name}_predictions.npz',**pred)
            assert (OUT/f'{arm_name}_predictions.npz').stat().st_size<=p['predictions_bytes']
            result[arm_name]=dict(fit=live,model=model,metrics=metrics);save(OUT/f'{arm_name}_result.json',result[arm_name])
            del objects,shadow;torch.cuda.empty_cache()
        screening=C.screen(zero,result['GAE']['metrics'],result['MC']['metrics'],p)
        result.update(zero_metrics=zero,screening=screening)
        state='COMPLETE'
    except BaseException:
        error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,arm=arm_name,pid=os.getpid(),native=0,Actor_forward=0))
    finally:
        signal.alarm(0)
        save(OUT/'summary.json',dict(state=state,pid=os.getpid(),elapsed_s=time.time()-start,result=result,error=error,
            native_steps=0,Actor_forward=0,Actor_updates=0,live_V_Adam=sum(x['fit']['steps'] for x in result.values() if isinstance(x,dict) and 'fit'in x),
            fresh16=None,reload78=None,six_gate=False,eight_started=False))
        save(OUT/'status.json',dict(state=state,epoch=time.time()))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert state=='COMPLETE',error


def audit():
    p=read(OUT/'protocol.json');verify(p,True);s=read(OUT/'summary.json');assert s['state']=='COMPLETE'
    assert not Path('/proc',str(s['pid'])).exists() and not (OUT/'audit.json').exists()
    start=time.time();launch=read(OUT/'launch.json');deadline=min(time.time()+p['audit_seconds'],launch['epoch']+p['main_seconds']+p['audit_seconds'])
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T442 audit deadline')));signal.alarm(max(1,int(deadline-time.time())))
    events=[json.loads(x) for x in (OUT/'optimizer_calls.jsonl').read_text().splitlines()];assert len(events)==1024
    dataset=C.data(p);zero,_=C.make(p);zero_metrics,zero_pred=C.evaluate(zero,dataset,p);del zero
    assert zero_metrics==s['result']['zero_metrics']
    for key,pred in zero_pred.items():
        with np.load(OUT/'zero_predictions.npz') as z:assert np.array_equal(z[key],pred)
    actual={};calls=0
    for arm in p['arms']:
        for which in ['live','shadow']:
            e=[r[2:] for r in events if r[:2]==[arm,which]]
            assert e==[[where,k] for k in range(128) for where in ['BEFORE','AFTER']]
        objects=C.make(p)
        fit=C.fit(objects,dataset,p,arm,lambda:None,lambda *a:None)
        expected=s['result'][arm];assert fit==expected['fit'];assert digest(C.state(*objects))==expected['model']['state_hash']
        assert sha(expected['model']['path'])==expected['model']['sha256']
        model=torch.load(__import__('io').BytesIO(__import__('gzip').decompress(Path(expected['model']['path']).read_bytes())),map_location='cpu',weights_only=False)
        assert digest(model['state'])==digest(C.state(*objects))
        metrics,pred=C.evaluate(objects[0],dataset,p);assert metrics==expected['metrics'];actual[arm]=metrics
        with np.load(OUT/f'{arm}_predictions.npz') as z:
            assert set(z.files)==set(pred)
            for key,value in pred.items():assert np.array_equal(z[key],value)
        calls+=fit['steps'];del objects;torch.cuda.empty_cache()
    assert C.screen(zero_metrics,actual['GAE'],actual['MC'],p)==s['result']['screening']
    verify(p,True);signal.alarm(0)
    save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),native_steps=0,Actor_forwards=0,
        discarded_historical_V_Adam=calls,all_V_Adam_traces_predictions_exact=True,
        split_no_actual_reset_overlap=True,live=256,shadow=256,screening=s['result']['screening'],elapsed_s=time.time()-start))
    print('T442_AUDIT_PASS',flush=True)


def launch():
    p=read(OUT/'protocol.json');verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS'
    assert not (OUT/'launch.json').exists()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    now=time.time()
    with (OUT/'driver_console.log').open('xb') as f:
        child=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T442',
            '--why=Bounded value calibration','timeout','--signal=TERM','--kill-after=2',str(p['all_seconds']),
            'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=child.pid,epoch=now,main_stop=now+p['main_seconds'],all_stop=now+p['all_seconds'],
        source_remote=receipt['remote_commit'],protocol_sha256=sha(OUT/'protocol.json'),native_steps_authorized=0,Actor_updates_authorized=0))
    print('T442_ACTUALLY_LAUNCHED',child.pid,flush=True)


if __name__=='__main__':globals()[sys.argv[1]]()
