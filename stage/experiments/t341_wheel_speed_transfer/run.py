"""One bounded wheel-command speed change and frozen-policy transfer test.

All motion is native MuJoCo. No state/time rescaling or navigation training.
"""
from pathlib import Path
import sys,os,copy,json,time,traceback,gzip,io,shutil,importlib.util,contextlib,signal
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stage/experiments/t340_demo_supported_recovery'))
import t340_core as T
np,torch,K=T.np,T.torch,T.K
OUT=ROOT/'stage/runs/t341_wheel_speed_transfer';DOC=ROOT/'stage/docs/review_20261104'
read,save,digest=T.read,T.save,T.digest

def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    assert K.S.used(OUT)<=p['final_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']

def module(p,folder):
    spec=importlib.util.spec_from_file_location('stage341_wide_course',ROOT/'stage/experiments/t331_policy_prefix_retry/recorder.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    course=read(p['course_protocol']);m.install(course)
    native,create,witness,reward=m.setup(folder)
    return m,course,native,create,witness,reward

def actor(p):
    q=T.Actor(read(p['actor_protocol']),'capg').eval().requires_grad_(False)
    v=torch.load(io.BytesIO(gzip.decompress(Path(p['model']).read_bytes())),weights_only=False,map_location='cpu')
    q.load_state_dict(dict(q.state_dict(),**v['state']['actor']),strict=True)
    assert digest(q.state_dict())==p['policy_hash'];return q

def speed_config(env,factor):
    before=K.R.integration(env).copy();m=env.model
    ids=[env.id_lr_wheel,env.id_rr_wheel]
    config=dict(original_max_wheel=float(env.max_wheel),factor=float(factor),max_wheel=float(env.max_wheel)*factor,
        actuator_ids=list(map(int,ids)),ctrlrange=m.actuator_ctrlrange[ids].tolist(),
        ctrllimited=m.actuator_ctrllimited[ids].astype(int).tolist(),gear=m.actuator_gear[ids].tolist(),
        gainprm=m.actuator_gainprm[ids].tolist(),biasprm=m.actuator_biasprm[ids].tolist(),
        timestep=float(m.opt.timestep),geometry_sha256=K.C.model_hash(env))
    assert factor in [1.,2.] and config['original_max_wheel']>0
    for aid in ids:
        assert not m.actuator_ctrllimited[aid] or config['max_wheel']<=float(m.actuator_ctrlrange[aid,1])
    env.max_wheel=config['max_wheel']
    assert np.array_equal(before,K.R.integration(env))
    return config

def prepare():
    assert not OUT.exists();assert read(DOC/'T340_AUDIT.json')['state']=='PASS'
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T340.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    old=read(ROOT/'stage/experiments/t340_demo_supported_recovery/protocol.json')
    p=dict(experiment='T341_NATIVE_WHEEL_LIMIT_DOUBLING_FROZEN_TRANSFER',actor_protocol=str(ROOT/'stage/experiments/t340_demo_supported_recovery/protocol.json'),
        model=str(ROOT/'stage/runs/t340_demo_supported_recovery/final.pt.gz'),policy_hash='58f5551259496bbf6123bd7700ab5bce8dc9d58dbf47b9b2b1e94944a2d65e46',
        course_protocol=old['course_protocol'],factors=[1.,2.],straight_native_cap=4000,nav_native_cap=40000,
        geometry_commands=2,max_commands=10,max_physics=176000,wall_seconds=900,audit_seconds=300,all_work_seconds=1800,
        final_bytes=int(3.25*1024**2),global_bytes=1024**3,min_free_bytes=40*1024**3,
        nav_file_bytes=768*1024,straight_file_bytes=128*1024,partial_bytes=768*1024,metadata_bytes=768*1024,
        native_added_by_audit=0,optimizer_calls=0,course_is_original_acceptance=False,
        contract='Frozen T340 single Actor; two exact wheel_command_limit factors1/2, no grid. Same real wide geometry1.17m/cap80s and100native hold. Fixed full-throttle straight8s diagnostic is administrative censor, not navigation. Each straight and autonomous center separately exact native replay; all prefixes/replays/failures counted. Only max_wheel command mapping changes, actuator gains/force limits/friction/integration/time untouched. Double command limit is not a claimed doubled actual top speed. Original narrow56s/old275/54/fresh/reload not executed. No promotion7/8, teacher/planner/router/Phase.')
    p['input_hashes']=dict(old['input_hashes'])
    for f in [HERE/'run.py',DOC/'T341_PLAN.md',Path(p['actor_protocol']),Path(p['model']),DOC/'T340_AUDIT.json']:
        p['input_hashes'][str(f)]=K.C.sha(f)
    assert K.S.used(ROOT/'stage/runs')+p['final_bytes']<=p['global_bytes']
    OUT.mkdir();(OUT/'episodes').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    print(json.dumps(dict(state='REGISTERED',native=0,optimizer=0,max_commands=10,max_physics=176000)))

def preflight():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists()
    torch.set_num_threads(1);torch.optim.Adam.step=lambda *_a,**_k:(_ for _ in ()).throw(AssertionError('No optimizer'))
    folder=OUT/'preflight';folder.mkdir();configs=[]
    with (folder/'factory.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        m,c,native,create,w,r=module(p,folder/'native');q=actor(p)
        try:
            for factor in p['factors']:
                s=m.R.Session(c,native,create,w,r,lambda:None,lambda:(_ for _ in ()).throw(AssertionError('Zero physics')))
                try:
                    x=K.T.L.packet(s.env,s.obs,0,40000,np.zeros(2,np.float32),0)
                    cfg=speed_config(s.env,factor);action=q.step(torch.tensor(x[None]))[0][0].detach().numpy()
                    command=s.env._policy_to_commands(np.array([1.,0.],np.float32))
                    assert abs(float(command[1])-cfg['max_wheel'])<1e-9
                    configs.append(dict(config=cfg,initial120_sha256=digest(x),initial_native_sha256=digest(s.initial),first_action=action.tolist(),full_throttle_wheel=float(command[1])))
                finally:s.close()
            assert configs[0]['initial120_sha256']==configs[1]['initial120_sha256']
            assert configs[0]['initial_native_sha256']==configs[1]['initial_native_sha256']
            assert configs[0]['first_action']==configs[1]['first_action']
        finally:w.close()
    save(OUT/'preflight.json',dict(state='PASS',geometry_commands=2,physics_steps=0,optimizer_calls=0,configs=configs,source_protocol_sha256=K.C.sha(HERE/'protocol.json')))
    print(json.dumps(dict(state='PASS',geometry=2,physics=0,configs=configs)))

def main():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'status.json').exists();assert read(OUT/'preflight.json')['state']=='PASS'
    torch.set_num_threads(1);torch.optim.Adam.step=lambda *_a,**_k:(_ for _ in ()).throw(AssertionError('No optimizer'))
    start=time.time();deadline=min(start+p['wall_seconds'],read(OUT/'launch.json')['deadline_epoch']);physics=0;commands=2;records=[];replays=[];state='FAILED';s=None;partial_saved=False
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T341 bounded native deadline')));signal.alarm(max(1,int(deadline-time.time())-3))
    def guard():
        assert time.time()<deadline and physics<=p['max_physics']
        assert K.S.used(OUT)<=p['final_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    def status(phase):save(OUT/'status.json',dict(state=phase,pid=os.getpid(),main_epoch=start,updated_epoch=time.time(),physics_steps=physics,commands=commands,deadline_epoch=deadline))
    folder=OUT/'native';folder.mkdir();q=actor(p);status('STARTING')
    try:
        with (folder/'factory.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            m,c,native,create,w,reward=module(p,folder/'runtime')
            try:
                for kind in ['straight','navigation']:
                    for factor in p['factors']:
                        reference=None
                        for replica in ['primary','exact_native_replay']:
                            guard();assert time.time()+90<deadline and commands<p['max_commands'];commands+=1
                            ident=f'{kind}_x{int(factor)}_{replica}';status(ident);speeds=[];telemetry=[]
                            def physical():
                                nonlocal physics
                                physics+=1;pose=s.env._pose();speeds.append(math_hypot(pose[3],pose[4]))
                            s=m.R.Session(c,native,create,w,reward,guard,physical);cfg=speed_config(s.env,factor)
                            initial=K.R.integration(s.env).copy();limit=p['straight_native_cap'] if kind=='straight' else p['nav_native_cap']
                            while not(s.term or s.trunc) and s.steps<limit:
                                guard();x=K.T.L.packet(s.env,s.obs,s.steps,40000,s.previous,s.pn)
                                with torch.no_grad():action=np.array([1.,0.],np.float32) if kind=='straight' else q.step(torch.tensor(x[None]))[0][0].numpy()
                                s.step(action,min(100,limit-s.steps));ids=cfg['actuator_ids']
                                telemetry.append([*s.env.data.ctrl[ids].tolist(),*s.env.data.actuator_velocity[ids].tolist()])
                            a=s.arrays();a.update(native_speed_mps=np.asarray(speeds,np.float64),actuator_telemetry=np.asarray(telemetry,np.float64),initial_native_integration=initial,final_native_integration=K.R.integration(s.env).copy())
                            row=dict(id=ident,kind=kind,factor=factor,replica=replica,physics_steps=s.steps,complete=bool(s.term or s.trunc),
                                success=bool(s.info.get('reach_goal')) and not bool(s.info.get('collision')),collision=bool(s.info.get('collision')),timeout=bool(s.trunc),
                                administrative_diagnostic_censor=kind=='straight' and not(s.term or s.trunc),remaining_m=float(s.env._goal_state()[1]),
                                config=cfg,native_speed_peak_mps=float(max(speeds)),native_speed_mean_mps=float(np.mean(speeds)),
                                latter_half_mean_mps=float(np.mean(speeds[len(speeds)//2:])),policy_hash=p['policy_hash'] if kind=='navigation' else None,
                                array_hashes={k:digest(v) for k,v in a.items()},integration_sha256=K.R.arrays_hash(K.R.integration(s.env)))
                            assert row['physics_steps']==len(speeds)==int(a['native_counts'].astype(np.int64).sum())
                            if replica=='primary':
                                b=io.BytesIO();np.savez_compressed(b,**T.packed(a));blob=b.getvalue();assert len(blob)<=p['nav_file_bytes' if kind=='navigation' else 'straight_file_bytes']
                                path=OUT/'episodes'/(ident+'.npz');path.write_bytes(blob);row.update(path=str(path),sha256=K.C.sha(path));records.append(row);reference=a
                                with (OUT/'cases.jsonl').open('a') as f:f.write(json.dumps(row,separators=(',',':'))+'\n')
                            else:
                                assert a.keys()==reference.keys() and all(np.array_equal(v,reference[k]) for k,v in a.items())
                                row.update(all_arrays_and_integration_exact=True,independent_sample=False);replays.append(row)
                                with (OUT/'replays.jsonl').open('a') as f:f.write(json.dumps(row,separators=(',',':'))+'\n')
                            s.close();s=None;guard()
                state='COMPLETE'
            finally:
                if s is not None:
                    b=io.BytesIO();np.savez_compressed(b,**T.packed(s.arrays()));blob=b.getvalue();assert len(blob)<=p['partial_bytes'];(OUT/'partial.npz').write_bytes(blob);partial_saved=True;s.close();s=None
                w.close()
    except BaseException:save(OUT/'failure.json',dict(error=traceback.format_exc(),physics_steps=physics,commands=commands,partial_saved=partial_saved))
    finally:signal.alarm(0)
    result=dict(state=state,pid=os.getpid(),main_epoch=start,elapsed_s=time.time()-start,physics_steps=physics,commands=commands,cases=records,replays=replays,
        all_workers_stopped=True,workers_spawned=0,optimizer_calls=0,original_six_stable=False,entered7or8=False,source_protocol_sha256=K.C.sha(HERE/'protocol.json'),own_bytes=K.S.used(OUT),global_bytes=K.S.used(ROOT/'stage/runs'))
    save(OUT/'summary.json',result);save(OUT/'status.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','replays']}))

def math_hypot(x,y):return float(np.hypot(float(x),float(y)))

def audit():
    p=read(HERE/'protocol.json');verify(p);s=read(OUT/'summary.json');assert not Path('/proc',str(s['pid'])).exists();assert not (DOC/'T341_AUDIT.json').exists()
    assert s['state']=='COMPLETE' and len(s['cases'])==len(s['replays'])==4
    q=actor(p);physics=0
    for row,rep in zip(s['cases'],s['replays']):
        assert K.C.sha(row['path'])==row['sha256'];a=T.arrays(row['path'])
        assert {k:digest(v) for k,v in a.items()}==row['array_hashes']==rep['array_hashes']
        assert rep['all_arrays_and_integration_exact'] and row['integration_sha256']==rep['integration_sha256']
        assert len(a['native_speed_mps'])==row['physics_steps']==int(a['native_counts'].astype(np.int64).sum());physics+=row['physics_steps']+rep['physics_steps']
        assert not a['end_flags'][:-1].any() and np.array_equal(a['states'][1:],a['next_states'][:-1])
        if row['kind']=='navigation':
            assert row['complete'] and a['end_flags'][-1].any()
            with torch.no_grad():expected=q.step(torch.tensor(a['states']))[0].numpy()
            assert np.array_equal(expected,a['actions'])
        else:assert np.array_equal(a['actions'],np.tile(np.array([1.,0.],np.float32),(len(a['actions']),1)))
        assert row['native_speed_peak_mps']==float(max(a['native_speed_mps'])) and row['native_speed_mean_mps']==float(np.mean(a['native_speed_mps']))
    assert physics==s['physics_steps']<=p['max_physics'] and s['commands']==10
    nav=[v for v in s['cases'] if v['kind']=='navigation'];base=next(z for z in nav if z['factor']==1.)
    old=[json.loads(l) for l in (ROOT/'stage/runs/t340_demo_supported_recovery/cases.jsonl').read_text().splitlines()]
    ref=next(z for z in old if z['id']=='capg_frontier_r3_p0')
    for k in set(ref['array_hashes'])&set(base['array_hashes']):assert ref['array_hashes'][k]==base['array_hashes'][k],k
    assert base['integration_sha256']==ref['integration_sha256'] and base['physics_steps']==33532
    save(DOC/'T341_AUDIT.json',dict(state='PASS',native_added=0,optimizer_added=0,all_arrays_density_actions_and_speed_recomputed=True,
        exact_native_replays=4,original_speed_control_matches_T340=True,physics_steps=physics,commands=10,summary_sha256=K.C.sha(OUT/'summary.json')))
    print('T341 audit PASS,0newnative/optimizer')

if __name__=='__main__':
    {'prepare':prepare,'preflight':preflight,'run':main,'audit':audit}[sys.argv[1]]()
