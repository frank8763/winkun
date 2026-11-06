"""Sealed three-point zero-learning diagnostic registration and admission."""
from t394_contract import *
import ast

def prepare():
    assert not OUT.exists();previous=read(PRIOR/'summary.json')
    assert previous['state']=='COMPLETE' and read(PRIOR/'audit.json')['state']=='PASS'
    for f in ['launch.json','driver_started.json','main_started.json']:
        assert not Path('/proc',str(read(PRIOR/f)['pid'])).exists()
    rc=ROOT/'stage/docs/long_distance_20261101/release_receipt_T393.json'
    assert read(rc)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    source=ROOT/'stage/runs/t377_full_human_observation_transfer/protocol.json';base=read(source);old=read(PRIOR/'protocol.json')
    a=N.S.arrays(REFERENCE)
    for n in [260,280,300]:assert len(a['states'])>n and not a['end_flags'][:n].any() and int(a['native_counts'][:n].sum())==100*n
    paths=[*HERE.glob('*.py'),PARENT,REFERENCE,source,PRIOR/'protocol.json',PRIOR/'summary.json',PRIOR/'audit.json',rc,
        DOC/'T394_PLAN.md',DOC/'T394_TAKEOVER_METHODS.md',DOC/'T393_FINDINGS.md',
        ROOT/'stage/experiments/t393_safe_demo_pg/t393_learning.py',ROOT/'stage/experiments/t392_demo_closed_loop/t392_learning.py',
        ROOT/'stage/experiments/t387_full_horizon_night/night_common.py',ROOT/'stage/experiments/t381_native_mppi_teacher/run.py']
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    p=dict(id='T394_FROZEN_TAKEOVER_THREE_POINTS',source_protocol=str(source),inputs={str(f):sha(f) for f in sorted(paths)},
        user_answer='Do takeover-point diagnostic first',prefix_holds=[260,280,300],peers=4,seed=20261106394,
        raw_sigma=[.2,.2],shift_x_m=old['shift_x_m'],job=old['centre_left'],parent_policy_hash=previous['result']['model']['policy_hash'],
        parent_model_sha256=previous['result']['model']['sha256'],task_version='new070_80_LEFT_training_support_only',
        final_bytes=6*1024**2,full_bytes=400*1024,global_bytes=1024**3,min_free_bytes=40*1024**3,
        max_physics=960000,max_commands=30,main_seconds=1800,audit_seconds=120,backup_seconds=240,all_seconds=2220,
        Actor_updates=0,optimizer_updates=0,backward_calls=0,new_unique=12,exact_native_repeats=12,
        native_setup_commands=1,preflight_geometry_commands=1,preflight_native_setup_commands=1,
        deterministic_mode_evaluation=None,old275=None,main54=None,fresh16=None,reload78=None,
        no_expansion='Exactly three declared prefixes, four independent streams each; no grid, extra seed, training, 7/8 or frozen-mode rollout',
        deployment='single causal120 recurrent Actor; forced successful prefixes only diagnostic/training support, no runtime hybrid navigation')
    used=N.K.S.used(ROOT/'stage/runs');assert used+p['final_bytes']<=p['global_bytes']
    assert shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    OUT.mkdir();(OUT/'episodes').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    worst=13*p['full_bytes']+512*1024
    assert worst<=p['final_bytes']
    save(DOC/'T394_BUDGET.json',dict(state='FULL_RESERVATION_PASS',global_before=used,reserved=used+p['final_bytes'],limit=p['global_bytes'],
        own=p['final_bytes'],full12_plus_partial1=13*p['full_bytes'],metadata=512*1024,worst_output=worst,
        all_native_physics=24*40000,all_commands=24+3,max_commands=p['max_commands'],
        new_forced_prefix_physics=4*(26000+28000+30000),new_suffix_physics_upper=4*(14000+12000+10000),
        new_unique=12,exact_repeats=12,all_optimizer=0,all_backward=0,main_wall=1800,audit_wall=120,backup_wall=240,all_wall=2220))
    print('T394_REGISTERED_ZERO_LEARNING_NOT_STARTED')

def preflight():
    N.offline();p=read(OUT/'protocol.json');base=verify(p);N.configure(OUT)
    assert not (OUT/'preflight.json').exists();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);freeze_optimizer()
    q=actor(base,p);ref=N.S.arrays(REFERENCE);q.rng=np.random.default_rng(p['seed']);q.reset_noise();q.sample=True;h=None
    with torch.no_grad():
        for row in ref['states']:_,_,h,_=q.step(torch.tensor(row[None]),h)
    x=torch.tensor(ref['states']);tape={k:np.asarray([v[k] for v in q.tape]) for k in q.tape[0]}
    with torch.no_grad():
        loc=q.sequence(x,q.base_actions(x));mean_error=float((loc-torch.tensor(tape['locations'])).abs().max())
        lp_error=float((logp(loc,torch.tensor(tape['raw_actions']),torch.tensor(tape['atom']),p)-torch.tensor(tape['logp'])).abs().max())
    assert mean_error<1e-10 and lp_error<1e-9
    rng=np.random.default_rng(394);padded={}
    arr={k:v for k,v in ref.items() if not k.startswith('actor_proposal_')};arr.update(tape)
    for k,a in N.S.packed(arr).items():
        shape=(401,*a.shape[1:]) if k.startswith('history_') else (400,*a.shape[1:]) if a.ndim and len(a)==len(x) else a.shape
        padded[k]=rng.normal(size=shape).astype(a.dtype) if np.issubdtype(a.dtype,np.floating) else rng.integers(0,100,size=shape,dtype=a.dtype)
    padded.update(ownership=rng.integers(0,2,size=400,dtype=np.uint8),hold_sha256=rng.integers(0,256,size=(400,32),dtype=np.uint8),task_goal_xy=np.zeros(2),scene_sha256=np.zeros(32,np.uint8))
    full=len(N.R.blob(padded));assert full<=p['full_bytes']
    m,native,create,w,_=N.S.P.C.B.setup(OUT/'preflight_native',base);m.ACTIVE=copy.deepcopy(m.ACTIVE);m.ACTIVE['circle_shift_x_m']=p['shift_x_m'];s=None
    try:
        def forbidden(physical=False):
            if physical:raise C.StageStop('Zero-native guard admission')
        s=C.Session(p['job'],native,create,w,forbidden,lambda:(_ for _ in ()).throw(AssertionError('No preflight physics')),q)
        snapshot=s.capture();s.restore(snapshot);before=s.initial.copy()
        try:s.step();raise AssertionError('Physical guard not checked')
        except C.StageStop:pass
        assert s.steps==0 and equal(before,N.K.R.integration(s.env))
        C.install_stop_handlers();os.kill(os.getpid(),signal.SIGTERM)
        try:C.check_stop();raise AssertionError('Stop token missing')
        except C.StageStop:pass
        C.STOP_REQUESTED=None
        initial=dict(scene=s.scene,integration=digest(s.initial),input=digest(s.packet()))
    finally:
        if s:s.close()
        w.close()
    save(OUT/'preflight.json',dict(state='PASS',native=0,geometry=1,native_setup_commands=1,Actor=0,Adam=0,backward=0,
        parent_serialized_policy_exact=True,mean_error=mean_error,density_error=lp_error,full400_incompressible=full,
        prefix_52_56_60_before_true_goal_terminal=True,original_total_clock_80s=True,
        snapshot_native_history_hidden_restore_exact=True,physical_guard_before_native=True,actual_process_SIGTERM_token=True,initial=initial))
    print('T394_ZERO_NATIVE_ZERO_ADAM_PREFLIGHT_PASS')

if __name__=='__main__':globals()[sys.argv[1]]()
