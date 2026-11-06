"""Register only the two user-approved frozen handoff durations."""
from t397_contract import *
import ast

def prepare():
    assert not OUT.exists();previous=read(PREVIOUS/'summary.json');cohort=previous['teachers'][0]
    assert previous['state']=='COMPLETE' and read(PREVIOUS/'audit.json')['state']=='PASS'
    assert cohort['full_teacher_safe'] and not cohort['short_bridge_safe'] and not previous['rounds']
    for f in ['launch.json','driver_started.json','main_started.json']:assert not Path('/proc',str(read(PREVIOUS/f)['pid'])).exists()
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T396.json';assert read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    old=read(PREVIOUS/'protocol.json');a=N.S.arrays(TEACHER)
    assert int(a['native_counts'][:267].sum())==26700 and not a['end_flags'][:287].any()
    paths=[*HERE.glob('*.py'),PARENT,REFERENCE,TEACHER,PREVIOUS/'teachers/0/bridge.npz',PREVIOUS/'teachers/0/result.json',PREVIOUS/'protocol.json',PREVIOUS/'summary.json',PREVIOUS/'audit.json',receipt,Path(old['source_protocol']),DOC/'T397_PLAN.md',DOC/'T397_HANDOFF_METHODS.md',DOC/'T396_FINDINGS.md',ROOT/'stage/experiments/t394_takeover_diagnostic/t394_contract.py',ROOT/'stage/experiments/t393_safe_demo_pg/t393_learning.py',ROOT/'stage/experiments/t392_demo_closed_loop/t392_learning.py',ROOT/'stage/experiments/t387_full_horizon_night/night_common.py',ROOT/'stage/experiments/t381_native_mppi_teacher/run.py']
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    p=dict(id='T397_FROZEN_2S_4S_TEACHER_HANDOFF_DIAGNOSTIC',source_protocol=old['source_protocol'],inputs={str(f):sha(f) for f in sorted(set(paths))},user_answer='First do frozen 2sec and4sec handoff diagnostic',teacher_holds=[10,20],teacher_start_hold=267,reference_prefix_holds=265,learner_holds=2,job=old['centre_left'],shift_x_m=old['shift_x_m'],raw_sigma=[.2,.2],parent_policy_hash=old['parent_policy_hash'],source_snapshot_sha256=cohort['snapshot_sha256'],seed=None,stochastic=False,Actor_updates=0,optimizer_updates=0,backward_calls=0,final_bytes=2*1024**2,full_bytes=400*1024,max_commands=8,max_physics=160000,main_seconds=1200,audit_seconds=120,backup_seconds=240,source_seconds=180,all_seconds=1800,historical_controls=dict(one_second=previous['teachers'][0]['short_bridge_safe'],full_teacher=previous['teachers'][0]['full_teacher_safe'],source_summary=str(PREVIOUS/'summary.json'),counted_as_new_independent_samples=False),old275=None,main54=None,fresh16=None,reload78=None,seven_eight=False,limits='Exactly2 frozen conditions+2 serialized model native repeats; no training, new teacher search, stochastic seeds, intermediate durations or automatic next experiment',deployment='single causal120 recurrent Actor, teacher forced actions only this diagnostic; no runtime hybrid deployment claim')
    used=N.K.S.used(ROOT/'stage/runs');worst=3*p['full_bytes']+512*1024
    assert worst<=p['final_bytes'] and used+p['final_bytes']<=1024**3 and shutil.disk_usage(ROOT).free>=40*1024**3
    OUT.mkdir();(OUT/'episodes').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T397_BUDGET.json',dict(state='FULL_RESERVATION_PASS',global_before=used,reserved=used+p['final_bytes'],global_limit=1024**3,own=p['final_bytes'],worst_output_bytes=worst,full2_plus_partial1=3*p['full_bytes'],metadata=512*1024,new_conditions=2,exact_repeats=2,main_commands=5,preflight_zero_physics_commands=2,all_commands=7,max_commands=8,worst_physics=160000,all_optimizer=0,all_backward=0,main_seconds=1200,audit_seconds=120,backup_seconds=240,source_seconds=180,all_seconds=1800))
    print('T397_REGISTERED_NOT_STARTED')

def preflight():
    N.offline();p=read(OUT/'protocol.json');base=verify(p);N.configure(OUT)
    assert not (OUT/'preflight.json').exists();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);freeze_optimizer()
    q=actor(base,p);fixed=digest(q.state_dict());a=N.S.arrays(TEACHER);h=None
    with torch.no_grad():
        for i,x in enumerate(a['states']):
            hb=np.zeros(16) if h is None else h[0,0].numpy().copy();_,_,h,_=q.step(torch.tensor(x[None]),h)
            assert equal(hb,a['hidden_before'][i])
            for key,v in q.tape[-1].items():assert equal(np.asarray(v),a[key][i]),(i,key)
        xx=torch.tensor(a['states']);loc=q.sequence(xx,q.base_actions(xx));meanerr=float((loc-torch.tensor(a['locations'])).abs().max())
        changed=xx.clone();changed[287:]+=1.;other=q.sequence(changed,q.base_actions(changed));assert torch.equal(loc[:287],other[:287])
    assert meanerr<1e-10 and digest(q.state_dict())==fixed
    rng=np.random.default_rng(397);padded={}
    for k,v in N.S.packed(a).items():
        shape=(401,*v.shape[1:]) if k.startswith('history_') else (400,*v.shape[1:]) if v.ndim and len(v)==len(xx) else v.shape
        padded[k]=rng.normal(size=shape).astype(v.dtype) if np.issubdtype(v.dtype,np.floating) else rng.integers(0,100,size=shape,dtype=v.dtype)
    incompressible=len(N.R.blob(padded));assert incompressible<=p['full_bytes']
    m,native,create,w,_=N.S.P.C.B.setup(OUT/'preflight_native',base);m.ACTIVE=copy.deepcopy(m.ACTIVE);m.ACTIVE['circle_shift_x_m']=p['shift_x_m'];s=None
    try:
        def forbid(physical=False):
            if physical:raise C.StageStop('T397 zero-native admission guard')
        s=C.Session(p['job'],native,create,w,forbid,lambda:(_ for _ in ()).throw(AssertionError('No physics in preflight')),q)
        snap=s.capture();s.restore(snap);initial=s.initial.copy();assert equal(initial,a['initial_native_integration'])
        try:s.step();raise AssertionError('Native guard missing')
        except C.StageStop:pass
        assert s.steps==0 and not s.rows and equal(initial,N.K.R.integration(s.env))
        C.install_stop_handlers();os.kill(os.getpid(),signal.SIGTERM)
        try:C.check_stop();raise AssertionError('SIGTERM token missing')
        except C.StageStop:pass
        C.STOP_REQUESTED=None
        geometry=dict(scene=s.scene,input=digest(s.packet()),integration=digest(initial))
    finally:
        if s:s.close()
        w.close()
    save(OUT/'preflight.json',dict(state='PASS',native=0,geometry=1,native_setup_commands=1,Actor=0,optimizer=0,backward=0,existing_full_teacher_all_hidden_and_proposals_exact=True,parent_weights_unchanged=True,mean_error=meanerr,future_causality=True,full400_incompressible=incompressible,snapshot_restore_exact=True,guard_before_native=True,SIGTERM_persistent_guard=True,geometry_contract=geometry))
    print('T397_ZERO_NATIVE_ZERO_OPTIMIZER_PREFLIGHT_PASS')
if __name__=='__main__':globals()[sys.argv[1]]()
