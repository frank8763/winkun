"""One fixed missing-measurement diagnostic, no model optimization."""
from t399_contract import *
import ast

def prepare():
    assert not OUT.exists();previous=read(PRIOR/'summary.json');warm=previous['result']['warm']
    assert previous['state']=='COMPLETE' and read(PRIOR/'audit.json')['state']=='PASS'
    assert not warm['accepted'] and warm['old8']['success']==7 and not previous['rounds']
    for name in ['launch.json','driver_started.json','main_started.json']:
        assert not Path('/proc',str(read(PRIOR/name)['pid'])).exists()
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T398.json'
    assert read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    old=read(PRIOR/'protocol.json');paths=[*HERE.glob('*.py'),MODEL,REFERENCE,PRIOR/'protocol.json',PRIOR/'summary.json',PRIOR/'audit.json',receipt,DOC/'T399_PLAN.md',DOC/'T398_FINDINGS.md',Path(old['source_protocol']),ROOT/'stage/experiments/t394_takeover_diagnostic/t394_contract.py',ROOT/'stage/experiments/t393_safe_demo_pg/t393_learning.py',ROOT/'stage/experiments/t392_demo_closed_loop/t392_learning.py',ROOT/'stage/experiments/t387_full_horizon_night/night_common.py',ROOT/'stage/experiments/t381_native_mppi_teacher/run.py']
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    conditions=[dict(id='recover53',job=old['centre_left'],prefix_holds=265),dict(id='recover54',job=old['centre_left'],prefix_holds=270),*[dict(id='origin_'+j['group'],job=j,prefix_holds=0) for j in old['centres']]]
    p=dict(id='T399_FROZEN_REJECTED_WARM_MISSING_TASK_MEASUREMENTS',source_protocol=old['source_protocol'],inputs={str(f):sha(f) for f in sorted(set(paths))},user_authorization='2026-11-06 self-arrange bounded scientifically rigorous next experiments and continue',conditions=conditions,raw_sigma=old['raw_sigma'],policy_hash=warm['proposed_model']['policy_hash'],parent_policy_hash=old['parent_policy_hash'],final_bytes=3584*1024,full_bytes=400*1024,max_commands=12,max_physics=320000,main_seconds=1200,audit_seconds=180,backup_seconds=240,source_seconds=180,all_seconds=1920,Actor_updates=0,optimizer_updates=0,backward_calls=0,old275=None,main54=None,fresh16=None,reload78=None,seven_eight=False,historical_parent_controls='T398 final_mode 53/54 and final_main LEFT/RIGHT; exact same immutable T393; reused not new physics or independent samples',stop='exactly4 frozen conditions plus4 serialized native repeats; no learning, search, grids or automatic successor',deployment=old['deployment'])
    used=N.K.S.used(ROOT/'stage/runs');worst=5*p['full_bytes']+1024**2
    assert worst<=p['final_bytes'] and used+p['final_bytes']<=1024**3 and shutil.disk_usage(ROOT).free>=40*1024**3
    OUT.mkdir();(OUT/'episodes').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T399_BUDGET.json',dict(state='FULL_RESERVATION_PASS',global_before=used,reserved=used+p['final_bytes'],global_limit=1024**3,own=p['final_bytes'],worst_output_bytes=worst,new_conditions=4,exact_repeats=4,main_commands=9,preflight_commands=2,all_commands=11,max_commands=12,max_physics=320000,optimizer=0,backward=0))
    print('T399_REGISTERED_NOT_STARTED')

def preflight():
    N.offline();p=read(OUT/'protocol.json');base=verify(p);N.configure(OUT)
    assert not (OUT/'preflight.json').exists();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);forbid_learning()
    q=frozen(base,p);other=frozen(base,p);fixed=digest(q.state_dict());a=N.S.arrays(REFERENCE);h=k=None
    with torch.no_grad():
        for row in a['states']:
            x=torch.tensor(row[None]);aa,al,h,_=q.step(x,h);bb,bl,k,_=other.step(x,k)
            assert torch.equal(aa,bb) and torch.equal(al,bl) and torch.equal(h,k)
        xx=torch.tensor(a['states']);loc=q.sequence(xx,q.base_actions(xx));changed=xx.clone();changed[270:]+=1
        alternative=q.sequence(changed,q.base_actions(changed));assert torch.equal(loc[:270],alternative[:270])
    assert digest(q.state_dict())==fixed
    rng=np.random.default_rng(399);padded={}
    for name,v in N.S.packed(a).items():
        shape=(401,*v.shape[1:]) if name.startswith('history_') else (400,*v.shape[1:]) if v.ndim and len(v)==len(xx) else v.shape
        padded[name]=rng.normal(size=shape).astype(v.dtype) if np.issubdtype(v.dtype,np.floating) else rng.integers(0,100,size=shape,dtype=v.dtype)
    n=len(N.R.blob(padded));assert n<=p['full_bytes']
    m,native,create,w,_=N.S.P.C.B.setup(OUT/'preflight_native',base);m.ACTIVE=copy.deepcopy(m.ACTIVE);m.ACTIVE['circle_shift_x_m']=p['conditions'][0]['job']['circle_shift_x_m'];s=None
    try:
        def stop(physical=False):
            if physical:raise C.StageStop('T399 zero native preflight')
        s=C.Session(p['conditions'][0]['job'],native,create,w,stop,lambda:(_ for _ in ()).throw(AssertionError('No preflight physical steps')),q)
        snap=s.capture();s.restore(snap);initial=s.initial.copy();assert equal(initial,a['initial_native_integration'])
        try:s.step();raise AssertionError('Physical cancellation missing')
        except C.StageStop:pass
        assert not s.rows and s.steps==0 and equal(initial,N.K.R.integration(s.env))
        C.install_stop_handlers();os.kill(os.getpid(),signal.SIGTERM)
        try:C.check_stop();raise AssertionError('Persistent signal cancellation missing')
        except C.StageStop:pass
        C.STOP_REQUESTED=None
    finally:
        if s:s.close()
        w.close()
    save(OUT/'preflight.json',dict(state='PASS',native=0,geometry=1,native_setup_commands=1,optimizer=0,backward=0,policy_hash=q.phash(),model_sha256=sha(MODEL),independent_frozen_forward_exact=True,causal_history=True,weights_unchanged=True,incompressible_bytes=n,physical_cancel_and_SIGTERM=True,snapshot_restore_exact=True))
    print('T399_ZERO_NATIVE_ZERO_OPTIMIZER_PREFLIGHT_PASS')

if __name__=='__main__':globals()[sys.argv[1]]()
