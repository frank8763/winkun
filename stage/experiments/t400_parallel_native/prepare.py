"""Exact 1/2/4 process pilot; all admission precedes native physics."""
from t400_contract import *
import ast

def prepare():
    assert not OUT.exists();s=read(PREVIOUS/'summary.json');assert s['state']=='COMPLETE' and read(PREVIOUS/'audit.json')['state']=='PASS'
    for name in ['launch.json','driver_started.json','main_started.json']:assert not Path('/proc',str(read(PREVIOUS/name)['pid'])).exists()
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T399.json';assert read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    old=read(PRIOR/'protocol.json');paths=[*HERE.glob('*.py'),MODEL,REFERENCE,PREVIOUS/'summary.json',PREVIOUS/'audit.json',PREVIOUS/'protocol.json',receipt,DOC/'T400_PLAN.md',DOC/'PARALLEL_SIMULATION_REVIEW.md',Path(old['source_protocol']),ROOT/'stage/experiments/t399_rejected_warm_diagnostic/t399_contract.py',ROOT/'stage/experiments/t394_takeover_diagnostic/t394_contract.py',ROOT/'stage/experiments/t393_safe_demo_pg/t393_learning.py',ROOT/'stage/experiments/t392_demo_closed_loop/t392_learning.py',ROOT/'stage/experiments/t387_full_horizon_night/night_common.py',ROOT/'stage/experiments/t381_native_mppi_teacher/run.py']
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    jobs=[dict(id=i,job=old['centres'][i%2]) for i in range(4)]
    p=dict(id='T400_FROZEN_FIXED_WORK_CPU_PROCESS_SCALING',source_protocol=old['source_protocol'],inputs={str(f):sha(f) for f in sorted(set(paths))},user_authorization='2026-11-06 many cloned cars in parallel, self-arrange rigorous bounded work',raw_sigma=old['raw_sigma'],policy_hash=s['policy_hash'],jobs=jobs,concurrency=[1,2,4],holds=50,physics_per_job=5000,max_physics=60000,max_commands=24,full_bytes=64*1024,final_bytes=2560*1024,main_seconds=300,audit_seconds=60,backup_seconds=120,source_seconds=120,all_seconds=600,Actor_updates=0,optimizer_updates=0,backward_calls=0,old275=None,main54=None,fresh16=None,reload78=None,seven_eight=False,limits='Exactlysame4jobs in1/2/4processes;10sengineeringcensor;0learning;notnavigationacceptance;stoponanydifference')
    used=N.K.S.used(ROOT/'stage/runs');worst=19*p['full_bytes']+1024**2
    assert worst<=p['final_bytes'] and used+p['final_bytes']<=1024**3 and shutil.disk_usage(ROOT).free>=40*1024**3 and os.cpu_count()>=4
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T400_BUDGET.json',dict(state='FULL_RESERVATION_PASS',global_before=used,reserved=used+p['final_bytes'],own=p['final_bytes'],worst_output_bytes=worst,max_physics=60000,main_commands=19,preflight_commands=2,all_commands=21,max_commands=24,optimizer=0,backward=0,all_seconds=600))
    print('T400_REGISTERED_NOT_STARTED')

def preflight():
    N.offline();p=read(OUT/'protocol.json');base=verify(p);N.configure(OUT);assert not (OUT/'preflight.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);forbid_learning();q=frozen(base,p);a=N.S.arrays(PREVIOUS/'episodes/origin_6m_LEFT.npz')
    # Reuse the existing, fully audited physical inputs for zero-native checks.
    audit_array(q,a,0);n=sizes(a);assert n<=p['full_bytes']
    m,native,create,w,_=N.S.P.C.B.setup(OUT/'preflight_native',base);m.ACTIVE=copy.deepcopy(m.ACTIVE);m.ACTIVE['circle_shift_x_m']=p['jobs'][0]['job']['circle_shift_x_m'];s=None
    try:
        def stop(physical=False):
            if physical:raise C.StageStop('T400 zero physical preflight')
        s=C.Session(p['jobs'][0]['job'],native,create,w,stop,lambda:(_ for _ in ()).throw(AssertionError('No preflight physics')),q)
        snap=s.capture();s.restore(snap);initial=s.initial.copy();assert equal(initial,a['initial_native_integration'])
        try:s.step();raise AssertionError('No cancellation')
        except C.StageStop:pass
        assert s.steps==0 and not s.rows and equal(initial,N.K.R.integration(s.env))
        C.install_stop_handlers();os.kill(os.getpid(),signal.SIGTERM)
        try:C.check_stop();raise AssertionError('No SIGTERM token')
        except C.StageStop:pass
        C.STOP_REQUESTED=None
    finally:
        if s:s.close()
        w.close()
    save(OUT/'preflight.json',dict(state='PASS',native=0,geometry=1,native_setup_commands=1,optimizer=0,backward=0,existing_history_actions_rewards_exact=True,physical_cancel_and_SIGTERM=True,incompressible50_bytes=n,policy_hash=q.phash()))
    print('T400_ZERO_NATIVE_ZERO_OPTIMIZER_PREFLIGHT_PASS')

if __name__=='__main__':globals()[sys.argv[1]]()
