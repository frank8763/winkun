"""Register one daytime diagnostic; never restart a completed directory."""
from pathlib import Path
import json,math,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t514_runtime as R
G,P,torch,np,read,save,sha,digest=R.G,R.P,R.torch,R.np,R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/day_20261113'

def prepare():
    assert not OUT.exists()
    prior=ROOT/'stage/runs/t513_goal_feedback_teacher'
    assert read(prior/'probe/audit.json')['state']=='PASS'
    assert all(not Path('/proc',str(i)).exists() for i in read(prior/'SEALED.json')['scientific_pids'])
    assert not Path('/proc',str(read(prior/'launch.json')['pid'])).exists()
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T513.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p=read(prior/'protocol.json');now=time.time();cap=p['probe_job']['case']['step_cap']
    p.update(id='T514_STRICT_MAPPING_AND_EARLY_TEACHER_RECOVERY',out=str(OUT),teacher_first=p['recovery_start_decision']-20,physical_wheelbase_m=.60,
        nominal_speed_m_s=.08,slow_distance_m=.5,minimum_heading_speed_fraction=.25,curvature_distance_floor_m=.2,
        max_bytes=6*2**20,main_seconds=600,learning_seconds=0,audit_seconds=300,max_physics=2*cap,max_commands=6,
        root_max_physics=2*cap,root_max_commands=12,preflight_commands=6,
        absolute_learning_stop=now,absolute_native_stop=p["absolute_native_stop"],absolute_all_stop=p["absolute_all_stop"],
        controller_contract='Strict real actuator inverse: drive=2sqrt(command_v/(speed_gain*maxwheel))-1, turn=desiredsteer/maxsteer; point-goal curvature 2sin(error)/max(distance,.2), actual fullplant Ackermann wheelbase.60 no kinematic yawgain override. Desired speed .08 *min(1,d/.5)*max(.25,coserror). Actual bound strict method identity and command roundtrip asserted every teacher decision.',
        scientific_scope='Engineering recovery after verified T513 wrong base-class inverse; complete raw T513 retained. Correct mapping plus earlier33.2s prefix are joint changes, not isolated causal test. One teacher+exact,0Actor/Adam/PG. All physical caps and original09? night are separate; this daytime recovery retains T513 earlier absolute deadlines.',
        authorization='Nov13 user start next round and autonomous implementation fixes; training-only simulator teacher authorized, deployment unchanged.')
    a=R.E.N.S.arrays(p['baseline_row']['path']);p['prefix_physics_steps']=int(a['native_counts'][:p['teacher_first']].sum())
    p['combined_T513_T514_max_physics']=4*cap;p['combined_T513_T514_max_commands']=24
    p.pop('recovery_seeds',None)
    p['inputs']={f:v for f,v in p['inputs'].items() if Path(f).suffix=='.py'}
    for f in [*HERE.glob('*.py'),DOC/'T514_PLAN.md',DOC/'T514_PRIMARY_SOURCES.md',DOC/'T513_MAPPING_POSTMORTEM.md',prior/'probe/summary.json',prior/'probe/audit.json',prior/'SEALED.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T513.json']:
        p['inputs'][str(f)]=sha(f)
    for f in [ROOT/'stage/runtime/mk-mini/training/e2e_phase10_lidar_v7_curriculum/phase10_strict_mujoco_env.py',ROOT/'stage/runtime/mk-mini/training/e2e_phase10_lidar_v7_curriculum/stage9_runtime_fullplant_fix_v5_2.py']:p['inputs'][str(f)]=sha(f)
    p['inputs'][p['baseline_row']['path']]=p['baseline_row']['sha256'];p['inputs'].update(p['protected_files'])
    shape=1352*math.ceil(cap/100)+4480+65536+4096
    assert shape+3*2**20<p['max_bytes'] and G.M.used()+p['max_bytes']+512*1024<2**30
    OUT.mkdir();blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n'
    (HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    save(DOC/'T514_BUDGET.json',dict(native_max_seconds=600,audit_max_seconds=300,all_stop=p['absolute_all_stop'],max_physics=2*cap,all_commands=12,own_bytes=p['max_bytes'],full_array_shape_bound=shape,global_reserved=G.M.used()-G.M.used(OUT)+p['max_bytes'],CPU=[14,15],CPUQuota='100%',GPU=False,Adam=0))
    print('T514_REGISTERED_TRAINING_ONLY_TEACHER',flush=True)

def preflight():
    p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'preflight.json').exists()
    # Test the actual active strict mapper without constructing/stepping physics.
    parameters=np.asarray([.04,5.,.3,1.,.5]);pos=np.zeros(3)
    left=R.command(p,pos,np.asarray([1.,1.]),parameters);right=R.command(p,pos,np.asarray([1.,-1.]),parameters)
    assert left[1]>0 and left[1]==-right[1] and left[0]==right[0]
    assert R.command(p,pos,np.asarray([.3,0.]),parameters)[0]<R.command(p,pos,np.asarray([1.,0.]),parameters)[0]
    for goal in [[1,0],[0,1],[-1,0],[0,-1],[.0001,0]]:
        u=R.command(p,pos,np.asarray(goal),parameters);assert u.dtype==np.float32 and np.isfinite(u).all() and abs(u).max()<=1
    from training.e2e_phase10_lidar_v7_curriculum.phase10_strict_mujoco_env import Phase10StrictMuJoCoEnv
    env=object.__new__(Phase10StrictMuJoCoEnv)
    env.speed_gain,env.max_wheel,env.wheelbase,env.yaw_gain,env.max_steer=parameters
    for goal in [[1,0],[1,1],[1,-1],[.3,0],[0,1],[-1,0]]:
        u=R.command(p,pos,np.asarray(goal),parameters);R.roundtrip(p,env,pos,np.asarray(goal),u)
    folder=OUT/'preflight_native' ;folder.mkdir();counts=G.R.M.mp.get_context('spawn').RawArray('q',3)
    pool=R.Pool(dict(p,phase='student',out=str(folder),native_stop_epoch=time.time()+120,max_physics=1),counts,lambda:None)
    try:
        values=pool.request([dict(model=p['initial']['path'],job=p['probe_job'],seed=None,cancel_probe=True) for i in range(2)])
        assert all(v['state']=='CANCEL_PASS' for v in values) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,Adam=0,commands=6,actual_teacher_workers_load_and_cancel=True,feedback_direction_symmetry_clipping_and_slowdown=True,all_workers_stopped=True))
    print('T514_PREFLIGHT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');R.verify(p,True)
    assert read(OUT/'preflight.json')['state']=='PASS' and not Path('/proc',str(read(OUT/'preflight.json')['pid'])).exists() and not (OUT/'launch.json').exists()
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert source['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time());assert seconds>1800
    args=['systemd-run','--user','--unit=stage-t514-strict-teacher-recovery','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T514','--why=Bounded training-only terminal feedback teacher','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')]
    subprocess.run(args,check=True);print('T514_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
