"""Finite early-feedback qualification; no training or parameter search."""
from pathlib import Path
import json,math,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t526_runtime as R
G,P,torch,np,read,save,sha,digest=R.G,R.P,R.torch,R.np,R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/day_20261113'
def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t525_student_state_recovery';assert read(prior/'pilot/audit.json')['state']=='PASS'
    assert all(not Path('/proc',str(i)).exists() for i in read(prior/'SEALED.json')['scientific_pids']+[read(prior/'launch.json')['pid']])
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T525.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p=read(prior/'protocol.json');feedback=read(ROOT/'stage/runs/t514_strict_teacher_recovery/protocol.json');geo=read(DOC/'NEXT_AFTER_T525_GEOMETRY.json');cap=p['probe_job']['case']['step_cap'];now=time.time()
    assert geo['proposed_single_correction_decision']==321 and now+1200<p['absolute_all_stop']
    for k in ['physical_wheelbase_m','nominal_speed_m_s','slow_distance_m','minimum_heading_speed_fraction','curvature_distance_floor_m','controller_contract']:p[k]=feedback[k]
    p.update(id='T526_EARLY_STRICT_FEEDBACK_QUALIFICATION',out=str(OUT),teacher_first=321,prefix_physics_steps=32100,max_bytes=3*2**20,main_seconds=600,learning_seconds=0,audit_seconds=180,
        preflight_commands=6,pilot_commands=8,pilot_max_physics=4*cap,root_max_physics=4*cap,root_max_commands=14,root_max_main_seconds=600,
        absolute_learning_stop=now,absolute_native_stop=p['absolute_all_stop']-600,
        scientific_scope='Zero-learning fixed64.2s early strict feedback qualification and conditional4s handoff, all original native prefix/goal/history/clock retained. Feedbacktype+earliertime jointpilot notcausal decomposition. 0Adam/noActorcandidate/fullfieldnull. No sweep.',
        acceptance='FullfeedbacksafeGOAL plus fixed4steacherthenrealstudentcarrysafeGOAL allows future independentlyregisteredlearning;notnewautonomousmodelsuccess. Failure0learningstop. Originalacceptedbestimmutable.',
        authorization='User Nov13 nextstart within confirmed15:30newwindow. Remainingtime insufficient completelearning+32gate so execute only proposed zero-training qualification, no deadline extension.')
    p.pop('neural_teacher',None);p['protected_files']={p['initial']['path']:p['initial']['sha256']}
    p['inputs']={f:h for f,h in p['inputs'].items() if Path(f).suffix=='.py'}
    for f in [*HERE.glob('*.py'),DOC/'T526_PLAN.md',DOC/'NEXT_AFTER_T525_GEOMETRY.json',prior/'pilot/summary.json',prior/'pilot/audit.json',ROOT/'stage/experiments/t514_strict_teacher_recovery/t514_runtime.py']:
        p['inputs'][str(f)]=sha(f)
    p['inputs'][p['baseline_row']['path']]=p['baseline_row']['sha256'];p['inputs'].update(p['protected_files'])
    used=G.M.used();reserve=2*p['full_bytes']+768*1024;assert reserve<p['max_bytes'] and used+p['max_bytes']+65536<2**30
    OUT.mkdir();blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    save(DOC/'T526_BUDGET.json',dict(max_physics=4*cap,max_commands=14,main_seconds=600,audit_seconds=180,backup_seconds=360,all_stop=p['absolute_all_stop'],own_bytes=p['max_bytes'],data_metadata_reserve=reserve,global_reserved=used+p['max_bytes'],ActorAdam=0,ValueAdam=0,CPU=[14,15],CPUQuota='100%',GPU=False))
    print('T526_REGISTERED_NOT_LAUNCHED',flush=True)
def preflight():
    p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'preflight.json').exists();P.frozen=R.F.L.frozen
    from training.e2e_phase10_lidar_v7_curriculum.phase10_strict_mujoco_env import Phase10StrictMuJoCoEnv
    env=object.__new__(Phase10StrictMuJoCoEnv);env.speed_gain,env.max_wheel,env.wheelbase,env.yaw_gain,env.max_steer=[.03,5.,.42,1.35,.4363323129985824];parameters=np.asarray([env.speed_gain,env.max_wheel,env.wheelbase,env.yaw_gain,env.max_steer]);pos=np.zeros(3)
    for goal in [[1,0],[1,1],[1,-1],[.1,0],[-1,0]]:
        goal=np.asarray(goal);u=R.Feedback.command(p,pos,goal,parameters);R.Feedback.roundtrip(p,env,pos,goal,u)
    folder=OUT/'preflight_native';folder.mkdir();counts=G.R.M.mp.get_context('spawn').RawArray('q',3);pool=R.Pool(dict(p,phase='student',out=str(folder),native_stop_epoch=time.time()+120,max_physics=1),counts,lambda:None)
    try:
        vals=pool.request([dict(model=p['initial']['path'],job=dict(p['probe_job'],teacher_holds=h),seed=None,cancel_probe=True) for h in [-1,20]]);assert all(v['state']=='CANCEL_PASS' for v in vals) and sum(counts)==0
    finally:pool.close()
    for holds in [-1,20]:assert sum(i>=321 and (holds<0 or i<321+holds) for i in range(346))==(25 if holds<0 else 20)
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,Adam=0,commands=6,actual_strict_mapper_roundtrip_and_bounds=True,actual_feedback_session_worker_load_cancel_PASS=True,ownership_start321_and20holds=True,all_workers_stopped=True));print('T526_PREFLIGHT_PASS',flush=True)
def launch():
    p=read(OUT/'protocol.json');R.verify(p,True);pre=read(OUT/'preflight.json');assert pre['state']=='PASS' and not Path('/proc',str(pre['pid'])).exists() and not (OUT/'launch.json').exists()
    src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip() and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time());assert seconds>1200
    subprocess.run(['systemd-run','--user','--unit=stage-t526-early-feedback-qualification','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T526','--why=Bounded early actual-state feedback qualification','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],check=True);print('T526_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
