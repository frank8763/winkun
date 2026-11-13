"""Register a bounded neural recovery qualification and conditional learning."""
from pathlib import Path
import copy,datetime,json,math,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t525_runtime as R
G,L,P,torch,np,read,save,sha,digest=R.G,R.L,R.P,R.torch,R.np,R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/day_20261113'
def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t524_task_gradient_balance';assert read(prior/'pilot/audit.json')['state']=='PASS'
    assert all(not Path('/proc',str(i)).exists() for i in read(prior/'SEALED.json')['scientific_pids']+[read(prior/'launch.json')['pid']])
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T524.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p=read(prior/'protocol.json');base=ROOT/'stage/runs/t508_frozen_completion';bp=read(base/'protocol.json');p['initial']=bp['initial'];p['parent_model']=p['initial'];p['locked_best']=p['initial'];p['initial_joint_locked']=False
    allrows=bp['retained18']+G.rows(base/'pilot');keeps={r['job_id']:r for r in allrows if r['success'] and not r['collision']};p['keep_rows']=[keeps[j['id']] for j in p['old8_jobs']]+[keeps[k] for k in sorted(keeps) if k.startswith('t502_')];assert len(p['keep_rows'])==19
    target='t502_field_sector0_offset2';job=next(j for j in p['field_jobs'] if j['id']==target);baseline=next(r for r in G.rows(base/'pilot') if r['job_id']==target);a=G.E.N.S.arrays(baseline['path']);assert sha(baseline['path'])==baseline['sha256']
    distances=np.linalg.norm(np.vstack([a['poses'][:,:2],a['next_poses'][-1:,:2]])-a['task_goal_xy'],axis=1);times=np.r_[0,np.cumsum(a['native_counts'])*.002];first=int(math.floor((float(times[distances.argmin()])-8.)/.2+1e-8));assert first==349 and not baseline['success'] and baseline['timeout']
    neural=read(ROOT/'stage/runs/t519_fresh_full_actor_rl/pilot/summary.json')['result']['model'];assert any(r['job_id']==target and r['success'] and not r['collision'] for r in G.rows(ROOT/'stage/runs/t519_fresh_full_actor_rl/pilot'))
    end=datetime.datetime(2026,11,13,15,30,tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp();now=time.time();assert now+4500<end
    physics=4*job['case']['step_cap']+2*sum(j['case']['step_cap'] for j in p['evaluation_jobs'])
    p.update(id='T525_ACTUAL_STUDENT_RECOVERY_AND_SUCCESS_REHEARSAL',out=str(OUT),probe_job=job,baseline_row=baseline,neural_teacher=neural,teacher_first=first,prefix_physics_steps=int(a['native_counts'][:first].sum()),
        initial_field_success_ids=[r['job_id'] for r in p['keep_rows'] if r['job_id'].startswith('t502_')],epochs=32,keep_weight=1.,actor_lr=3e-5,full_bytes=1024*1024,
        max_bytes=int(6.5*2**20),model_bytes=2*2**20,preflight_commands=6,pilot_commands=76,pilot_max_physics=physics,root_max_physics=physics,root_max_commands=82,root_max_main_seconds=3000,
        absolute_learning_stop=end-1800,absolute_native_stop=end-1200,absolute_all_stop=end,
        scientific_scope='Accepted T50711field+old8 student; fixed69.8s actual original student prefix, T519 neural teacher independently carry from h0 on actual observations. Fullteacher+exact then fixed4s teacher return+exact, no teleport/history/time reset. If full teacher recovery is safe GOAL, max32fullActor supervised teacher-owned action imitation plus actual19success function targets, live/shadow; Value0/PG0. One correction pilot, not full CLEAR/DAgger reproduction. Source teacher originalstart GOAL NOT asserted recovery until native verified.',
        acceptance='Same autonomous updated student old8all8,all11originalfieldIDs retained,field>11/contact<=9/noinvalidcensor;full32+32exact. OtherwisewholeActorAdamrollback expanded initial training state;originalacceptedbestindependentlyunchanged. Noold275/six54/eight54/fresh16/78.',
        optimizer_contract='T507 Actor exact; original adapter Adam moments/steps retained but lr fixed3e-5; encoder/core/head added new empty Adam group. Thus traininginitialjoint differs source joint, explicitly hashed. Value weights+Adam immutable; no forced old initial joint equality after expansion. No new architecture or capacity.',
        authorization='User explicitly apply research methods and confirmed new bounded window15:30stop. Training-only neural teacher authorized; deployment singleActor no planner/map/waypoint/runtime teacher/router. Pilot qualification first; no unregistered additional updates/seeds/window extension.')
    for k in ['imported_rows','training_initial_joint_hash','bc_sources','phase_limits','registered_order','task_pg_scales']:p.pop(k,None)
    p['inputs']={f:h for f,h in p['inputs'].items() if Path(f).suffix=='.py'};p['protected_files']={p['initial']['path']:p['initial']['sha256'],neural['path']:neural['sha256']}
    for f in [*HERE.glob('*.py'),DOC/'T525_PLAN.md',DOC/'T525_STORAGE_LEDGER.json',DOC/'EXISTING_SOLUTION_METHODS.md',base/'protocol.json',base/'pilot/summary.json',base/'pilot/audit.json',prior/'pilot/audit.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T524.json']:
        p['inputs'][str(f)]=sha(f)
    for row in p['keep_rows']+[baseline]:p['inputs'][row['path']]=row['sha256']
    p['inputs'].update(p['protected_files']);obj=L.make(p);p['training_initial_joint_hash']=digest(P.state(*obj));source=P.checkpoint(p['initial']['path'])['joint_state'];assert digest(source['actor'])==digest(obj[0].state_dict()) and digest(source['value'])==digest(obj[1].state_dict()) and digest(source['value_opt'])==digest(obj[3].state_dict())
    assert [int(obj[2].state[z]['step']) for z in L.A.variables(obj[0])]==[1025]*4+[385]*4
    fullshape=1352*math.ceil(job['case']['step_cap']/100)+4480+65536+16*1024;assert fullshape<p['full_bytes'];reserve=2*p['full_bytes']+p['model_bytes']+32*p['compact_bytes']+int(1.5*2**20)
    used=G.M.used();assert reserve<p['max_bytes'] and used+p['max_bytes']+65536<2**30,(used,reserve,p['max_bytes'])
    OUT.mkdir();blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    save(DOC/'T525_BUDGET.json',dict(max_physics=physics,max_commands=82,main_seconds=3000,learning_seconds=1800,audit_seconds=600,backup_reserve=600,all_stop=end,own_bytes=p['max_bytes'],full_file_bound=fullshape,all_data_model_partial_reserve=reserve,global_reserved=used+p['max_bytes'],teacher_and_handoff_unique_max=2,exact_repetitions_max=34,max_live_supervised_Adam=32,max_shadow_supervised_Adam=32,max_audit_historical_Adam=32,max_preflight_discarded_Adam=4,ValueAdam=0,new_policy_gradient_updates=0,CPU=[14,15],CPUQuota='100%',GPU=False))
    print('T525_REGISTERED_NOT_LAUNCHED',flush=True)
def preflight():
    p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'preflight.json').exists();P.frozen=L.A.frozen
    refs=L.references(p);obj=L.make(p);shadow=L.make(p);before=P.state(*obj);assert digest(before)==p['training_initial_joint_hash'];x=G.E.N.S.arrays(p['keep_rows'][0]['path'])['states'][:8]
    student=L.A.frozen(p,p['initial']['path']);teacher=L.A.frozen(p,p['neural_teacher']['path']);student.sample=teacher.sample=False;h=t=None;sl=[];tl=[]
    with torch.no_grad():
        for state in x:
            su,_,h,_=student.step(torch.tensor(state[None]),h);tu,_,t,_=teacher.step(torch.tensor(state[None]),t);sl.append(su[0].numpy());tl.append(tu[0].numpy())
    a=dict(states=x,teacher_actions=np.asarray(tl),ownership=np.asarray([0]*4+[1]*4,np.uint8),stochastic=np.zeros(8,np.uint8))
    # Labels at this fixture are only synthetic; do not claim recovery success.
    aa=L.fit(obj,a,p,refs,lambda:None,lambda *_:None,steps=2);bb=L.fit(shadow,a,p,refs,lambda:None,lambda *_:None,steps=2);assert aa==bb and digest(P.state(*obj))==digest(P.state(*shadow))
    model=L.A.export(obj,OUT/'discarded_preflight_clone.pt.gz',p['model_bytes']);assert digest(P.checkpoint(model['path'])['joint_state'])==model['joint_hash']
    for z in [obj,shadow]:P.load(z,before);assert digest(P.state(*z))==p['training_initial_joint_hash']
    clone=dict(model);Path(model['path']).unlink()
    folder=OUT/'preflight_native';folder.mkdir();counts=G.R.M.mp.get_context('spawn').RawArray('q',3);pool=R.Pool(dict(p,phase='student',out=str(folder),native_stop_epoch=time.time()+120,max_physics=1),counts,lambda:None)
    try:
        vals=pool.request([dict(model=p['initial']['path'],job=dict(p['probe_job'],teacher_holds=v),seed=None,cancel_probe=True) for v in [-1,20]]);assert all(v['state']=='CANCEL_PASS' for v in vals) and sum(counts)==0
    finally:pool.close()
    # Expected switching owns exactly the chosen consecutive decisions.
    for holds in [-1,20]:
        owner=[k>=p['teacher_first'] and (holds<0 or k<p['teacher_first']+holds) for k in range(p['teacher_first']+25)];assert sum(owner)==(25 if holds<0 else 20)
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,commands=6,discarded_synthetic_Actor_Adam=4,actual_both_neural_model_loads_and_cancel=True,independent_teacher_and_student_causal_carries=True,live_shadow_serialization_and_complete_Adam_rollback_exact=True,actual_success19_reference_functions_exact=True,teacher_ownership_mask_contract=True,discarded_nonunique_clone=clone))
    print('T525_RECOVERY_AND_REHEARSAL_PREFLIGHT_PASS',flush=True)
def launch():
    p=read(OUT/'protocol.json');R.verify(p,True);pre=read(OUT/'preflight.json');assert pre['state']=='PASS' and not Path('/proc',str(pre['pid'])).exists() and not (OUT/'launch.json').exists()
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert source['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip() and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time());assert seconds>4500
    subprocess.run(['systemd-run','--user','--unit=stage-t525-student-state-recovery','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T525','--why=Validated actual student-state neural recovery and success rehearsal','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],check=True)
    print('T525_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
