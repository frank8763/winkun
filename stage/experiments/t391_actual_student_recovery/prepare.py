from recovery_common import *
from teacher import compact,controls,candidates,objective
import ast,multiprocessing as mp


def main():
    assert not OUT.exists()
    previous=read(PRIOR/'summary.json')
    assert previous['state']=='COMPLETE' and previous['all_workers_stopped']
    assert read(PRIOR/'audit.json')['state']=='PASS'
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T389-T390R.json'
    assert read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    for f in ['launch.json','driver_started.json','main_started.json']:
        assert not Path('/proc',str(read(PRIOR/f)['pid'])).exists()
    assert not Path('/proc',str(previous['pid'])).exists()
    for pid in read(PRIOR/'workers.json')['workers']:assert not Path('/proc',str(pid)).exists()
    source=N.S.arrays(SOURCE);reference=N.S.arrays(REFERENCE)
    assert source['end_flags'][-1,0] and reference['end_flags'][-1,1]
    n=min(len(source['poses']),len(reference['poses']))
    position=np.linalg.norm(source['poses'][:n,:2]-reference['poses'][:n,:2],axis=1)
    angle=source['poses'][:n,2]-reference['poses'][:n,2];angle=np.arctan2(np.sin(angle),np.cos(angle))
    ids=np.flatnonzero((position>.05)|(np.abs(angle)>np.deg2rad(5)))
    assert len(ids);first=int(ids[0]);assert first==12
    assert int(source['native_counts'][:first].sum())==100*first
    paths=[*HERE.glob('*.py'),PARENT,SOURCE,REFERENCE,PRIOR/'summary.json',PRIOR/'audit.json',receipt,
           ROOT/'stage/runs/t377_full_human_observation_transfer/protocol.json',
           ROOT/'stage/experiments/t387_full_horizon_night/night_common.py',ROOT/'stage/experiments/t381_native_mppi_teacher/run.py',
           ROOT/'stage/experiments/t390r_worker_entry_recovery/run.py',
           DOC/'T391_PLAN.md',ROOT/'stage/docs/research_20261106_complex_navigation/CLOSED_LOOP_TRANSFER_METHODS.md']
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    p=dict(id='T391_ACTUAL_STUDENT_RECOVERY',source_protocol=str(ROOT/'stage/runs/t377_full_human_observation_transfer/protocol.json'),
           inputs={str(f):sha(f) for f in sorted(set(paths))},user_answer='Training-only automatic simulation teacher; verify recovery success before learning',
           prefix_holds=first,remaining_holds=400-first,trigger=dict(first_position_over_m=.05,or_yaw_over_degrees=5.,
           actual_position_m=float(position[first]),actual_yaw_degrees=float(np.rad2deg(angle[first])),actual_clock_s=first*.2),
           teacher=dict(seed=20261106391,initial_sd=[.12,1.],minimum_sd=[.03,.2],iterations=3,candidates=12,elite=3,alpha=.5,
                        main_seconds=3000,audit_seconds=180,max_physics=3000000,max_commands=80,final_bytes=3*1024**2),
           student=dict(live_steps=1000,shadow_steps=1000,audit_steps=1000,lr=.0001,main_seconds=1800,learning_seconds=900,
                        audit_seconds=300,max_physics=10000000,max_commands=460,final_bytes=int(9.5*1024**2)),
           all_seconds=6000,backup_seconds=600,root_bytes=512*1024,final_bytes=13*1024**2,
           combined_max_physics=13000000,combined_max_commands=550,new_RL_updates=0,
           source_parent_policy_hash=previous['results']['coherent_teacher_only']['model']['policy_hash'],
           deployment='Single causal120 recurrent Actor without maps/planner/waypoints/teacher/route/mask/phase input')
    worst_teacher=2*(first*100+36*(400-first)*100+(400-first)*100)
    base=read(p['source_protocol'])
    jobs=[*base['narrow_jobs'],*base['old_jobs'],*base['original_jobs'],*base['final_old_jobs'],*base['six_jobs'],*base['fresh_jobs'],*base['reload_jobs'],base['narrow_jobs'][0]]
    worst_student=sum(v['case']['step_cap'] for v in jobs)
    assert worst_teacher<=p['teacher']['max_physics'] and worst_student<=p['student']['max_physics']
    assert p['root_bytes']+p['teacher']['final_bytes']+p['student']['final_bytes']==p['final_bytes']
    used=N.K.S.used(ROOT/'stage/runs');assert used+p['final_bytes']<=1024**3
    assert shutil.disk_usage(ROOT).free>=40*1024**3
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T391_BUDGET.json',dict(state='FULL_RESERVATION_PASS',global_before=used,global_after_reservation=used+p['final_bytes'],
         max_global=1024**3,own=p['final_bytes'],worst_teacher_physics=worst_teacher,worst_student_physics=worst_student,
         worst_teacher_commands=1+2*(2+36),worst_student_commands=2+len(jobs),preflight_geometry_commands=3,
         worst_wall_seconds=3000+180+1800+300+600,all_seconds=p['all_seconds']))
    print('T391_REGISTERED_ACTUAL_STUDENT_STATE_QUERY_AND_CONDITIONAL_SINGLE_ACTOR_LEARNING')


def preflight():
    N.offline();p=read(OUT/'protocol.json');base=verify(p);N.configure(OUT)
    assert not (OUT/'preflight.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    pc=dict(p['teacher'],prefix_holds=p['prefix_holds'],remaining_holds=p['remaining_holds'])
    reference=N.S.arrays(REFERENCE);data=compact(pc);rng=np.random.default_rng(391)
    for array in data.values():
        if np.issubdtype(array.dtype,np.floating):array[:]=rng.normal(size=array.shape)
        elif array.dtype==np.uint16:array[:]=rng.integers(0,pc['remaining_holds'],size=array.shape,dtype=np.uint16)
        else:array[:]=rng.integers(0,256,size=array.shape,dtype=np.uint8)
    compact_bytes=len(N.R.blob(data));assert compact_bytes<=1152*1024
    zero=controls(np.zeros((16,2)),reference['actions'],pc)
    assert equal(zero,reference['actions'][np.minimum(p['prefix_holds']+np.arange(p['remaining_holds']),len(reference['actions'])-1)])
    assert objective(zero,1,[0,1,0],.1)<objective(zero,1,[1,0,0],.1)
    a=candidates(np.zeros((16,2)),np.broadcast_to(pc['initial_sd'],(16,2)),None,0,pc)
    assert a.shape==(12,16,2) and equal(a,candidates(np.zeros((16,2)),np.broadcast_to(pc['initial_sd'],(16,2)),None,0,pc))
    padded={}
    for k,array in N.S.packed(reference).items():
        if array.ndim and array.shape[0] in [len(reference['states']),len(reference['states'])+1]:
            shape=(400+int(array.shape[0]==len(reference['states'])+1),*array.shape[1:])
            if np.issubdtype(array.dtype,np.floating):padded[k]=rng.normal(size=shape).astype(array.dtype)
            else:padded[k]=rng.integers(0,256,size=shape,dtype=array.dtype)
        else:padded[k]=array
    full_bytes=len(N.R.blob(padded));assert full_bytes<=640*1024
    # Test clone optimization and masked-prefix labels using a verified successful
    # stand-in; no claim of new recovery data before native search succeeds.
    q,opt=learner(base);other,oo=learner(base);before=N.learned_state(q,opt)
    groups,_=dataset(base,q,REFERENCE,p['prefix_holds'])
    a=fit(q,opt,groups,None,2,lambda:None,lambda *_:None)
    b=fit(other,oo,groups,None,2,lambda:None,lambda *_:None)
    assert equal(a,b) and parameter_hashes(q,opt)==parameter_hashes(other,oo)
    q.load_learned(before['learned']);opt.load_state_dict(before['optimizer']);assert digest(N.learned_state(q,opt))==digest(before)
    altered=[dict(g) for g in groups];altered[1]['target']=groups[1]['target'].clone();altered[1]['target'][:p['prefix_holds']]=12345
    def grad(gs):
        opt.zero_grad(set_to_none=True)
        for g in gs:
            pred=q.sequence(g['x'],g['b']);(g['weight']*(pred[g['first']:]-g['target'][g['first']:]).square().mean()).backward()
        return [v.grad.clone() for v in q.parameters() if v.requires_grad]
    assert all(torch.equal(x,y) for x,y in zip(grad(groups),grad(altered)))
    native,create,w,job=N.R.setup(base);s=None
    try:
        s=N.R.Session(job,native,create,w,lambda **_:None,lambda:(_ for _ in ()).throw(AssertionError('Zero native preflight')),actor(base))
        snapshot=s.capture();s.restore(snapshot)
    finally:
        if s:s.close()
        w.close()
    context=mp.get_context('spawn');counts=context.Array('q',2);processes=[];pipes=[];ready=[]
    bp=dict(base,out=str(OUT/'worker_preflight'))
    try:
        for index in range(2):
            pa,pb=context.Pipe();os.environ['STAGE_PROCESS_CPU']=str(index+1)
            proc=context.Process(target=native_worker,args=(pb,index,bp,counts));proc.start();pb.close();processes.append(proc);pipes.append(pa)
        os.environ['STAGE_PROCESS_CPU']='0'
        for pipe in pipes:
            assert pipe.poll(45);result=pipe.recv();assert result.get('ready'),result;ready.append(result)
        assert all(v['affinity']==[k+1] for k,v in enumerate(ready))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except (OSError,BrokenPipeError):pass
        for proc in processes:
            proc.join(15)
            if proc.is_alive():proc.kill();proc.join(5)
    assert len(processes)==2 and all(p.exitcode==0 for p in processes) and sum(counts[:])==0
    save(OUT/'preflight.json',dict(state='PASS',native=0,geometry=3,discarded_Adam=4,
         compact_incompressible_bytes=compact_bytes,full400_incompressible_bytes=full_bytes,
         actual_startup_workers=ready,both_exit_zero=True,zero_coeff_reference_controls_exact=True,
         source_snapshot_restore_exact=True,clone_and_fresh_Adam_restore_exact=True,
         recovery_prefix_target_mask_gradient_exact=True,training_standin_is_not_new_recovery_data=True))
    print('T391_PREFLIGHT_PASS_ZERO_NATIVE_4_CLONE_ADAM_3_GEOMETRY')


if __name__=='__main__':globals()[sys.argv[1]]()
