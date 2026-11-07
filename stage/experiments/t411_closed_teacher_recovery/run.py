"""One bounded frozen recovery diagnostic; invocation main or preflight."""
from t411_core import *

def preflight():
    N.offline();p=plan();base=verify(p);assert not (OUT/'preflight.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);student,teacher=actors(base,p)
    source=N.S.arrays(p['source_control_file']);teacher.reset_noise();h=None
    with torch.no_grad():
        for i,x in enumerate(source['states']):
            before_h=np.zeros(16) if h is None else h[0,0].numpy().copy()
            action,location,h,_=teacher.step(torch.tensor(x[None]),h)
            assert equal(before_h,source['hidden_before'][i])
            assert equal(action[0].numpy(),source['actions'][i])
            assert equal(location[0].numpy(),source['locations'][i])
        assert equal(h[0,0].numpy(),source['last_hidden_after'])
        sequence=teacher.sequence(torch.tensor(source['states']),teacher.base_actions(torch.tensor(source['states'])))
        assert float((sequence-torch.tensor(source['locations'])).abs().max())<1e-10
    # Actual teacher state is reconstructed from the student-distribution RIGHT
    # history too. Its nonzero handoff state must remain the prefix result.
    actual=N.S.arrays(p['student_history_file']);h=None;student_h=None;teacher.reset_noise();student.reset_noise()
    with torch.no_grad():
        for i,x in enumerate(actual['states']):
            _,_,h,_=teacher.step(torch.tensor(x[None]),h)
            action,_,student_h,_=student.step(torch.tensor(x[None]),student_h)
            assert equal(action[0].numpy(),actual['actions'][i])
            if i==19:assert torch.count_nonzero(h)>0;prefix_hash=digest(h)
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    rng=np.random.default_rng(411);padded={}
    for key,value in N.S.packed(source).items():
        shape=(401,*value.shape[1:]) if key.startswith('history_') else (400,*value.shape[1:]) if value.ndim and len(value)==len(source['states']) else value.shape
        padded[key]=rng.normal(size=shape).astype(value.dtype) if np.issubdtype(value.dtype,np.floating) else rng.integers(0,100,size=shape,dtype=value.dtype)
    for key,shape,dtype in [('teacher_actions',(400,2),np.float32),('teacher_locations',(400,2),np.float64),('teacher_hidden_before',(400,16),np.float64),('teacher_hidden_after',(400,16),np.float64)]:padded[key]=rng.normal(size=shape).astype(dtype)
    padded.update(ownership=rng.integers(0,2,400,np.uint8),hold_sha256=rng.integers(0,256,(400,32),np.uint8),task_goal_xy=rng.normal(size=2),scene_sha256=rng.integers(0,256,32,np.uint8))
    full=len(N.R.blob(padded));assert full<=p['full_bytes']
    install_no_learning()
    N.configure(OUT);module,native,create,witness,_=N.S.P.C.B.setup(OUT/'preflight_native',base);original=copy.deepcopy(module.ACTIVE);session=None
    try:
        job=p['cases'][0]['job'];module.ACTIVE=copy.deepcopy(original);module.ACTIVE['circle_shift_x_m']=job['circle_shift_x_m']
        def reject(physical=False):
            if physical:raise L.StageStop('T411 zero-native cancellation')
        session=L.Session(job,native,create,witness,reject,lambda:(_ for _ in ()).throw(AssertionError('Physics in preflight')),student)
        try:session.step();raise AssertionError('Cancellation not raised')
        except L.StageStop:pass
        assert not session.rows and session.steps==0
    finally:
        if session:session.close()
        witness.close()
    save(OUT/'preflight.json',dict(state='PASS',native=0,Adam=0,backward=0,zero_physics_commands=2,
        teacher_historical_actions_locations_hidden_exact=True,teacher_sequence_error_below_1e_10=True,
        teacher_actual_student_history_nonzero_prefix_hash=prefix_hash,student_actual_history_actions_exact=True,
        actual_writer_lock=True,actual_session_before_physics_cancel=True,incompressible_combined_full_bytes=full,
        teacher_actions_closed_loop_not_timed_suffix=True));N.manifest(OUT);print('T411_PREFLIGHT_PASS_ZERO_NATIVE_ZERO_ADAM',flush=True)

def main():
    N.offline();p=plan();base=verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'summary.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);student,teacher=actors(base,p);install_no_learning();L.install_stop_handlers()
    start=time.time();deadline=min(start+p['main_seconds'],p.get('native_stop_epoch',float('inf')),p['native_stop_absolute']);physics=0;commands=0;last=0.;phase='SETUP';records=[];replays=[];state='FAILED';witness=None;source_proof=None
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(OUT/'main_started.json',dict(pid=os.getpid(),epoch=start,native_deadline=deadline,Adam=0,learning=0))
    def guard(physical=False):
        nonlocal last
        L.check_stop();assert time.time()<deadline-3 and physics<=p['max_physics'] and commands<=p['max_commands']
        if not physical and time.time()-last>3:
            last=time.time();N.capacity(OUT,p['final_bytes']);save(OUT/'status.json',dict(state=phase,pid=os.getpid(),updated_epoch=last,physics=physics,commands=commands,complete_cases=len(records),Adam=0))
    def charge():
        nonlocal physics
        physics+=1;assert physics<=p['max_physics']
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T411 native deadline')));signal.alarm(max(1,int(deadline-time.time()-2)))
    try:
        N.configure(OUT);module,native,create,witness,_=N.S.P.C.B.setup(OUT/'native',base);original=copy.deepcopy(module.ACTIVE);commands=1
        with (OUT/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(id=1,role='native_setup',physics=0,error=None,pid=os.getpid()))+'\n')
        assert p['cases'][0]['source_control'] and p['cases'][0]['first']==0
        for case in p['cases']:
            guard();assert physics+2*case['job']['case']['step_cap']<=p['max_physics']
            for repeat in range(2):
                phase=case['id']+('_exact' if repeat else '');module.ACTIVE=copy.deepcopy(original);module.ACTIVE['circle_shift_x_m']=case['job']['circle_shift_x_m'];before=physics;commands+=1;assert commands<=p['max_commands']
                arrays,row=rollout(case,student,teacher,native,create,witness,guard,charge,OUT/'partial.npz')
                row.update(role=phase,physics=physics-before,pid=os.getpid())
                assert row['physics']==row['physics_steps']
                if repeat:
                    prior=records[-1]
                    for key in ['array_digest','integration_sha256','physics_steps','success','collision','timeout','student_policy_hash','teacher_policy_hash']:assert row[key]==prior[key],key
                    row.update(exact=True,independent_sample=False);replays.append(row);filename='replays.jsonl'
                else:
                    path=OUT/'episodes'/(case['id']+'.npz');N.write_npz(path,N.S.packed(arrays),p['full_bytes']);row.update(path=str(path),sha256=sha(path),full=True);records.append(row);filename='cases.jsonl'
                    if case['source_control']:source_proof=source_control(arrays,p);save(OUT/'source_control.json',source_proof)
                with (OUT/filename).open('a') as f:f.write(json.dumps(row)+'\n')
                with (OUT/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(id=commands,role=phase,physics=physics-before,error=None,pid=os.getpid()))+'\n')
        state='COMPLETE'
    except BaseException:save(OUT/'failure.json',dict(error=traceback.format_exc(),phase=phase,physics=physics,commands=commands,Adam=0))
    finally:
        if witness:witness.close()
        signal.alarm(0);fcntl.flock(lock,fcntl.LOCK_UN);lock.close();summary=dict(state=state,pid=os.getpid(),elapsed_s=time.time()-start,physics=physics,commands=commands,Adam=0,backward=0,Actor_updates=0,records=records,replays=len(replays),source_control=source_proof,all_native_sessions_closed=True,all_workers_stopped=True,student_model_sha256=sha(p['initial_model']),teacher_model_sha256=sha(p['teacher_model']),no_six_metre_acceptance=True,seven_eight_entered=False)
        save(OUT/'summary.json',summary);save(OUT/'status.json',dict(state=state,updated_epoch=time.time(),physics=physics,commands=commands,Adam=0));N.manifest(OUT)
    assert state=='COMPLETE'

if __name__=='__main__':
    assert sys.argv[1:] in (['main'],['preflight']);globals()[sys.argv[1]]()
