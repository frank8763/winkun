from pathlib import Path
import sys,copy,contextlib,importlib.util,signal
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stage/experiments/t333_remaining_aggregation'))
import t333_learning as L
K,np,torch=L.K,L.np,L.torch
read,save=L.read,L.save
OUT=ROOT/'stage/runs/t334_trained_takeover';DOC=ROOT/'stage/docs/distance_20261103'
spec=importlib.util.spec_from_file_location('stage334_course',ROOT/'stage/experiments/t331_policy_prefix_retry/recorder.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
M.OUT=M.R.OUT=M.W.OUT=OUT

def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    assert K.S.used(OUT)<=p['final_bytes'] and L.shutil.disk_usage(ROOT).free>=p['min_free_bytes']

def policy(p,arm):
    q=L.model(read(p['learning_protocol'])).requires_grad_(False);v=L.read_patch(p['models'][arm]['path']);L.load(q,v['residual'])
    assert K.S.digest(q.state_dict())==p['models'][arm]['policy_hash'];return q

def prepare():
    assert not OUT.exists() and read(DOC/'T333_AUDIT.json')['state']=='PASS'
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T333.json'
    assert read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    lp=ROOT/'stage/experiments/t333_remaining_aggregation/protocol.json';p=read(lp);models=read(ROOT/'stage/runs/t333_remaining_aggregation/patches.json')
    p.update(experiment='T334_TRAINED_POLICY_TRUE_HANDOFF_DIAGNOSIS',learning_protocol=str(lp),
        course_protocol=str(ROOT/'stage/experiments/t331_policy_prefix_retry/protocol.json'),
        reference=str(ROOT/'stage/runs/t331_policy_prefix_retry/attempt_01.npz'),
        models={arm:models[arm+'_candidate'] for arm in ['control','recovery']},
        final_bytes=3*1024**2,payload_bytes=2304*1024,metadata_bytes=512*1024,
        max_commands=5,max_physics=160000,wall_seconds=600,final_reserve_seconds=120,
        preflight_geometry_commands=1,episode_bytes=768*1024)
    files=set(p['input_hashes'])|{str(f) for f in HERE.glob('*.py')}
    files|={str(lp),p['course_protocol'],p['reference'],str(receipt),str(DOC/'T333_AUDIT.json'),str(DOC/'T334_PLAN.md')}
    for v in p['models'].values():files.add(v['path'])
    p['input_hashes']={f:K.C.sha(f) for f in sorted(files)}
    assert K.S.used(ROOT/'stage/runs')+p['final_bytes']<=p['global_bytes']
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)

def setup(p,folder):
    course=read(p['course_protocol']);M.install(course)
    native,create,witness,reward=M.setup(folder)
    return course,native,create,witness,reward

def preflight():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists();torch.set_num_threads(1)
    folder=OUT/'preflight';folder.mkdir();s=None
    with (folder/'factory.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        course,native,create,witness,reward=setup(p,folder/'native')
        try:
            s=M.R.Session(course,native,create,witness,reward,lambda:None,lambda:(_ for _ in ()).throw(AssertionError('Zero physics')))
            result={}
            for arm in ['control','recovery']:
                q=policy(p,arm)
                with torch.no_grad():action=q.step(torch.tensor(s.initial120[None]))[0][0].numpy()
                assert bool((np.abs(action)<=1.000001).all());result[arm]=action.tolist()
            assert s.steps==0
            save(OUT/'preflight.json',dict(state='PASS',geometry_commands=1,physics_steps=0,Adam_calls=0,
                both_trained_policies_strict_load=True,first_actions=result,course_initial_native_hash_exact=True))
        finally:
            if s is not None:s.close()
            witness.close()
    print('T334 bothtrainedActor/native preflight PASS0physics0Adam')

def main():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'status.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.optim.Adam.step=M.R.U.forbid_step
    start=L.time.time();deadline=start+p['wall_seconds'];meter=K.Budget(ROOT,OUT,p,start_epoch=start);meter.charge_payload(p['payload_bytes'])
    rows=[];reps=[];s=None;witness=None;state='FAILED';phase='STARTING'
    def guard():assert L.time.time()<deadline-15;meter.hold();assert meter.physics<p['max_physics']
    def physical():meter.physics+=1
    def status():save(OUT/'status.json',dict(state=phase,pid=L.os.getpid(),main_epoch=start,updated_epoch=L.time.time(),
        deadline_epoch=deadline,commands=1+meter.commands,physics_steps=meter.physics,cases=len(rows),replays=len(reps)))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T334600s')));signal.alarm(595)
    try:
        assert read(OUT/'preflight.json')['state']=='PASS'
        with (OUT/'native_factory.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            course,native,create,witness,reward=setup(p,OUT/'native');ref,_,_=M.R.load_arrays(p['reference'])
            for arm in ['control','recovery']:
                q=policy(p,arm);reference=None
                for replay in [False,True]:
                    phase=arm+('_exact_replay' if replay else '_trained_takeover');status();meter.begin(40000)
                    s=M.R.Session(course,native,create,witness,reward,guard,physical)
                    for k in range(150):s.step(ref['actions'][k],int(ref['native_counts'][k]))
                    a=s.arrays()
                    for name,v in a.items():
                        n=151 if name.startswith('history_') else 150
                        assert np.array_equal(v,ref[name][:n]),name
                    assert K.R.arrays_hash(K.R.integration(s.env))==read(ROOT/'stage/runs/t331_policy_prefix_retry/handoff.json')['integration_sha256']
                    while not(s.term or s.trunc):
                        x=K.T.L.packet(s.env,s.obs,s.steps,40000,s.previous,s.pn)
                        with torch.no_grad():action=q.step(torch.tensor(x[None]))[0][0].numpy()
                        s.step(action);status()
                    blob,a=s.packet();assert len(blob)<=p['episode_bytes']
                    row=dict(arm=arm,physics_steps=s.steps,prefix_steps=15000,Actor_suffix_steps=s.steps-15000,
                        complete=bool(s.term or s.trunc),success=bool(s.info.get('reach_goal')) and not bool(s.info.get('collision')),
                        collision=bool(s.info.get('collision')),timeout=bool(s.trunc),remaining_m=float(s.env._goal_state()[1]),
                        integration_sha256=K.R.arrays_hash(K.R.integration(s.env)),array_hashes={k:K.S.digest(v) for k,v in a.items()},
                        policy_hash=K.S.digest(q.state_dict()),goal_xy=s.goal.tolist(),not_autonomous_from_original_start=True)
                    if replay:
                        assert row==reference['row'] and all(np.array_equal(v,reference['arrays'][k]) for k,v in a.items())
                        reps.append(dict(arm=arm,physics_steps=s.steps,all12_arrays_and_integration_exact=True,not_independent_sample=True));save(OUT/'replays.json',reps)
                    else:
                        f=OUT/(arm+'.npz');assert not f.exists();f.write_bytes(blob);row.update(path=str(f),sha256=K.C.sha(f))
                        rows.append(row);save(OUT/'cases.json',rows)
                        reference=dict(row={k:v for k,v in row.items() if k not in ['path','sha256']},arrays=a)
                    s.close();s=None;status()
        state='COMPLETE'
    except BaseException:
        save(OUT/'failure.json',dict(error=L.traceback.format_exc(),phase=phase,physics_steps=meter.physics,commands=1+meter.commands))
        if s is not None:
            try:blob,_=s.packet();(OUT/'partial.npz').write_bytes(blob)
            except BaseException:save(OUT/'partial_failure.json',dict(error=L.traceback.format_exc()))
    finally:
        if s is not None:s.close()
        if witness is not None:witness.close()
        signal.alarm(0);meter.close()
    summary=dict(state=state,pid=L.os.getpid(),main_epoch=start,elapsed_s=L.time.time()-start,deadline_epoch=deadline,
        commands=1+meter.commands,physics_steps=meter.physics,cases=rows,replays=reps,Actor_updates=0,Adam_calls=0,
        original_start_six_success=False,seven_eight_entered=False,own_bytes=K.S.used(OUT),global_bytes=K.S.used(ROOT/'stage/runs'))
    save(OUT/'summary.json',summary);save(OUT/'status.json',summary);print(L.json.dumps(summary))

def audit():
    p=read(HERE/'protocol.json');verify(p);s=read(OUT/'summary.json');dest=DOC/'T334_AUDIT.json'
    assert s['state']=='COMPLETE' and not dest.exists()
    for pid in [s['pid'],read(OUT/'launch.json')['pid']]:assert not Path('/proc/'+str(pid)).exists()
    torch.set_num_threads(1);ref,_,_=M.R.load_arrays(p['reference']);physics=0
    for row in s['cases']:
        assert K.C.sha(row['path'])==row['sha256'];a,initial,final=M.R.load_arrays(row['path'])
        assert {k:K.S.digest(v) for k,v in a.items()}==row['array_hashes']
        assert K.R.arrays_hash(final)==row['integration_sha256']
        for name,v in a.items():
            n=151 if name.startswith('history_') else 150
            assert np.array_equal(v[:n],ref[name][:n])
        q=policy(p,row['arm'])
        for k in range(150,len(a['states'])):
            with torch.no_grad():action=q.step(torch.tensor(a['states'][k:k+1]))[0][0].numpy()
            assert np.array_equal(action,a['actions'][k])
        assert int(a['native_counts'].astype(np.int64).sum())==row['physics_steps']
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].any()
        assert bool(a['end_flags'][-1,0])==row['collision'] and bool(a['end_flags'][-1,1])==row['success'] and bool(a['end_flags'][-1,2])==row['timeout']
        steps=np.r_[0,np.cumsum(a['native_counts'].astype(np.int64))[:-1]]
        assert np.max(np.abs(a['states'][:,112]-(1-steps/40000)))<1e-7
        for k,x in enumerate(a['states']):
            assert np.abs(K.H.rebuild(x,a['poses'][k],a['history_poses'][k],row['goal_xy'],a['history_goals'][k])-x).max()<=2e-6
            d0=K.H.fields(a['poses'][k],row['goal_xy'])[1];d1=K.H.fields(a['next_poses'][k],row['goal_xy'])[1];n=int(a['native_counts'][k]);c,g,t=map(bool,a['end_flags'][k]);z=K.R.GAMMA**n
            reward=-.000002*(1-z)/(1-K.R.GAMMA)+.03*d0-z*.03*d1*(not(c or g or t))+(K.R.GAMMA**(n-1)*(float(g)-float(c)-.5*float(t)) if c or g or t else 0)
            assert abs(reward-a['rewards'][k])<1e-12
        rep=next(v for v in s['replays'] if v['arm']==row['arm']);assert rep['all12_arrays_and_integration_exact'] and rep['physics_steps']==row['physics_steps']
        physics+=2*row['physics_steps']
    assert physics==s['physics_steps']<=p['max_physics'] and s['commands']==5
    save(dest,dict(state='PASS',native_added=0,Adam_calls=0,all_prefix_arrays_and_native_handoff_exact=True,
        all_trained_Actor_suffix_actions_exact=True,full_raw_hash_clock_history_verified=True,
        all_original_processes_absent=True,physics_steps=physics,commands=s['commands'],not_original_start_six_success=True))
    print('T334 trained takeover independent fullraw audit PASS')

if __name__=='__main__':globals()[sys.argv[1]]()
