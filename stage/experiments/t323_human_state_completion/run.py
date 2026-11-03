"""Two fixed human-reached40s states followed by frozen safeActor completion.

Not human actions, not autonomous Actor navigation, no learned-policy update.
Original human prefix preserved exactly; original elapsed56s total cap remains.
"""
from pathlib import Path
import importlib.util,json,time,os,signal,traceback,contextlib
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t323_human_state_completion';DOC=ROOT/'stage/docs/distance_20261103'
spec=importlib.util.spec_from_file_location('stage323_recording_primitives',ROOT/'stage/experiments/t317_human_demonstration/recorder.py');R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
K,np,torch=R.K,R.np,R.torch
read=R.read

def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    assert K.S.used(OUT)<=p['final_bytes'] and R.shutil.disk_usage(ROOT).free>=p['min_free_bytes']

def native_action(value):
    a=np.asarray(value,dtype=np.float32)
    assert a.shape==(2,) and np.isfinite(a).all() and (np.abs(a)<1.).all()
    return a

# Human UI intentionally used a narrower bound. This process executes the actual
# existing Actor mode in its original open(-1,1) native domain without clipping.
R.action_vector=native_action

def source_arrays(source):
    a,initial,final=R.load_arrays(source)
    assert a['states'].shape==(280,120) and (a['native_counts']==100).all()
    assert not a['end_flags'][:-1].any() and a['end_flags'][-1].tolist()==[0,0,1]
    return a

def preflight():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists()
    torch.set_num_threads(1);torch.optim.Adam.step=R.U.forbid_step
    q=K.policy(p).eval().requires_grad_(False);before=K.S.digest(q.state_dict());records=[]
    assert before==p['policy_hash'] and not q.memory_on
    for source in p['source_demonstrations']:
        a=source_arrays(source);k=p['prefix_decisions'];assert k==200
        assert K.S.digest(a['states'][0])==p['initial_geometry']['initial120_sha256']
        assert np.array_equal(a['states'][k,-3:-1],a['actions'][k-1])
        assert a['states'][k,-1]==1. and int(a['native_counts'][:k].astype('int64').sum())==20000
        with torch.no_grad():
            mode=q.step(torch.tensor(a['states'][k:k+1]))[0][0].numpy();native_action(mode)
        records.append(dict(source=source,boundary120_sha256=K.S.digest(a['states'][k]),elapsed_physics=20000,remaining_physics=8000,
            original_goal_distance=float(np.linalg.norm(a['poses'][k,:2]-p['initial_geometry']['goal_xy'])),frozen_first_mode=mode.tolist()))
    assert K.S.digest(q.state_dict())==before
    K.C.save(OUT/'preflight.json',dict(state='PASS',physics_steps=0,geometry_commands=0,Adam_calls=0,Actor_updates=0,gradient_calls=0,
        source_real_history_and_native_action_domain_verified=True,policy_hash=before,records=records,
        protocol_sha256=K.C.sha(HERE/'protocol.json'),worst_reserved_global_bytes=K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']))
    print('T323 frozenActor/history/action-domain preflight PASS,0native0Adam')

def main():
    p=read(HERE/'protocol.json');assert not (OUT/'status.json').exists();start=time.time();deadline=start+p['wall_seconds'];meter=None;session=None;witness=None;records=[];replays=[];state='FAILED';phase='STARTING';last=0;current_file=None
    def status(force=False):
        nonlocal last
        if not force and time.time()-last<10:return
        last=time.time();K.C.save(OUT/'status.json',dict(state=phase,pid=os.getpid(),main_epoch=start,deadline_epoch=deadline,updated_epoch=last,
            commands=0 if meter is None else meter.commands,physics_steps=0 if meter is None else meter.physics,derived_trajectories=len(records),exact_replays=len(replays),Actor_updates=0,Adam_calls=0))
    def guard():assert time.time()<deadline-p['final_reserve_seconds'];meter.hold();assert meter.physics<p['max_physics'];status()
    def physical():meter.physics+=1
    def save_current():
        blob,arrays=session.packet();assert len(blob)<=p['episode_bytes'];tmp=current_file.with_suffix('.partial');tmp.write_bytes(blob);os.replace(tmp,current_file);return arrays
    def alarm(*_):raise TimeoutError('T323 fixed two-source augmentation deadline')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(p['wall_seconds']-2)
    try:
        verify(p);assert read(OUT/'preflight.json')['state']=='PASS';torch.set_num_threads(1);torch.optim.Adam.step=R.U.forbid_step
        meter=K.Budget(ROOT,OUT,p,start_epoch=start);meter.charge_payload(p['payload_bytes']);status(True);q=K.policy(p).eval().requires_grad_(False);assert K.S.digest(q.state_dict())==p['policy_hash']
        with (OUT/'native_factory.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            native,create,witness,reward=R.setup(OUT/'runtime')
            for index,source in enumerate(p['source_demonstrations']):
                phase='HUMAN_STATE_COMPLETION_'+str(index+1);original=source_arrays(source);actions=original['actions'];boundary=None;meter.begin(28000)
                session=R.Session(p,native,create,witness,reward,guard,physical);current_file=OUT/('completion_%02d.npz'%(index+1));assert not current_file.exists();save_current();status(True)
                for k in range(280):
                    if k==p['prefix_decisions']:
                        prefix=session.arrays()
                        for key,v in prefix.items():
                            n=k+1 if key.startswith('history_') else k
                            assert np.array_equal(v,original[key][:n]),('prefix mismatch',key)
                        x=K.T.L.packet(session.env,session.obs,session.steps,28000,session.previous,session.pn)
                        assert np.array_equal(x,original['states'][k]) and session.steps==20000
                        boundary=dict(decision=k,physics_steps=session.steps,remaining_physics=28000-session.steps,
                            state120_sha256=K.S.digest(x),integration_sha256=K.R.arrays_hash(K.R.integration(session.env)),
                            real_history_sha256={key:K.S.digest(v) for key,v in prefix.items() if key.startswith('history_')},
                            prefix_arrays_sha256={key:K.S.digest(v) for key,v in prefix.items()},remaining_m=float(session.env._goal_state()[1]))
                        K.C.save(OUT/('boundary_%02d.json'%(index+1)),boundary)
                    if k<p['prefix_decisions']:a=actions[k];behavior='HUMAN_PREFIX_RECONSTRUCTION'
                    else:
                        x=K.T.L.packet(session.env,session.obs,session.steps,28000,session.previous,session.pn)
                        with torch.no_grad():a=q.step(torch.tensor(x[None]))[0][0].numpy()
                        native_action(a);behavior='FROZEN_SAFE_ACTOR_SUFFIX'
                    with (OUT/'behavior_inputs.jsonl').open('a') as f:f.write(json.dumps(dict(source_index=index,decision=k,behavior=behavior,action=a.tolist()),separators=(',',':'))+'\n')
                    session.step(a);save_current()
                    if session.term or session.trunc:break
                assert boundary is not None
                arrays=save_current();initial=session.initial.copy();final=K.R.integration(session.env).copy()
                row=dict(id='completion_%02d'%(index+1),path=str(current_file),sha256=K.C.sha(current_file),source_path=source,source_sha256=K.C.sha(source),
                    behavior_source='HUMAN_PREFIX_FROZEN_ACTOR_SUFFIX_DIAGNOSTIC',boundary=boundary,policy_hash=p['policy_hash'],prefix_human_steps=20000,suffix_actor_steps=session.steps-20000,not_human_demonstration=True,not_learned_Actor_navigation=True,not_onpolicy=True,
                    complete=bool(session.term or session.trunc),success=bool(session.info.get('reach_goal')) and not bool(session.info.get('collision')),
                    collision=bool(session.info.get('collision')),timeout=bool(session.trunc),physics_steps=session.steps,remaining_m=float(session.env._goal_state()[1]),
                    initial120_sha256=K.S.digest(session.initial120),initial_integration_sha256=K.R.arrays_hash(initial),integration_sha256=K.R.arrays_hash(final),
                    scene_sha256=session.scene,goal_xy=session.goal.tolist(),array_hashes={key:K.S.digest(v) for key,v in arrays.items()})
                records.append(row);K.C.save(OUT/'cases.json',records);assert row['complete'];session.close();session=None;current_file=None;status(True)
                phase='EXACT_REPLAY_'+str(index+1);meter.begin(28000);session=R.Session(p,native,create,witness,reward,guard,physical)
                for a,n in zip(arrays['actions'],arrays['native_counts']):session.step(a,int(n))
                actual=session.arrays();assert actual.keys()==arrays.keys() and all(np.array_equal(actual[key],v) for key,v in arrays.items()) and K.R.arrays_hash(K.R.integration(session.env))==row['integration_sha256']
                replays.append(dict(id=row['id'],physics_steps=session.steps,all12_arrays_exact=True,integration_sha256=row['integration_sha256'],not_independent_sample=True))
                K.C.save(OUT/'replays.json',replays);session.close();session=None;meter.checkpoint();status(True)
        state='COMPLETE'
    except BaseException:
        K.C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),phase=phase,commands=0 if meter is None else meter.commands,charged_physics=0 if meter is None else meter.physics))
        if session is not None and current_file is not None:
            try:save_current()
            except BaseException:pass
        elif session is not None:
            # Never overwrite the completed original with an interrupted replay.
            try:
                blob,_=session.packet();f=OUT/('partial_replay_%02d.npz'%len(records));assert not f.exists();f.write_bytes(blob)
            except BaseException:pass
    finally:
        signal.alarm(0)
        if session is not None:session.close()
        if witness is not None:witness.close()
        if meter is not None:meter.close()
    s=dict(state=state,pid=os.getpid(),main_epoch=start,elapsed_s=time.time()-start,deadline_epoch=deadline,commands=0 if meter is None else meter.commands,
        physics_steps=0 if meter is None else meter.physics,derived_trajectories=len(records),exact_replays=len(replays),
        counts={key:sum(bool(r[key]) for r in records) for key in ['success','collision','timeout']},
        all_workers_stopped=True,workers_spawned=0,Actor_updates=0,Adam_calls=0,gradient_calls=0,autonomous_six_metre_pass=False,automatic_training=False,
        own_bytes=K.S.used(OUT),global_bytes=K.S.used(ROOT/'stage/runs'))
    K.C.save(OUT/'summary.json',s);K.C.save(OUT/'status.json',s);print(json.dumps(s),flush=True)

def audit():
    p=read(HERE/'protocol.json');verify(p);s=read(OUT/'summary.json');dest=DOC/'T323_AUDIT.json';assert s['state']=='COMPLETE' and not dest.exists()
    for pid in [s['pid'],read(OUT/'launch.json')['pid']]:
        try:os.kill(pid,0)
        except ProcessLookupError:continue
        raise AssertionError(('Original process still exists',pid))
    start=time.time();rows=read(OUT/'cases.json');reps=read(OUT/'replays.json');assert len(rows)==len(reps)==2 and s['commands']==4
    inputs=[json.loads(v) for v in (OUT/'behavior_inputs.jsonl').read_text().splitlines()];torch.set_num_threads(1);q=K.policy(p).eval().requires_grad_(False)
    spec=importlib.util.spec_from_file_location('stage323_independent_reward',ROOT/'stage/experiments/t295_hindsight_support/run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    for r,rep,source in zip(rows,reps,p['source_demonstrations']):
        assert K.C.sha(r['path'])==r['sha256'];a,initial,final=R.load_arrays(r['path']);original=source_arrays(source);boundary=p['prefix_decisions'];expected=original['actions'][:boundary]
        assert np.array_equal(a['actions'][:boundary],expected) and {k:K.S.digest(v) for k,v in a.items()}==r['array_hashes']
        written=[v['action'] for v in inputs if v['source_index']==p['source_demonstrations'].index(source)]
        assert np.array_equal(np.asarray(written,np.float32),a['actions'])
        assert r['prefix_human_steps']==20000 and r['suffix_actor_steps']==r['physics_steps']-20000
        for key,v in original.items():
            n=boundary+1 if key.startswith('history_') else boundary
            assert np.array_equal(a[key][:n],v[:n]),key
        assert K.S.digest(a['states'][boundary])==r['boundary']['state120_sha256'] and np.array_equal(a['states'][boundary],original['states'][boundary])
        with torch.no_grad():
            for x,action in zip(a['states'][boundary:],a['actions'][boundary:]):assert np.array_equal(q.step(torch.tensor(x[None]))[0][0].numpy(),action)
        assert K.R.arrays_hash(initial)==p['initial_geometry']['integration_sha256'] and K.R.arrays_hash(final)==r['integration_sha256']
        assert K.S.digest(a['states'][0])==p['initial_geometry']['initial120_sha256'] and np.array_equal(a['states'][1:],a['next_states'][:-1])
        assert int(a['native_counts'].astype('int64').sum())==r['physics_steps'] and not a['end_flags'][:-1].any()
        for k,x in enumerate(a['states']):
            assert time.time()<start+p['audit_seconds']
            assert np.abs(K.H.rebuild(x,a['poses'][k],a['history_poses'][k],r['goal_xy'],a['history_goals'][k])-x).max()<=2e-6
            d0=K.H.fields(a['poses'][k],r['goal_xy'])[1];d1=K.H.fields(a['next_poses'][k],r['goal_xy'])[1]
            assert abs(m.reward(d0,d1,int(a['native_counts'][k]),*map(bool,a['end_flags'][k]))-a['rewards'][k])<1e-12
        assert rep['id']==r['id'] and rep['physics_steps']==r['physics_steps'] and rep['all12_arrays_exact'] and rep['integration_sha256']==r['integration_sha256']
    assert sum(r['physics_steps'] for r in rows)+sum(r['physics_steps'] for r in reps)==s['physics_steps']<=112000
    K.C.save(dest,dict(state='PASS',native_added=0,Actor_updates=0,all_full120_history_reward_prefix_and_frozen_Actor_suffix_verified=True,
        existing_full12_native_replay_receipts_verified=True,commands=4,physics_steps=s['physics_steps'],derived_safe_GOAL_ids=[r['id'] for r in rows if r['success'] and not r['collision']],
        not_human_or_autonomous_Actor_success=True,all_processes_absent=True,policy_hash=K.S.digest(q.state_dict()),summary_sha256=K.C.sha(OUT/'summary.json')))
    print('T323 full raw/prefix/frozenActor suffix audit PASS,0newnative0Actor')

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['preflight','main','audit']);globals()[a.parse_args().mode]()
