"""Two human-derived, open-loop throttle augmentations for training data only.

Not human actions, not autonomous Actor navigation, no learned-policy update.
Original human steering and decision timing remain; drive alone becomes .999.
"""
from pathlib import Path
import importlib.util,json,time,os,signal,traceback,contextlib
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t320_human_throttle_augmentation';DOC=ROOT/'stage/docs/distance_20261103'
spec=importlib.util.spec_from_file_location('stage320_recording_primitives',ROOT/'stage/experiments/t317_human_demonstration/recorder.py');R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
K,np,torch=R.K,R.np,R.torch
read=R.read

def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    assert K.S.used(OUT)<=p['final_bytes'] and R.shutil.disk_usage(ROOT).free>=p['min_free_bytes']

def transformed(source):
    with np.load(source) as z:
        a=z['actions'].copy();n=z['native_counts'].copy();initial=z['states'][0].copy()
    assert a.shape==(280,2) and (n==100).all()
    result=a.copy();result[:,0]=np.float32(.999)
    assert np.array_equal(result[:,1],a[:,1]) and np.isfinite(result).all()
    return result,initial

def preflight():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists();torch.optim.Adam.step=R.U.forbid_step
    counts=[]
    for source in p['source_demonstrations']:
        a,initial=transformed(source);assert K.S.digest(initial)==p['initial_geometry']['initial120_sha256']
        for v in a:R.action_vector(v)
        counts.append(dict(source=source,actions=280,action_sha256=K.S.digest(a)))
    K.C.save(OUT/'preflight.json',dict(state='PASS',geometry_commands=0,physics_steps=0,Actor_updates=0,Adam_calls=0,gradient_calls=0,
        original_LEFT_initial_geometry_reused=True,only_drive_changed_steering_and_hold_count_exact=True,transformations=counts,
        protocol_sha256=K.C.sha(HERE/'protocol.json'),worst_reserved_global_bytes=K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']))
    print('T320 transformation contract PASS,0native0Actor0Adam')

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
    def alarm(*_):raise TimeoutError('T320 fixed two-source augmentation deadline')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(p['wall_seconds']-2)
    try:
        verify(p);assert read(OUT/'preflight.json')['state']=='PASS';torch.set_num_threads(1);torch.optim.Adam.step=R.U.forbid_step
        meter=K.Budget(ROOT,OUT,p,start_epoch=start);meter.charge_payload(p['payload_bytes']);status(True)
        with (OUT/'native_factory.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            native,create,witness,reward=R.setup(OUT/'runtime')
            for index,source in enumerate(p['source_demonstrations']):
                phase='AUGMENTATION_'+str(index+1);actions,_=transformed(source);meter.begin(28000)
                session=R.Session(p,native,create,witness,reward,guard,physical);current_file=OUT/('augmentation_%02d.npz'%(index+1));assert not current_file.exists();save_current();status(True)
                for k,a in enumerate(actions):
                    with (OUT/'augmented_action_inputs.jsonl').open('a') as f:f.write(json.dumps(dict(source_index=index,decision=k,action=a.tolist()),separators=(',',':'))+'\n')
                    session.step(a);save_current()
                    if session.term or session.trunc:break
                arrays=save_current();initial=session.initial.copy();final=K.R.integration(session.env).copy()
                row=dict(id='augmentation_%02d'%(index+1),path=str(current_file),sha256=K.C.sha(current_file),source_path=source,source_sha256=K.C.sha(source),
                    behavior_source='HUMAN_DERIVED_OPEN_LOOP_THROTTLE_AUGMENTATION',not_human_demonstration=True,not_learned_Actor_navigation=True,not_onpolicy=True,
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
    p=read(HERE/'protocol.json');verify(p);s=read(OUT/'summary.json');dest=DOC/'T320_AUDIT.json';assert s['state']=='COMPLETE' and not dest.exists()
    for pid in [s['pid'],read(OUT/'launch.json')['pid']]:
        try:os.kill(pid,0)
        except ProcessLookupError:continue
        raise AssertionError(('Original process still exists',pid))
    start=time.time();rows=read(OUT/'cases.json');reps=read(OUT/'replays.json');assert len(rows)==len(reps)==2 and s['commands']==4
    inputs=[json.loads(v) for v in (OUT/'augmented_action_inputs.jsonl').read_text().splitlines()]
    spec=importlib.util.spec_from_file_location('stage320_independent_reward',ROOT/'stage/experiments/t295_hindsight_support/run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    for r,rep,source in zip(rows,reps,p['source_demonstrations']):
        assert K.C.sha(r['path'])==r['sha256'];a,initial,final=R.load_arrays(r['path']);expected,_=transformed(source)
        assert np.array_equal(a['actions'],expected[:len(a['actions'])]) and {k:K.S.digest(v) for k,v in a.items()}==r['array_hashes']
        written=[v['action'] for v in inputs if v['source_index']==p['source_demonstrations'].index(source)]
        assert np.array_equal(np.asarray(written,np.float32),a['actions'])
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
    K.C.save(dest,dict(state='PASS',native_added=0,Actor_updates=0,all_full120_history_actions_reward_hashes_and_source_transformation_verified=True,
        existing_full12_native_replay_receipts_verified=True,commands=4,physics_steps=s['physics_steps'],derived_safe_GOAL_ids=[r['id'] for r in rows if r['success'] and not r['collision']],
        not_human_or_autonomous_Actor_success=True,all_processes_absent=True,summary_sha256=K.C.sha(OUT/'summary.json')))
    print('T320 full raw/source transformation audit PASS,0newnative0Actor')

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['preflight','main','audit']);globals()[a.parse_args().mode]()
