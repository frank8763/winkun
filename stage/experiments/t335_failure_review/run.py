"""Bounded posthoc frozen-contact restoration and feedback decomposition.

No mj_step, policy rollout, optimizer, planning or model selection.
"""
from pathlib import Path
import sys,signal,contextlib
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stage/experiments/t334_trained_takeover'))
import run as H
L,K,np,torch,M=H.L,H.K,H.np,H.torch,H.M
read,save=L.read,L.save
OUT=ROOT/'stage/runs/t335_failure_review';DOC=ROOT/'stage/docs/distance_20261103'

def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    assert K.S.used(OUT)<=p['final_bytes'] and L.shutil.disk_usage(ROOT).free>=p['min_free_bytes']

def prepare():
    assert not OUT.exists();source=ROOT/'stage/experiments/t334_trained_takeover/protocol.json'
    p=read(source);receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T334.json'
    assert read(DOC/'T334_AUDIT.json')['state']=='PASS'
    assert read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p.update(experiment='T335_READ_ONLY_FAILURE_REVIEW',source_protocol=str(source),
        cases=str(ROOT/'stage/runs/t334_trained_takeover/cases.json'),final_bytes=1024**2,
        payload_bytes=0,metadata_bytes=896*1024,wall_seconds=180,audit_seconds=180,
        max_commands=2,max_physics=0,total_main_audit_geometry_commands=4)
    files=set(p['input_hashes'])|{str(f) for f in HERE.glob('*.py')}
    files|={str(source),p['cases'],str(receipt),str(DOC/'T334_AUDIT.json'),str(DOC/'T335_PLAN.md')}
    for r in read(p['cases']):files.add(r['path'])
    p['input_hashes']={f:K.C.sha(f) for f in sorted(files)}
    assert K.S.used(ROOT/'stage/runs')+p['final_bytes']<=p['global_bytes']
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)

def preflight():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists()
    ref,_,_=M.R.load_arrays(p['reference'])
    for r in read(p['cases']):
        a,initial,final=M.R.load_arrays(r['path'])
        assert len(initial)==len(final)==203 and K.R.arrays_hash(final)==r['integration_sha256']
        assert np.array_equal(a['states'][:150],ref['states'][:150])
    save(OUT/'preflight.json',dict(state='PASS',geometry_commands=0,physics_steps=0,Actor_updates=0,Adam_calls=0,
        final_state_sizes_hashes_and_prefix_verified=True,protocol_sha256=K.C.sha(HERE/'protocol.json')))

def contact_review(p,folder):
    folder.mkdir();rows=[];witness=None;geometry=[]
    M.R.mujoco.mj_step=lambda *_args,**_kwargs:(_ for _ in ()).throw(AssertionError('T335 forbids mj_step'))
    with (folder/'factory.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        course,native,create,witness,reward=H.setup(p,folder/'native')
        try:
            for r in read(p['cases']):
                session=None
                try:
                    session=M.R.Session(course,native,create,witness,reward,lambda:None,lambda:(_ for _ in ()).throw(AssertionError('No native physical steps')))
                    geometry.append(dict(arm=r['arm'],physics_steps=0));save(folder/'geometry_calls.json',geometry)
                    env=session.env;_,initial,final=M.R.load_arrays(r['path']);mj=M.R.mujoco
                    # Recompute contacts at saved qpos, then preserve every integration
                    # field. No simulation/kinematic/controller step is performed.
                    mj.mj_setState(env.model,env.data,final,mj.mjtState.mjSTATE_INTEGRATION)
                    mj.mj_forward(env.model,env.data)
                    mj.mj_setState(env.model,env.data,final,mj.mjtState.mjSTATE_INTEGRATION)
                    assert np.array_equal(K.R.integration(env),final)
                    assert K.C.model_hash(env)==course['initial_geometry']['scene_sha256']
                    contacts=[]
                    for i in range(int(env.data.ncon)):
                        c=env.data.contact[i];g1,g2=int(c.geom1),int(c.geom2);b1,b2=int(env.model.geom_bodyid[g1]),int(env.model.geom_bodyid[g2])
                        robot1,robot2=b1 in env.robot_body_ids,b2 in env.robot_body_ids
                        if robot1==robot2:continue
                        rg,og,ob=(g1,g2,b2) if robot1 else (g2,g1,b1)
                        name=mj.mj_id2name(env.model,mj.mjtObj.mjOBJ_GEOM,og) or ''
                        body=mj.mj_id2name(env.model,mj.mjtObj.mjOBJ_BODY,ob) or ''
                        floor=any(t in (name+' '+body).lower() for t in ['floor','ground','plane'])
                        if floor:continue
                        contacts.append(dict(robot_geom=mj.mj_id2name(env.model,mj.mjtObj.mjOBJ_GEOM,rg),
                            other_geom=name,other_body=body,distance_m=float(c.dist),
                            triggers_native_collision=float(c.dist)<=.002,contact_position=c.pos.tolist()))
                    assert env._collision() and any(c['triggers_native_collision'] for c in contacts)
                    rows.append(dict(arm=r['arm'],contacts=contacts,scene_sha256=K.C.model_hash(env),
                        final_integration_sha256=K.R.arrays_hash(K.R.integration(env)),
                        pose=list(map(float,env._pose()[:3])),geometry_commands=1,physics_steps=0))
                finally:
                    if session is not None:session.close()
        finally:
            if witness is not None:witness.close()
    return rows

def feedback_review(p):
    ref,_,_=M.R.load_arrays(p['reference']);result={}
    for r in read(p['cases']):
        a,_,_=M.R.load_arrays(r['path']);q=H.policy(p,r['arm']);n=len(a['states'])
        with torch.no_grad():
            same=L.mode(q,torch.tensor(ref['states'][150:n])).numpy()
            own=L.mode(q,torch.tensor(a['states'][150:])).numpy()
        assert np.array_equal(own,a['actions'][150:])
        human=ref['actions'][150:n];static=same-human;feedback=own-same;actual=own-human
        assert np.max(np.abs(actual.astype(np.float64)-(static.astype(np.float64)+feedback.astype(np.float64))))<2e-7
        positions=np.linalg.norm(a['poses'][150:,:2]-ref['poses'][150:n,:2],axis=1)
        def first(threshold):
            ix=np.flatnonzero(positions>threshold)
            return None if not len(ix) else dict(after_handoff_s=float(ix[0]*.2),recorded_decision=int(ix[0]+150),position_error_m=float(positions[ix[0]]))
        result[r['arm']]=dict(aligned_decision_starts=n-150,static_imitation_rms=np.sqrt((static.astype(np.float64)**2).mean(0)).tolist(),
            state_feedback_rms=np.sqrt((feedback.astype(np.float64)**2).mean(0)).tolist(),
            total_action_difference_rms=np.sqrt((actual.astype(np.float64)**2).mean(0)).tolist(),
            first_recorded_position_error_over_1cm=first(.01),first_recorded_position_error_over_5cm=first(.05),
            max_recorded_position_error_m=float(positions.max()),algebraic_decomposition_not_physical_causality=True)
    return result

def compute(p,folder):
    return dict(contacts=contact_review(p,folder),feedback=feedback_review(p),geometry_commands=2,
        physics_steps=0,Actor_updates=0,Adam_calls=0,gradient_calls=0,posthoc=True,
        no_new_navigation_trials=True,not_independent_success_or_universal_cause_proof=True)

def main():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'status.json').exists();assert read(OUT/'preflight.json')['state']=='PASS'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.optim.Adam.step=M.R.U.forbid_step
    start=L.time.time();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T335180s')));signal.alarm(180)
    save(OUT/'status.json',dict(state='FROZEN_POSTHOC',pid=L.os.getpid(),main_epoch=start,physics_steps=0))
    try:
        value=compute(p,OUT/'main');save(OUT/'results.json',value);state='COMPLETE'
    except BaseException:
        save(OUT/'failure.json',dict(error=L.traceback.format_exc(),physics_steps=0));state='FAILED'
    finally:signal.alarm(0)
    summary=dict(state=state,pid=L.os.getpid(),main_epoch=start,elapsed_s=L.time.time()-start,
        deadline_epoch=start+180,commands=len(read(OUT/'main/geometry_calls.json')) if (OUT/'main/geometry_calls.json').exists() else 0,geometry_commands=len(read(OUT/'main/geometry_calls.json')) if (OUT/'main/geometry_calls.json').exists() else 0,physics_steps=0,Actor_updates=0,Adam_calls=0,
        all_workers_stopped=True,workers_spawned=0,all_native_environments_closed=True,
        own_bytes=K.S.used(OUT),global_bytes=K.S.used(ROOT/'stage/runs'))
    save(OUT/'summary.json',summary);save(OUT/'status.json',summary);print(L.json.dumps(summary))

def audit():
    p=read(HERE/'protocol.json');verify(p);s=read(OUT/'summary.json');dest=DOC/'T335_AUDIT.json'
    assert s['state']=='COMPLETE' and not dest.exists()
    for pid in [s['pid'],read(OUT/'launch.json')['pid']]:assert not Path('/proc/'+str(pid)).exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.optim.Adam.step=M.R.U.forbid_step
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T335audit180s')));signal.alarm(180)
    value=compute(p,OUT/'audit');assert value==read(OUT/'results.json');signal.alarm(0)
    save(dest,dict(state='PASS',independent_geometry_commands=2,total_main_audit_geometry_commands=4,
        native_physics_steps=0,optimizer_calls=0,all_contact_names_distances_and_feedback_metrics_exact=True,
        all_input_final_integration_bytes_unchanged=True,all_original_processes_absent=True))
    print('T335 two frozen geometry and feedback independent audit PASS0physics0Adam')

if __name__=='__main__':globals()[sys.argv[1]]()
