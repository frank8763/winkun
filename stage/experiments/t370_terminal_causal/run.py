"""Fixed terminal steering probes, exact whole-prefix simulation, zero optimization."""
from pathlib import Path
import sys,ast,contextlib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t369_future_action_supervision'))
from t369_core import *
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t370_terminal_causal';DOC=ROOT/'stage/docs/narrow_20261105'
BASE_NAMES=['states','actions','native_counts','next_states','poses','next_poses','end_flags','rewards','history_poses','history_goals','history_times','environment_rewards']


def forbid(*_a,**_k):raise AssertionError('T370 forbids optimizer updates')


def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes'] and K.S.used(OUT)<=p['final_bytes']
    assert shutil.disk_usage(ROOT).free>=p['min_free_bytes']


def actor(p):
    q=NightActor(p).eval().requires_grad_(False);q.load_learned(read_model(p['frozen_model']['path']));assert q.phash()==p['frozen_model']['policy_hash'];return q


def prepare():
    assert not OUT.exists() and read(DOC/'T369_CLOSEOUT.json')['state']=='PASS_LOCAL_CLOSURE_WITH_TRANSPORT_RECOVERY'
    assert read(ROOT/'stage/runs/t369_future_action_supervision/audit.json')['state']=='PASS'
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T369.json';assert read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p=copy.deepcopy(read(ROOT/'stage/runs/t369_future_action_supervision/protocol.json'));source=next(r for r in read(ROOT/'stage/runs/t369_future_action_supervision/cases.json') if r['role']=='final_future' and r['job_id']=='narrow070_dev_0');a=arrays(source['path'])
    n=int(np.flatnonzero(np.linalg.norm(a['poses'][:,:2]-np.asarray(source['goal_xy']),axis=1)<=1.)[0]);assert n==248 and np.all(a['native_counts'][:n]==100) and source['timeout'] and source['complete']
    p.update(id='T370',folder=OUT.name,out=str(OUT),experiment='T370_FIXED_TERMINAL_CAUSAL_STEERING',source=source,frozen_model=read(ROOT/'stage/runs/t369_future_action_supervision/summary.json')['models']['future'],
        job=copy.deepcopy(p['narrow_jobs'][0]),prefix_decisions=n,prefix_physics=24800,intervention_decisions=5,conditions=[['actor',None],['minus',-1.],['zero',0.],['plus',1.]],
        max_commands=9,max_physics=320000,wall_seconds=900,final_reserve_seconds=120,audit_seconds=300,all_work_seconds=1800,
        final_bytes=4*1024**2,episode_bytes=512*1024,partial_bytes=512*1024,geometry_commands=1,Actor_updates=0,optimizer_calls=0,contract=(DOC/'T370_PLAN.md').read_text())
    files=set(p['input_hashes'])|{str(HERE/'run.py'),str(DOC/'T370_PLAN.md'),str(DOC/'T369_CLOSEOUT.json'),str(DOC/'T369_FINDINGS.md'),str(receipt),source['path'],p['frozen_model']['path'],str(ROOT/'stage/experiments/t260_distance_archive/backup.py')}
    p['input_hashes']={f:K.C.sha(f) for f in sorted(files)};used=K.S.used(ROOT/'stage/runs');assert used+p['final_bytes']<p['global_bytes']
    OUT.mkdir();(OUT/'episodes').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T370_BUDGET.json',dict(state='FULL_RESERVATION_PASS',global_before=used,own_max=p['final_bytes'],full_global_reservation=used+p['final_bytes'],global_limit=p['global_bytes'],max_commands=9,max_physics=320000,Adam=0))
    print(json.dumps(dict(state='REGISTERED',native=0,Adam=0,full_reservation=used+p['final_bytes'])))


def preflight():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.optim.Adam.step=forbid
    ast.parse((HERE/'run.py').read_text());a=arrays(p['source']['path']);q=actor(p);h=None
    with torch.no_grad():
        for k,x in enumerate(a['states']):
            bh=np.zeros(16) if h is None else h[0,0].numpy();assert np.array_equal(bh,a['hidden_before'][k]);act,_,h,_=q.step(torch.tensor(x[None]),h);assert np.array_equal(act[0].numpy(),a['actions'][k])
            for name,v in q.tape[-1].items():assert np.array_equal(v,a[name][k])
    folder=OUT/'preflight';folder.mkdir()
    with (folder/'factory.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        m,native,create,w,reward=P.C.B.setup(folder/'native',p);m.ACTIVE['circle_shift_x_m']=p['job']['circle_shift_x_m'];env,obs,_=K.R.task_env(create,p['job'])
        try:
            obs=K.C.perturb_anchor(env,obs,p['job']);g=P.C.B.geometry(env,obs,w,40000);assert g['valid'] and np.array_equal(np.asarray(g['full120'],np.float32),a['states'][0]) and np.array_equal(K.R.integration(env),a['initial_native_integration'])
            assert env.max_wheel==5 and env.data.time==0 and env.step_count==0 and np.array_equal(env.model.actuator_forcerange[[env.id_lr_wheel,env.id_rr_wheel]],[[-120.,120.],[-120.,120.]])
            save(OUT/'geometry.json',g)
        finally:w.events.pop(id(env),None);env.close();w.close()
    save(OUT/'preflight.json',dict(state='PASS',geometry_commands=1,native=0,Adam=0,full400_source_Actor_hidden_actions_exact=True,source_prefix248_verified=True,source_model_SHA_verified=True,geometry_sha256=K.C.sha(OUT/'geometry.json')))
    print(json.dumps(dict(state='PREFLIGHT_PASS',native=0,geometry=1,Adam=0)))


def prefix_exact(a,source,n):
    for name in BASE_NAMES+['hidden_before','innovations','raw_actions','locations','logp','future_locations']:
        assert np.array_equal(a[name][:n+1 if name.startswith('history_') else n],source[name][:n+1 if name.startswith('history_') else n]),name


def collect(p,s,q,source,condition,guard):
    n=p['prefix_decisions'];h=None;hb=[];proposed=[];mask=[];handoff=None;min_native=float(s.env._goal_state()[1]);min_step=0
    # Session's physical callback records continuous-native distance as well.
    original_count=s.physical
    def count():
        nonlocal min_native,min_step
        original_count();dist=float(s.env._goal_state()[1])
        if dist<min_native:min_native=dist;min_step=s.steps
    s.physical=count;q.reset_noise();k=0
    while not(s.term or s.trunc):
        guard();x=K.T.L.packet(s.env,s.obs,s.steps,40000,s.previous,s.pn)
        if k==n:
            assert s.steps==24800 and np.array_equal(x,source['states'][n]);handoff=K.R.integration(s.env).copy();assert np.array_equal(h[0,0].numpy(),source['hidden_before'][n])
        hb.append(np.zeros(16) if h is None else h[0,0].numpy().copy())
        with torch.no_grad():pred,_,h,_=q.step(torch.tensor(x[None]),h)
        pred=pred[0].numpy();a=pred.copy();forced=condition is not None and n<=k<n+5
        if forced:a[1]=condition
        if k<n:assert np.array_equal(a,source['actions'][k])
        proposed.append(pred.copy());mask.append(forced);s.step(a,100);k+=1
    a=s.arrays();a.update(hidden_before=np.asarray(hb),last_hidden_after=h[0,0].numpy().copy(),initial_native_integration=s.initial.copy(),final_native_integration=K.R.integration(s.env).copy())
    a.update({name:np.asarray([v[name] for v in q.tape]) for name in q.tape[0]});prefix_exact(a,source,n)
    if condition is None:
        for name in source:assert np.array_equal(a[name],source[name]),name
    assert handoff is not None;a.update(proposed_actions=np.asarray(proposed),override_mask=np.asarray(mask,np.uint8),handoff_native_integration=handoff)
    return a,dict(min_native_goal_distance_m=min_native,min_native_goal_step=min_step,min_native_goal_seconds=min_step*.002)


def main():
    p=read(HERE/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'status.json').exists();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.optim.Adam.step=forbid
    start=time.time();deadline=read(OUT/'launch.json')['deadline_epoch'];physics=0;commands=1;phase='STARTING';last=0.;rows=[];reps=[];state='FAILED';s=w=None
    meter=P.C.ReservedBudget(ROOT/'stage/runs',OUT,p['final_bytes'],p['global_bytes'],p['min_free_bytes'])
    def status(force=False):
        nonlocal last
        if not force and time.time()-last<3:return
        last=time.time();meter.check();save(OUT/'status.json',dict(state=phase,pid=os.getpid(),main_epoch=start,updated_epoch=last,deadline_epoch=deadline,commands=commands,physics_steps=physics,Adam=0,cases=len(rows),replays=len(reps)))
    def guard():assert time.time()<deadline-10 and physics<p['max_physics'];status();meter.check()
    def count():
        nonlocal physics
        physics+=1
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T370 native deadline')));signal.alarm(max(1,int(deadline-time.time())-3))
    try:
        source=arrays(p['source']['path']);q=actor(p);geo=read(OUT/'geometry.json')
        with (OUT/'factory.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            m,native,create,w,reward=P.C.B.setup(OUT/'native',p);m.ACTIVE['circle_shift_x_m']=p['job']['circle_shift_x_m'];handoff=None
            for name,condition in p['conditions']:
                ref=None
                for replica in ['primary','exact_native_replay']:
                    assert commands<9 and physics+40000<=p['max_physics'] and time.time()+60<deadline;commands+=1;phase=name+'_'+replica;status(True)
                    cp=copy.deepcopy(read(p['course_protocol']));cp.update(job=p['job'],initial_geometry=geo);s=m.R.Session(cp,native,create,w,reward,guard,count)
                    a,minimum=collect(p,s,q,source,condition,guard)
                    if handoff is None:handoff=a['handoff_native_integration']
                    else:assert np.array_equal(handoff,a['handoff_native_integration'])
                    row=dict(id=name,condition_steer=condition,prefix_decisions=248,intervention_decisions=5 if condition is not None else 0,physics_steps=s.steps,complete=bool(s.term or s.trunc),success=bool(s.info.get('reach_goal')) and not bool(s.info.get('collision')),collision=bool(s.info.get('collision')),timeout=bool(s.trunc),administrative_censor=not bool(s.term or s.trunc),remaining_m=float(s.env._goal_state()[1]),goal_xy=s.goal.tolist(),scene_sha256=s.scene,integration_sha256=K.R.arrays_hash(a['final_native_integration']),handoff_sha256=digest(handoff),array_hashes={k:digest(v) for k,v in a.items()},diagnostic_forced_actions=condition is not None,not_onpolicy_training=True,not_autonomous_six_acceptance=True,**minimum)
                    if replica=='primary':
                        b=io.BytesIO();np.savez_compressed(b,**packed(a));blob=b.getvalue();assert len(blob)<=p['episode_bytes'];path=OUT/'episodes'/(name+'.npz');assert not path.exists();path.write_bytes(blob);row.update(path=str(path),sha256=K.C.sha(path));rows.append(row);save(OUT/'cases.json',rows);ref=dict(row=row,arrays=a)
                    else:
                        assert row['array_hashes']==ref['row']['array_hashes'] and row['min_native_goal_distance_m']==ref['row']['min_native_goal_distance_m'] and all(np.array_equal(a[k],ref['arrays'][k]) for k in a)
                        row.update(all_arrays_and_native_integration_exact=True,independent_sample=False);reps.append(row);save(OUT/'replays.json',reps)
                    s.close();s=None;status(True)
            state='COMPLETE'
    except BaseException:
        save(OUT/'failure.json',dict(error=traceback.format_exc(),phase=phase,commands=commands,physics_steps=physics))
        if s is not None:
            try:
                a=s.arrays();a['final_native_integration']=K.R.integration(s.env).copy();b=io.BytesIO();np.savez_compressed(b,**packed(a));blob=b.getvalue();assert len(blob)<=p['partial_bytes'];(OUT/'partial.npz').write_bytes(blob)
            except BaseException:save(OUT/'partial_failure.json',dict(error=traceback.format_exc()))
    finally:
        if s is not None:s.close()
        if w is not None:w.close()
        signal.alarm(0)
    result=dict(state=state,pid=os.getpid(),main_epoch=start,elapsed_s=time.time()-start,commands=commands,physics_steps=physics,cases=len(rows),replays=len(reps),Actor_updates=0,Adam=0,new_onpolicy_training=0,all_native_sessions_closed=True,workers_spawned=0,old275=None,six54=None,fresh16=None,reload78=None,stable_six=False,entered7or8=False)
    save(OUT/'summary.json',result);save(OUT/'status.json',result);meter.check(force=True);meter.close();print(json.dumps(result),flush=True)


def audit():
    p=read(HERE/'protocol.json');verify(p);result=read(OUT/'summary.json');assert result['state']=='COMPLETE' and not (OUT/'audit.json').exists();assert not Path('/proc',str(result['pid'])).exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.optim.Adam.step=forbid;source=arrays(p['source']['path']);rows=read(OUT/'cases.json');reps=read(OUT/'replays.json');assert len(rows)==len(reps)==4;physics=0;handoff=None
    for r in rows:
        assert K.C.sha(r['path'])==r['sha256'];a=arrays(r['path']);assert {k:digest(v) for k,v in a.items()}==r['array_hashes'];prefix_exact(a,source,248)
        assert int(a['native_counts'].astype(int).sum())==r['physics_steps'] and not a['end_flags'][:-1].any() and a['end_flags'][-1].any();assert np.array_equal(a['states'][1:],a['next_states'][:-1])
        if handoff is None:handoff=a['handoff_native_integration']
        else:assert np.array_equal(handoff,a['handoff_native_integration'])
        assert digest(handoff)==r['handoff_sha256'];q=actor(p);q.reset_noise();h=None
        with torch.no_grad():
            for k,x in enumerate(a['states']):
                bh=np.zeros(16) if h is None else h[0,0].numpy();assert np.array_equal(bh,a['hidden_before'][k]);pred,_,h,_=q.step(torch.tensor(x[None]),h);action=pred[0].numpy().copy();assert np.array_equal(action,a['proposed_actions'][k])
                forced=r['condition_steer'] is not None and 248<=k<253
                if forced:action[1]=r['condition_steer']
                assert forced==bool(a['override_mask'][k]) and np.array_equal(action,a['actions'][k])
                for name,v in q.tape[-1].items():assert np.array_equal(v,a[name][k])
                assert np.abs(K.H.rebuild(x,a['poses'][k],a['history_poses'][k],r['goal_xy'],a['history_goals'][k])-x).max()<=2e-6
                nn=int(a['native_counts'][k]);c,g,t=map(bool,a['end_flags'][k]);z=K.R.GAMMA**nn;d0=K.H.fields(a['poses'][k],r['goal_xy'])[1];d1=K.H.fields(a['next_poses'][k],r['goal_xy'])[1];rw=-.000002*(1-z)/(1-K.R.GAMMA)+.03*d0-z*.03*d1*(not(c or g or t))
                if c or g or t:rw+=K.R.GAMMA**(nn-1)*(float(g)-float(c)-.5*float(t))
                assert abs(rw-a['rewards'][k])<1e-12
        assert np.array_equal(h[0,0].numpy(),a['last_hidden_after']) and K.R.arrays_hash(a['final_native_integration'])==r['integration_sha256']
        if r['condition_steer'] is None:
            for key in source:assert np.array_equal(source[key],a[key])
        rep=next(x for x in reps if x['id']==r['id']);assert rep['array_hashes']==r['array_hashes'] and rep['all_arrays_and_native_integration_exact'];physics+=2*r['physics_steps']
        c,g,t=map(bool,a['end_flags'][-1]);assert c==r['collision'] and (g and not c)==r['success'] and t==r['timeout']
    assert physics==result['physics_steps']<=320000 and result['commands']==9 and result['Adam']==0 and result['elapsed_s']<=900
    s=dict(state='PASS',new_native=0,new_Adam=0,full120_hidden_history_prefix_actions_reward_model_SHA_exact=True,baseline_full_T369_source_exact=True,same_handoff_all_conditions=True,all_native_replays_exact=True,continuous_minimum_reproduced_by_native_repeat=True,commands=9,physics_steps=physics,diagnostic_safe_goal_ids=[r['id'] for r in rows if r['success'] and r['condition_steer'] is not None],not_autonomous_six_acceptance=True)
    save(OUT/'audit.json',s);save(DOC/'T370_AUDIT.json',s)


def launch():
    p=read(HERE/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    r=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert r['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();epoch=time.time()
    with (DOC/'T370_DRIVER.log').open('xb') as log:pr=subprocess.Popen(['systemd-inhibit','--what=sleep','--mode=block','--why=Stage fixed terminal causal probe','timeout','--signal=TERM','--kill-after=1s','1800s','taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'driver'],cwd=ROOT,env=dict(os.environ,STAGE_PROCESS_CPU='0'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=pr.pid,launch_epoch=epoch,deadline_epoch=epoch+900,all_work_deadline_epoch=epoch+1800,source_remote_commit=r['remote_commit'],root_tree=r['root_tree'],protocol_sha256=K.C.sha(HERE/'protocol.json')));print(json.dumps(dict(state='LAUNCHED',pid=pr.pid)))


def driver():
    p=read(HERE/'protocol.json');verify(p);l=read(OUT/'launch.json');active=None
    def call(args,seconds,log):
        nonlocal active
        with log.open('xb') as f:
            active=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),*args],cwd=ROOT,env=dict(os.environ,STAGE_PROCESS_CPU='0'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                rc=active.wait(timeout=seconds)
                if rc:raise subprocess.CalledProcessError(rc,str(args))
            except BaseException:
                try:os.killpg(active.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                if active.poll() is None:
                    try:active.wait(1)
                    except subprocess.TimeoutExpired:pass
                try:os.killpg(active.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                active.wait();raise
            finally:active=None
    def stop(*_):raise TimeoutError('T370 bounded finalizer interrupted')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int(l['all_work_deadline_epoch']-time.time())-3))
    try:
        save(DOC/'T370_DRIVER.json',dict(pid=os.getpid(),state='NATIVE'));call([str(HERE/'run.py'),'main'],min(900,l['deadline_epoch']-time.time()),OUT/'main.log');s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists()
        save(DOC/'T370_DRIVER.json',dict(pid=os.getpid(),state='AUDIT'));call([str(HERE/'run.py'),'audit'],min(300,l['all_work_deadline_epoch']-time.time()-30),OUT/'audit_console.log');assert read(OUT/'audit.json')['state']=='PASS'
        save(DOC/'T370_DRIVER.json',dict(pid=os.getpid(),state='RAW_BACKUP'));call([str(ROOT/'stage/experiments/t260_distance_archive/small_parts.py'),'T370'],min(600,l['all_work_deadline_epoch']-time.time()-20),DOC/'T370_BACKUP.log');receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T370.json';assert read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
        save(DOC/'T370_DRIVER.json',dict(pid=os.getpid(),state='COMPLETE_AUDIT_AND_RAW_BACKUP',summary_sha256=K.C.sha(OUT/'summary.json'),receipt_sha256=K.C.sha(receipt)))
    except BaseException:save(DOC/'T370_DRIVER_ERROR.json',dict(error=traceback.format_exc(),navigation_restarted=False));raise
    finally:signal.alarm(0)


if __name__=='__main__':globals()[sys.argv[1]]()
