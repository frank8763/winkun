"""Frozen Stage policy/history diagnosis. No optimizer or learned policy change."""
from pathlib import Path
import ast, contextlib, copy, gzip, importlib.util, json, os, shutil, subprocess, sys, time, traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stage/experiments/t441_joint_actor_critic'))
import t441_learning as L
spec=importlib.util.spec_from_file_location('stage444_pinned441_runtime',ROOT/'stage/experiments/t441_joint_actor_critic/run.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
C,E,torch,np=L.C,L.E,L.torch,L.np
read,save,sha,digest=L.read,L.save,L.sha,L.digest
OUT=ROOT/'stage/runs/t444_paired_retention';DOC=ROOT/'stage/docs/research_rescue_20261109'
IDS=['dev_task6_'+str(i).zfill(2) for i in [0,3,6,9,15,18,21,24]]

def used(folder=None):
    return sum(f.stat().st_size for f in (Path(folder) if folder else ROOT/'stage/runs').rglob('*') if f.is_file())

def verify(p,full=False):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    assert used(OUT)<=p['max_bytes'] and used()-used(OUT)+p['max_bytes']<=2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30
    for f,h in (p['inputs'] if full else p['protected_files']).items():assert sha(f)==h,f

def feedback():
    """Fast red-capable replay of already SHA-verified actual closed-loop traces."""
    p=read(ROOT/'stage/runs/t443_sampling_control/protocol.json')
    folder=ROOT/'stage/runs/t443_sampling_control/left_coverage'
    rows=[json.loads(x) for x in (folder/'cases.jsonl').read_text().splitlines()]
    records=[r for r in rows if r['role']=='FINAL329' and r['job_id'] in IDS]
    assert len(records)==8
    for r in records:
        assert sha(r['path'])==r['sha256']
        a=E.N.S.arrays(r['path']);assert bool(a['end_flags'][-1,0])==r['collision']
        assert int(a['native_counts'].sum())==r['physics']
    failed=[r['job_id'] for r in records if not r['success'] or r['collision']]
    print(json.dumps(dict(known_actual_closed_loop_failures=failed,new_physics=0,new_optimizer=0)),flush=True)
    assert not failed,'Known exact symptom: current wide54/54 policy still collides in old LEFT5 tasks'

class ResetPolicy:
    """Exact T255 hidden-reset action function, with explicitly padded trace hidden."""
    def __init__(self,q,p):
        self.base=q.base;self.sample=False;self.rng=None
        assert not self.base.memory_on and digest(self.base.state_dict())==p['safe_base_policy_hash']
        self._hash=digest(dict(T255_hidden_reset=True,weights=self.base.state_dict()))
        self.reset_noise()
    def phash(self):return self._hash
    def reset_noise(self):self.tape=[];self.previous_raw=self.previous_location=None
    def step(self,x,hidden=None):
        # The first returned field is the original T255 deterministic mode.
        with torch.no_grad():action=self.base.step(x,None)[0].float()
        loc=action.double();raw=loc.clone();atom=torch.where(raw>=L.EDGE,1,torch.where(raw<=-L.EDGE,-1,0)).to(torch.int8)
        self.tape.append(dict(innovations=np.zeros(2),raw_actions=raw[0].numpy(),locations=loc[0].numpy(),
            logp=np.asarray(0.,np.float32),future_locations=np.tile(loc[0].numpy(),(4,1)),
            atom=atom[0].numpy(),stochastic=np.asarray(0,np.uint8)))
        self.previous_raw=raw;self.previous_location=loc
        return action,loc,x.new_zeros((1,1,16),dtype=torch.float64),loc

def policy(p,kind):
    q=C.frozen(p,p['student_model']['path']).requires_grad_(False)
    return ResetPolicy(q,p) if kind=='safe_reset' else q

def geometry(env):
    """Current 3-D geom distances, capped at .30m; not continuous swept clearance."""
    import mujoco as mj
    model=env.model;robot=[i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in env.robot_body_ids]
    obstacles=[]
    for i in range(model.ngeom):
        if i in robot:continue
        name=((mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,i) or '')+' '+
              (mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[i])) or '')).lower()
        if not any(k in name for k in ('floor','ground','plane')):obstacles.append(i)
    pairs=[(i,j) for i in robot for j in obstacles if
        (int(model.geom_contype[i])&int(model.geom_conaffinity[j])) or
        (int(model.geom_contype[j])&int(model.geom_conaffinity[i]))]
    assert pairs
    return pairs

def clearance(model,data,pairs):
    import mujoco as mj
    best=.30;pair=(-1,-1)
    for i,j in pairs:
        d=float(mj.mj_geomDistance(model,data,i,j,.30,None))
        if d<best:best=d;pair=(i,j)
    return best,pair

def worker(pipe,index,p,counts):
    import mujoco as mj
    E.N.offline();C.ban_learning();os.sched_setaffinity(0,{index});torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    folder=Path(p['out'])/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True);E.set_out(p['out'])
    log=(folder/'console.log').open('w');stdout,stderr=sys.stdout,sys.stderr;s=w=None
    def guard(physical=False):E.L.check_stop();assert time.time()<p['native_stop_epoch']-5 and sum(counts)<p['max_physics']
    def charge():counts[index+1]+=1;assert sum(counts)<=p['max_physics']
    try:
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            module,native,create,w,_=E.N.S.P.C.B.setup(folder/'native',E.N.science(p));original=copy.deepcopy(module.ACTIVE)
            E.L.install_stop_handlers();pipe.send(dict(state='READY',pid=os.getpid(),cpu=index))
            while True:
                req=pipe.recv()
                if req is None:break
                kind=req['kind'];q=policy(p,kind);j=req['job'];module.ACTIVE=copy.deepcopy(original)
                module.ACTIVE['circle_shift_x_m']=j.get('circle_shift_x_m',original['circle_shift_x_m'])
                before=counts[index+1];s=E.L.Session(j,native,create,w,guard,charge,q);E.L.install_stop_handlers()
                if req.get('cancel_probe'):
                    def cancel(physical=False):
                        if physical:raise E.L.StageStop('T444 zero-native cancellation')
                    s.guard=cancel
                    try:s.step();raise AssertionError('Cancellation missing')
                    except E.L.StageStop:pass
                    assert not s.rows and s.steps==0
                    pairs=geometry(s.env);distance,pair=clearance(s.env.model,s.env.data,pairs)
                    assert np.isfinite(distance)
                    pipe.send(dict(state='CANCEL_PASS',physics=0,geometry_pairs=len(pairs),qpos_size=s.env.model.nq))
                    s.close();s=None;continue
                pairs=geometry(s.env);diagnostics=[]
                if index==0 and not (OUT/'geometry.mjb').exists():
                    mj.mj_saveModel(s.env.model,str(OUT/'geometry.mjb'),None)
                    save(OUT/'geometry.json',dict(scene_sha256=s.scene,pairs=pairs,nq=s.env.model.nq,
                        geoms=[dict(id=i,name=mj.mj_id2name(s.env.model,mj.mjtObj.mjOBJ_GEOM,i),body=int(s.env.model.geom_bodyid[i])) for i in range(s.env.model.ngeom)],
                        clearance_cap_m=.30,decision_endpoint_only=True))
                while not s.done():
                    qpos=s.env.data.qpos.copy();distance,pair=clearance(s.env.model,s.env.data,pairs)
                    diagnostics.append((qpos,distance,pair))
                    s.step(owner=0)
                a=E.payload(s);a.update(qpos_before=np.asarray([d[0] for d in diagnostics]),
                    body_clearance_capped_m=np.asarray([d[1] for d in diagnostics]),body_geom_pair=np.asarray([d[2] for d in diagnostics],np.int16))
                contacts=[]
                for contact in s.env.data.contact[:s.env.data.ncon]:
                    if (int(contact.geom1),int(contact.geom2)) in pairs or (int(contact.geom2),int(contact.geom1)) in pairs:
                        contacts.append(dict(geom1=int(contact.geom1),geom2=int(contact.geom2),distance_m=float(contact.dist)))
                row=s.result();row.update(policy_hash=q.phash(),physics=counts[index+1]-before,array_digest=digest(a),
                    array_hashes={k:digest(v) for k,v in a.items()},integration_sha256=digest(a['final_native_integration']),
                    pid=os.getpid(),cpu=index,teacher=False,kind=kind,mode='student',terminal_obstacle_contacts=contacts)
                s.close();s=None;pipe.send((a,row))
    except BaseException:
        save(folder/'failure.json',dict(error=traceback.format_exc(),physics=counts[index+1]))
        if s is not None and s.rows:
            try:E.N.write_npz(folder/'partial.npz',E.N.S.packed(E.payload(s)),p['full_bytes'])
            except BaseException:save(folder/'partial_save_error.json',dict(error=traceback.format_exc()))
        try:pipe.send(dict(state='FAILED',error=traceback.format_exc()))
        except BaseException:pass
        raise
    finally:
        if s:s.close()
        if w:w.close()
        sys.stdout,sys.stderr=stdout,stderr;log.close();pipe.close()

class Pool(M.Pool):
    def __init__(self,p,counts,guard):
        self.guard=guard;self.children=[];self.pipes=[];ctx=M.mp.get_context('spawn')
        try:
            for i in range(8):
                a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
            save(Path(p['out'])/'workers.json',dict(parent=os.getpid(),workers=[c.pid for c in self.children]))
            self.ready=[self.receive(i) for i in range(8)];assert all(x['state']=='READY' for x in self.ready)
        except BaseException:self.close();raise

def prepare():
    assert not OUT.exists(),'Never restart an existing run'
    previous=ROOT/'stage/runs/t443_sampling_control';p=copy.deepcopy(read(previous/'protocol.json'))
    for name in ['summary.json','audit.json','SEALED.json']:assert (previous/name).exists()
    assert read(previous/'audit.json')['state']=='PASS'
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T443.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert all(not Path('/proc',str(pid)).exists() for pid in read(previous/'SEALED.json')['scientific_pids'])
    model=read(previous/'left_coverage/development.json')['model'];q=C.frozen(p,model['path']).requires_grad_(False)
    assert q.phash()==model['policy_hash'] and not q.base.memory_on
    safe=ROOT/'stage/runs/t255_overnight/s2_reset_final_checkpoint.pt.gz'
    p.update(id='T444_FROZEN_PAIRED_OLD_LEFT5_HISTORY_DIAGNOSIS',out=str(OUT),student_model=model,
        safe_model=dict(path=str(safe),sha256=sha(safe)),safe_base_policy_hash=digest(q.base.state_dict()),
        max_bytes=10*2**20,full_bytes=448*1024,max_physics=800000,max_commands=80,
        main_seconds=1200,learning_seconds=0,audit_seconds=300,backup_seconds=600,all_seconds=2400,
        zero_training=True,maximum_new_Actor_Adam=0,maximum_new_V_Adam=0,
        maximum_new_closed_loop_pairs=8,maximum_primary_trajectories=16,maximum_exact_replays=16,
        preflight_commands=24,main_pipe_commands=48,fresh16=None,reload78=None,six_gate=False,eight_started=False,
        no_runtime_teacher=True,no_waypoints_planner_router=True,
        safe_trace_notes='hidden16 zeros are padding; safe locations/raw_actions are effective original deterministic mode, not latent Gaussian means; future_locations repeats are compatibility padding, not predictions; logp0 not onpolicy training data',
        comparison='Same full external history, separate prescribed h0/reset. Two policy-function/history orders are descriptive algebra, not pure parameter or causal effects.',
        thresholds=dict(action_abs=.01,position_m=.01,yaw_deg=1.),
        user_authorization='Nov9 next step after T443: frozen failure localization; independent bounded protocol, no previous-run restart.')
    p['jobs']=[next(j for j in p['old275_jobs'] if j['id']==name) for name in IDS]
    oldrows=[json.loads(x) for x in gzip.decompress((ROOT/'stage/runs/t255_overnight/s2_reset_development.jsonl.gz').read_bytes()).decode().splitlines()]
    p['safe_historical_rows']={r['source_reset_id']:r for r in oldrows if r['source_reset_id'] in IDS}
    p['student_historical_rows']={r['job_id']:r for r in map(json.loads,(previous/'left_coverage/cases.jsonl').read_text().splitlines()) if r['role']=='FINAL329' and r['job_id'] in IDS}
    assert len(p['safe_historical_rows'])==len(p['student_historical_rows'])==8
    p['safe_policy_hash']=ResetPolicy(q,p).phash();del q
    p['worst_physics']=4*sum(j['case']['step_cap'] for j in p['jobs']);assert p['worst_physics']<=p['max_physics']
    p['protected_files']={str(safe):sha(safe),model['path']:model['sha256']}
    files=set(p['inputs'])|{str(f) for f in HERE.glob('*.py')}|{str(DOC/'T444_PLAN.md'),str(DOC/'T444_PRIMARY_SOURCES.md'),
        str(previous/'summary.json'),str(previous/'audit.json'),str(previous/'left_coverage/development.json'),str(previous/'left_coverage/cases.jsonl'),
        str(ROOT/'stage/runs/t255_overnight/s2_reset_development.jsonl.gz')}|set(p['protected_files'])
    p['inputs']={f:sha(f) for f in sorted(files)}
    assert used()+p['max_bytes']<2**30;OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T444_BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',global_before=used()-used(OUT),
        global_full_reserved=used()-used(OUT)+p['max_bytes'],max_bytes=p['max_bytes'],
        max_physics=p['max_physics'],worst_physics=p['worst_physics'],max_commands=p['max_commands'],maximum_commands=72,
        storage=dict(full16=16*p['full_bytes'],metadata_models_logs=3*2**20),main_seconds=1200,audit_seconds=300,all_seconds=2400,
        new_Actor_Adam=0,new_V_Adam=0))
    verify(p,True);print('T444_REGISTERED_ZERO_TRAINING',p['worst_physics'],flush=True)

def preflight():
    p=read(OUT/'protocol.json');verify(p,True);assert not (OUT/'preflight.json').exists()
    C.ban_learning();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    source=E.N.S.arrays(ROOT/'stage/runs/t443_sampling_control/left_coverage/episodes/0000.npz')
    q=policy(p,'student');reset=policy(p,'safe_reset');x=torch.tensor(source['states'][:9])
    with torch.no_grad():
        h=None;loc=[]
        for row in x:
            command,mu,h,_=q.step(row[None],h);loc.append(mu[0])
            actual=reset.step(row[None],h)[0];expected=q.base.step(row[None],None)[0].float();assert torch.equal(actual,expected)
        error=float((torch.stack(loc)-q.sequence(x,q.base_actions(x))).abs().max());assert error<1e-10
    folder=OUT/'preflight_native';folder.mkdir();counts=M.mp.get_context('spawn').RawArray('q',9)
    pool=Pool(dict(p,out=str(folder),native_stop_epoch=time.time()+180,max_physics=1),counts,lambda:None)
    try:
        results=pool.request([dict(kind='safe_reset' if i%2 else 'student',job=j,cancel_probe=True) for i,j in enumerate(p['jobs'])])
        assert all(r['state']=='CANCEL_PASS' for r in results) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',native=0,Actor_Adam=0,V_Adam=0,commands=24,
        independent_safe_T255_hidden_reset_commands_bit_exact=True,student_step_sequence_max_error=error,
        hidden_padding_for_safe_trace_NOT_carried_memory=True,eight_workers_cancelled_and_stopped=True,geometry=results))
    print('T444_PREFLIGHT_PASS_ZERO_NATIVE_ZERO_ADAM',flush=True)

def responses(p,a,kind):
    q=policy(p,kind);q.reset_noise();h=None;commands=[];locations=[]
    with torch.no_grad():
        for row in a['states']:
            action,loc,h,_=q.step(torch.tensor(row[None]),h);commands.append(action[0].numpy().copy());locations.append(loc[0].numpy().copy())
    return np.asarray(commands),np.asarray(locations)

def first_time(mask,times):
    ids=np.flatnonzero(mask);return None if not len(ids) else float(times[ids[0]])

def analyze(p,rows):
    reports=[]
    for job in p['jobs']:
        r0=next(r for r in rows if r['job_id']==job['id'] and r['kind']=='safe_reset')
        r1=next(r for r in rows if r['job_id']==job['id'] and r['kind']=='student')
        a0,a1=[E.N.S.arrays(r['path']) for r in [r0,r1]]
        assert np.array_equal(a0['states'][0],a1['states'][0]) and np.array_equal(a0['initial_native_integration'],a1['initial_native_integration'])
        values={};locations={}
        for k,a in enumerate([a0,a1]):
            for i,kind in enumerate(['safe_reset','student']):values[i,k],locations[i,k]=responses(p,a,kind)
            assert np.array_equal(values[k,k],a['actions'])
        n=min(len(a0['states']),len(a1['states']));times=[.002*np.r_[0,np.cumsum(a['native_counts'].astype(np.int64))[:-1]] for a in [a0,a1]]
        assert np.array_equal(times[0][:n],times[1][:n]);t=times[0][:n]
        A00,A10,A01,A11=[values[k][:n].astype(np.float64) for k in [(0,0),(1,0),(0,1),(1,1)]]
        P0=A10-A00;P1=A11-A01;H0=A01-A00;H1=A11-A10;P=(P0+P1)/2;H=(H0+H1)/2;D=A11-A00
        assert np.max(np.abs(D-P-H))<1e-12
        xy=np.linalg.norm(a1['poses'][:n,:2]-a0['poses'][:n,:2],axis=1)
        yaw=np.abs((a1['poses'][:n,2]-a0['poses'][:n,2]+np.pi)%(2*np.pi)-np.pi)
        packed=dict(time_s=t,A00=A00,A10=A10,A01=A01,A11=A11,policy_on_safe=P0,policy_on_student=P1,
            history_under_safe=H0,history_under_student=H1,symmetric_policy=P,symmetric_history=H,total=D,
            safe_locations_on_safe=locations[0,0][:n],student_locations_on_safe=locations[1,0][:n],
            safe_locations_on_student=locations[0,1][:n],student_locations_on_student=locations[1,1][:n],xy_distance_m=xy,yaw_difference_rad=yaw)
        path=OUT/'comparisons'/f"{job['id']}.npz";path.parent.mkdir(exist_ok=True);E.N.write_npz(path,packed,64*1024)
        window=a1['states'][-10:];clear=a1['body_clearance_capped_m'][-10:]
        reports.append(dict(job_id=job['id'],safe_success=r0['success'],student_collision=r1['collision'],common_decisions=n,
            common_last_decision_time_s=float(t[-1]),first_action_difference_over_001_s=first_time(np.max(np.abs(D),axis=1)>.01,t),
            first_xy_difference_over_1cm_s=first_time(xy>.01,t),first_yaw_difference_over_1deg_s=first_time(yaw>np.pi/180,t),
            maximum_xy_difference_m=float(xy.max()),maximum_yaw_difference_deg=float(np.rad2deg(yaw).max()),
            symmetric_policy_RMS_drive_steer=np.sqrt(np.mean(P**2,axis=0)).tolist(),symmetric_history_RMS_drive_steer=np.sqrt(np.mean(H**2,axis=0)).tolist(),
            decomposition_residual=float(np.max(np.abs(D-P-H))),student_last10_decisions=dict(speed_m_s=window[:,56].tolist(),
                drive_steer=a1['actions'][-10:].tolist(),clearance_capped_m=clear.tolist(),
                native_time_s=times[1][-10:].tolist()),terminal_obstacle_contacts=r1['terminal_obstacle_contacts'],
            comparison_path=str(path),comparison_sha256=sha(path),not_an_independent_physical_causal_intervention=True))
    return reports

def main():
    p=read(OUT/'protocol.json');verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS'
    assert not (OUT/'summary.json').exists() and not (OUT/'main_started.json').exists()
    E.N.offline();C.ban_learning();E.L.install_stop_handlers();M.OUT=OUT;M.verify=verify
    pp=dict(p,learning_seconds=p['main_seconds']);rt=M.Runtime(pp);error=None;state='FAILED';results=[]
    try:
        rt.pool=Pool(dict(p,native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        for kind in ['safe_reset','student']:
            model=p['safe_model'] if kind=='safe_reset' else p['student_model'];ph=p['safe_policy_hash'] if kind=='safe_reset' else model['policy_hash']
            references=[]
            for replay in (False,True):
                rt.role=kind+('_EXACT' if replay else '_PRIMARY');rt.guard();rt.commands+=8
                replies=rt.pool.request([dict(kind=kind,job=j) for j in p['jobs']])
                for i,((a,r),j) in enumerate(zip(replies,p['jobs'])):
                    assert r['policy_hash']==ph and r['complete'];r.update(job_id=j['id'],role=rt.role,model_sha256=model['sha256'])
                    old=p['safe_historical_rows'][j['id']] if kind=='safe_reset' else p['student_historical_rows'][j['id']]
                    for name in ['actions','native_counts','poses','next_poses','end_flags','rewards']:
                        assert digest(a[name])==old['array_hashes'][name],(kind,j['id'],name)
                    if kind=='safe_reset':assert E.N.K.R.arrays_hash(a['final_native_integration'])==old['integration_sha256']
                    else:assert digest(a['final_native_integration'])==old['integration_sha256']
                    if replay:
                        ref=references[i];assert r['array_digest']==ref['array_digest']
                        r.update(reference_path=ref['path'],exact=True,independent_new_sample=False);rt.replays.append(r);log='replays.jsonl'
                    else:
                        path=OUT/'episodes'/f'{kind}_{j["id"]}.npz';path.parent.mkdir(exist_ok=True)
                        E.N.write_npz(path,E.N.S.packed(a),p['full_bytes']);r.update(path=str(path),sha256=sha(path),full=True)
                        rt.rows.append(r);references.append(r);log='cases.jsonl'
                    with (OUT/log).open('a') as f:f.write(json.dumps(r)+'\n')
        results=analyze(p,rt.rows);save(OUT/'comparisons.json',results);state='COMPLETE'
    except BaseException:
        error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,physics=sum(rt.counts),commands=rt.commands,role=rt.role))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=8
        summary=dict(state=state,pid=os.getpid(),elapsed_s=time.time()-rt.start,physics=sum(rt.counts),
            commands=rt.commands+24,main_commands=rt.commands,preflight_commands=24,primary=len(rt.rows),exact=len(rt.replays),
            safe_success=sum(r['success'] for r in rt.rows if r['kind']=='safe_reset'),student_collisions=sum(r['collision'] for r in rt.rows if r['kind']=='student'),
            error=error,all_native_workers_stopped=True,Actor_Adam=0,V_Adam=0,fresh16=None,reload78=None,six_gate=False,eight_started=False)
        save(OUT/'summary.json',summary);save(OUT/'status.json',dict(state=state,epoch=time.time()))
        M.fcntl.flock(rt.lock,M.fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error
    print('T444_MAIN_COMPLETE_ZERO_TRAINING',flush=True)

def audit():
    import mujoco as mj
    p=read(OUT/'protocol.json');verify(p,True);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists()
    assert all(not Path('/proc',str(pid)).exists() for pid in [s['pid']]+read(OUT/'workers.json')['workers'])
    E.N.offline();C.ban_learning();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rows=[json.loads(x) for x in (OUT/'cases.jsonl').read_text().splitlines()];replays=[json.loads(x) for x in (OUT/'replays.jsonl').read_text().splitlines()]
    model=mj.MjModel.from_binary_path(str(OUT/'geometry.mjb'));data=mj.MjData(model);geom=read(OUT/'geometry.json');pairs=geom['pairs']
    for r in rows:
        assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert digest(a)==r['array_digest']
        action,loc=responses(p,a,r['kind']);assert np.array_equal(action,a['actions']) and np.array_equal(loc,a['locations'])
        assert int(a['native_counts'].sum())==r['physics'] and not a['ownership'].any() and not a['stochastic'].any()
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1
        for i,x in enumerate(a['states']):
            assert np.abs(E.N.K.H.rebuild(x,a['poses'][i],a['history_poses'][i],a['task_goal_xy'],a['history_goals'][i])-x).max()<=2e-6
            data.qpos[:]=a['qpos_before'][i];mj.mj_forward(model,data);value,pair=clearance(model,data,pairs)
            assert abs(value-a['body_clearance_capped_m'][i])<1e-10 and np.array_equal(pair,a['body_geom_pair'][i])
            n=int(a['native_counts'][i]);z=E.N.K.R.GAMMA**n;f=a['end_flags'][i];d0=np.linalg.norm(a['poses'][i,:2]-a['task_goal_xy']);d1=np.linalg.norm(a['next_poses'][i,:2]-a['task_goal_xy'])
            rw=-.000002*(1-z)/(1-E.N.K.R.GAMMA)+.03*d0-z*.03*d1*(not f.any())
            if f.any():rw+=E.N.K.R.GAMMA**(n-1)*(float(f[1])-float(f[0])-.5*float(f[2]))
            assert abs(rw-a['rewards'][i])<1e-12
    for r in replays:
        ref=next(v for v in rows if v['path']==r['reference_path']);assert r['array_digest']==ref['array_digest'] and r['integration_sha256']==ref['integration_sha256']
    assert len(rows)==len(replays)==16 and s['physics']==sum(r['physics'] for r in rows+replays)
    assert s['commands']==8+len(rows)+len(replays)+8+24==72 and s['physics']<=p['max_physics']
    # Recompute comparisons without replacing any saved original evidence.
    comparisons=read(OUT/'comparisons.json')
    for row in comparisons:
        assert sha(row['comparison_path'])==row['comparison_sha256'];v=E.N.S.arrays(row['comparison_path'])
        pair=[next(r for r in rows if r['job_id']==row['job_id'] and r['kind']==kind) for kind in ['safe_reset','student']]
        arrays=[E.N.S.arrays(r['path']) for r in pair];n=len(v['time_s'])
        for i,kind in enumerate(['safe_reset','student']):
            for j,actual_history in enumerate(arrays):
                action,loc=responses(p,actual_history,kind);assert np.array_equal(v[f'A{i}{j}'],action[:n].astype(np.float64))
        assert np.array_equal(v['total'],v['A11']-v['A00'])
        assert np.array_equal(v['symmetric_policy'],((v['A10']-v['A00'])+(v['A11']-v['A01']))/2)
        assert np.array_equal(v['symmetric_history'],((v['A01']-v['A00'])+(v['A11']-v['A10']))/2)
        assert np.max(np.abs(v['total']-v['symmetric_policy']-v['symmetric_history']))<1e-12
    save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),physics=s['physics'],commands=s['commands'],new_native=0,
        new_Actor_Adam=0,new_V_Adam=0,primary=16,exact_replays=16,full_history_actions_rewards_clearance_arrays_SHA_verified=True,
        historical_T255_and_T443_actions_poses_integrals_exact=True,all_native_workers_stopped=True,
        two_by_two_algebra_NOT_causal_effect=True,safe_hidden_is_explicit_padding_NOT_carried_memory=True))
    print('T444_AUDIT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    now=time.time()
    with (OUT/'driver_console.log').open('xb') as log:
        child=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T444',
            '--why=Frozen paired history diagnosis','timeout','--signal=TERM','--kill-after=2',str(p['all_seconds']),
            'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=child.pid,epoch=now,all_stop=now+p['all_seconds'],native_stop=now+p['main_seconds'],
        source_remote=receipt['remote_commit'],root_tree=receipt['root_tree'],protocol_sha256=sha(OUT/'protocol.json')))
    print('T444_ACTUALLY_LAUNCHED',child.pid,flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
