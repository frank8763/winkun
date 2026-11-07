"""Fixed native full-remaining-horizon terminal-capture CEM teacher pilot.

The sole student is frozen. Every search branch is real charged simulation.
Both search passes and the selected complete trajectory are natively replayed.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import sys,os,json,time,copy,hashlib,shutil,traceback,fcntl,contextlib,io,importlib.util,dis,builtins,types
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t421_terminal_capture';DOC=ROOT/'stage/docs/day_20261107'
sys.path.insert(0,str(ROOT/'stage/experiments/t418_evening_repair_chain'))
import repair_core as E
torch,np=E.torch,E.np
read,save,sha,digest=E.read,E.save,E.sha,E.digest
def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    assert E.N.K.S.used(OUT)<=p['final_bytes'] and E.N.K.S.used(ROOT/'stage/runs')<=2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30
def prohibit_learning():
    def prohibited(*a,**k):raise AssertionError('T421 forbids Actor/Adam/backward updates')
    torch.optim.Adam.step=prohibited;torch.optim.AdamW.step=prohibited;torch.optim.SGD.step=prohibited;torch.Tensor.backward=prohibited
def controls(theta,nominal,p):
    theta=np.asarray(theta,np.float64);assert theta.shape==(6,) and nominal.shape==(100,2)
    result=nominal.copy()
    for j,(begin,end) in enumerate(zip(p['blocks'][:-1],p['blocks'][1:])):
        result[begin:end,0]=theta[j];result[begin:end,1]=np.clip(nominal[begin:end,1]+theta[j+3],-1,1)
    return np.clip(result,-1,1).astype(np.float32)
def sample_params(mu,sd,rng,p):
    values=rng.normal(size=(p['population'],6))*sd+mu;values[0]=mu
    return np.clip(values,np.asarray(p['theta_low']),np.asarray(p['theta_high']))
def score(result,actions,remaining_steps):
    smooth=float(np.square(np.diff(actions.astype(np.float64),axis=0)).sum())
    if result['success'] and not result['collision']:return [0,float(remaining_steps),smooth]
    if not result['collision']:return [1,float(result['remaining_m']),smooth]
    return [2,float(result['remaining_m']),smooth]
def snapshot_hash(s):
    return digest(dict(integration=E.N.K.R.integration(s.env),packet=s.packet(),history=s.w.get(s.env),hidden=s.hidden,steps=s.steps,previous=s.previous,pn=s.pn,goal=s.goal,scene=s.scene))
def prefix_check(a,source,n):
    keys=['states','next_states','poses','next_poses','actions','native_counts','end_flags','rewards','history_poses','history_goals','history_times','environment_rewards','hidden_before','innovations','raw_actions','locations','logp','future_locations','atom','stochastic','hold_sha256','ownership']
    for key in keys:
        if key in a:assert E.equal(a[key],source[key][:len(a[key])]),('Actual prefix changed',key)
    assert E.equal(a['initial_native_integration'],source['initial_native_integration'])
    assert E.equal(a['last_hidden_after'],source['hidden_before'][n])
    assert int(a['native_counts'].sum())==30000
    return dict(holds=n,native=30000,all_recorded_prefix_arrays_and_student_hidden_exact=True,not_pure_student_prefix=True)
def prepare():
    assert not OUT.exists();source=ROOT/'stage/runs/t420_fixed_early_recovery';p=copy.deepcopy(read(source/'protocol.json'))
    assert read(source/'audit.json')['state']=='PASS' and read(DOC/'T420_FINAL_REPORT.json')['state']=='COMPLETE_AUDITED_REMOTE_VERIFIED'
    prior=read(DOC/'T420_FINDINGS.json');a=E.N.S.arrays(source/'mixed.npz');assert len(a['states'])==400 and np.all(a['native_counts']==100)
    pose=a['poses'][300];relative=a['task_goal_xy']-pose[:2];bearing=float((np.arctan2(relative[1],relative[0])-pose[2]+np.pi)%(2*np.pi)-np.pi)
    stop=datetime(2026,11,7,22,0,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
    p.update(id='T421_NATIVE_FULL_REMAINING_TERMINAL_CAPTURE',out=str(OUT),source_mixed=str(source/'mixed.npz'),source_mixed_sha256=sha(source/'mixed.npz'),source_case=str(source/'case.json'),
        prefix_holds=300,horizon_holds=100,blocks=[0,20,50,100],seed=20261107421,generations=2,population=12,elite=4,
        mu=[.2,0.,-.4,float(.5*np.sign(bearing)),0.,0.],sd=[.7,.7,.7,.65,.65,.65],min_sd=[.15]*6,
        theta_low=[-1.,-1.,-1.,-1.25,-1.25,-1.25],theta_high=[1.,1.,1.,1.25,1.25,1.25],
        initialization_geometry=dict(pose=pose.tolist(),goal=a['task_goal_xy'].tolist(),distance=float(np.linalg.norm(relative)),bearing_rad=bearing,privileged_training_only=True),
        selection='lex trueGOAL/noCONTACT, then arrival native; nonGOAL safe before collision and rank original80s end distance; smoothness last. Selected actual tested proposal only, no unsafe mean deployment.',
        scientific_change='Full remaining20s native trajectory optimization of absolute drive3 and relative steer3; not old short-horizon MPPI configuration.',
        prior_day_actual=dict(physics=prior['day_cumulative_physics'],commands=prior['day_cumulative_commands'],main_wall_s=prior['day_main_wall_seconds']),
        max_physics=600000,max_commands=56,max_main_commands=54,preflight_zero_commands=2,final_bytes=8*2**20,
        main_seconds=900,audit_seconds=180,backup_seconds=180,source_seconds=120,all_seconds=1800,
        all_stop_absolute=stop,absolute_stop=stop,native_stop_absolute=stop-600,learning_stop_absolute=stop-600,
        Actor_updates=0,optimizer_steps=0,Actor_Adam_steps=0,control_search_generations=2,control_search_proposals=24,
        time_authorization='New user explicitly started the proposed terminalcapture test after21:00; one new bounded verification only, previous night protocols remain stopped. New launch-relative30min limit and22:00absolute limit.',
        full_search_passes=2,selected_complete_origin_replay=True,full_trajectory_label='Actual bounded optimized actions only; student locations/hidden are shadow diagnostics, not teacher latent actions; mixed/offpolicy.',
        no_automatic_student_training=True,no_parameter_or_seed_grid=True)
    files=set(p['inputs'])|{str(f) for f in HERE.glob('*.py')}|{p['source_mixed'],p['source_case'],str(source/'summary.json'),str(source/'audit.json'),str(DOC/'T420_FINDINGS.json'),str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T420.json')}
    p['inputs']={f:sha(f) for f in sorted(files)}
    assert time.time()+p['all_seconds']<stop and E.N.K.S.used(ROOT/'stage/runs')+p['final_bytes']<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    assert p['prior_day_actual']['physics']+600000<=50_000_000 and p['prior_day_actual']['commands']+56<=2600 and p['prior_day_actual']['main_wall_s']+900<=20000
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T421_BUDGET.json',dict(state='NEW_INDEPENDENT_TERMINAL_TEST_NOT_OLD_BUDGET_RESTART',own_bytes=p['final_bytes'],max_physics=600000,physics_breakdown=dict(two_prefixes=60000,two_nominal_suffixes=20000,two24proposal_searches=480000,selected_origin_replay=40000),main_commands=54,preflight_commands=2,all_commands=56,Actor_Adam=0,live_control_CEM_generations=2,control_proposals=24,native_exact_control_repeats=24,main_seconds=900,audit_seconds=180,backup_seconds=180,source_seconds=120,all_seconds=1800,absolute_all_stop=stop,global_reserved=E.N.K.S.used(ROOT/'stage/runs')+p['final_bytes'],prior_day_actual=p['prior_day_actual']))
    print('T421_REGISTERED_NOT_LAUNCHED',flush=True)
def setup(p,folder):
    E.set_out(Path(folder).parent)
    base=E.N.science(p);module,native,create,witness,_=E.N.S.P.C.B.setup(folder,base);module.ACTIVE=copy.deepcopy(module.ACTIVE)
    module.ACTIVE['circle_shift_x_m']=p['job'].get('circle_shift_x_m',module.ACTIVE['circle_shift_x_m'])
    return module,native,create,witness
def preflight():
    p=read(OUT/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists();prohibit_learning();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for name in ['science.py','runner.py','audit.py','transport.py']:
        spec=importlib.util.spec_from_file_location('stage421_preflight_'+Path(name).stem,HERE/name);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        def check(code):
            bad={i.argval for i in dis.get_instructions(code) if i.opname=='LOAD_GLOBAL' and i.argval not in vars(module) and not hasattr(builtins,i.argval)}
            for item in code.co_consts:
                if isinstance(item,types.CodeType):bad|=check(item)
            return bad
        for fun in vars(module).values():
            if isinstance(fun,types.FunctionType) and fun.__module__==module.__name__:assert not check(fun.__code__),(name,check(fun.__code__))
    source=E.N.S.arrays(p['source_mixed']);nominal=source['actions'][300:400];rng=np.random.default_rng(p['seed']);params=sample_params(np.asarray(p['mu']),np.asarray(p['sd']),rng,p)
    assert params.shape==(12,6) and E.equal(params[0],np.asarray(p['mu'])) and np.abs(controls(params[0],nominal,p)).max()<=1
    assert score(dict(success=True,collision=False,remaining_m=.19),nominal,5000)<score(dict(success=False,collision=False,remaining_m=.01),nominal,10000)<score(dict(success=False,collision=True,remaining_m=.001),nominal,1)
    folder=OUT/'preflight_native';folder.mkdir();s=w=None
    with (folder/'console.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            _,native,create,w=setup(p,folder/'native');q=E.safe_initial(E.N.science(p),p).eval().requires_grad_(False);assert q.phash()==p['student_policy_hash']
            def refuse(physical=False):
                if physical:raise E.L.StageStop('T421 zero native cancellation')
            s=E.L.Session(p['job'],native,create,w,refuse,lambda:(_ for _ in ()).throw(AssertionError('No preflight physics')),q)
            snap=s.capture();before=snapshot_hash(s);s.restore(snap);assert before==snapshot_hash(s) and s.steps==0
            try:s.step(source['actions'][0],owner=0,actor_active=True);raise AssertionError('No cancellation')
            except E.L.StageStop:pass
            assert s.steps==0 and not s.rows
        finally:
            if s:s.close()
            if w:w.close()
    save(OUT/'preflight.json',dict(state='PASS',native=0,Actor_Adam=0,zero_commands=2,snapshot_native_clock_history_hidden_goal_exact=True,actual_session_cancel_before_first_native=True,all_source_SHA=True,recursive_entry_globals=True,terminal_success_priority_and_sampler_bounds=True,first_control_mean=params[0].tolist(),no_new_recovery_claim=True));E.N.manifest(OUT)
    print('T421_PREFLIGHT_PASS_0NATIVE_0ACTOR_ADAM',flush=True)
def main():
    E.N.offline();prohibit_learning();p=read(OUT/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'summary.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);E.L.install_stop_handlers();source=E.N.S.arrays(p['source_mixed']);nominal=source['actions'][300:400].copy();start=time.time();deadline=min(start+p['main_seconds'],p['native_stop_absolute']);physical=0;commands=0;phase='SETUP';last=0.;state='FAILED';s=w=None;records=[];cem=[];first_arrays=[];selected=None;result={};snapshot_records=[]
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(OUT/'main_started.json',dict(pid=os.getpid(),epoch=start,deadline=deadline,Actor_updates=0,Actor_Adam_steps=0));pred=dict(controls=np.zeros((25,100,2),np.float32),poses=np.full((25,100,3),np.nan),next_poses=np.full((25,100,3),np.nan),native_counts=np.zeros((25,100),np.uint8),flags=np.zeros((25,100,3),np.uint8),rewards=np.full((25,100),np.nan),hold_sha256=np.zeros((25,100,32),np.uint8),final_integration=np.zeros((25,203)),lengths=np.zeros(25,np.int32))
    def guard(physical_step=False,**kwargs):
        nonlocal last
        E.L.check_stop();assert time.time()<deadline-5 and physical<=p['max_physics'] and commands<=p['max_main_commands']
        if not physical_step and not kwargs.get('physical',False) and time.time()-last>3:
            last=time.time();verify(p);save(OUT/'status.json',dict(state=phase,pid=os.getpid(),epoch=last,physics=physical,main_commands=commands,Actor_updates=0,Actor_Adam_steps=0,control_proposals_completed=sum(r['kind']=='candidate' and r['pass_index']==0 for r in records)))
    def charge():
        nonlocal physical
        physical+=1;assert physical<=p['max_physics']
    def command(role,before):
        with (OUT/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(id=commands,role=role,physics=physical-before,pid=os.getpid()))+'\n')
    def branch(pass_index,index,u,snap,kind):
        nonlocal commands,phase
        phase=f'PASS{pass_index}_{kind}{index}';guard();assert physical+10000<=p['max_physics'];s.restore(snap);assert snapshot_hash(s)==snapshot_records[pass_index]['sha256'];before=physical;commands+=1
        for action in u:
            s.step(action,owner=1,actor_active=True)
            if s.done():break
        assert s.done();a=E.payload(s);r=s.result();n=len(a['actions'])-300
        record=dict(pass_index=pass_index,index=index,kind=kind,length=n,physics=physical-before,success=r['success'],collision=r['collision'],timeout=r['timeout'],remaining_m=r['remaining_m'],terminal_native=s.steps,score=score(r,a['actions'][300:],s.steps-30000),array_digest=digest(a),array_hashes={k:digest(v) for k,v in a.items()},integration_sha256=digest(a['final_native_integration']))
        if pass_index==0:
            first_arrays.append(a);pred['controls'][index]=u;pred['lengths'][index]=n
            for key,target in [('poses','poses'),('next_poses','next_poses'),('native_counts','native_counts'),('end_flags','flags'),('rewards','rewards'),('hold_sha256','hold_sha256')]:pred[target][index,:n]=a[key][300:]
            pred['final_integration'][index]=a['final_native_integration']
        else:
            reference=next(v for v in records if v['pass_index']==0 and v['index']==index);assert {k:v for k,v in record.items() if k!='pass_index'}=={k:v for k,v in reference.items() if k!='pass_index'}
            for key in a:assert E.equal(a[key],first_arrays[index][key]),('Full branch native replay mismatch',index,key)
        records.append(record)
        with (OUT/'branches.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        command(phase,before);return record
    try:
        with (OUT/'native_console.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            _,native,create,w=setup(p,OUT/'native');E.L.install_stop_handlers();commands=1;command('native_setup',physical)
            for pass_index in range(2):
                phase=f'PASS{pass_index}_PREFIX';before=physical;commands+=1;q=E.safe_initial(E.N.science(p),p).eval().requires_grad_(False);assert q.phash()==p['student_policy_hash'];s=E.L.Session(p['job'],native,create,w,guard,charge,q);E.L.install_stop_handlers()
                for i in range(300):s.step(source['actions'][i],owner=int(source['ownership'][i]),actor_active=True)
                prefix=E.payload(s);prefix_check(prefix,source,300);assert s.steps==30000 and not s.done();assert abs(float(s.packet()[112])-.25)<2e-6;snap=s.capture();sh=snapshot_hash(s);snapshot_records.append(dict(pass_index=pass_index,sha256=sh,steps=s.steps,packet_sha256=digest(s.packet()),integration_sha256=digest(E.N.K.R.integration(s.env)),prefix_payload_sha256=digest(prefix)));command(phase,before)
                if pass_index:assert snapshot_records[0]['sha256']==sh and snapshot_records[0]['prefix_payload_sha256']==snapshot_records[1]['prefix_payload_sha256']
                baseline=branch(pass_index,0,nominal,snap,'baseline');baseline_a=E.payload(s)
                for key in baseline_a:assert E.equal(baseline_a[key],source[key]),('T420 nominal suffix not exact',key)
                rng=np.random.default_rng(p['seed']);mu=np.asarray(p['mu']);sd=np.asarray(p['sd'])
                for gen in range(p['generations']):
                    theta=sample_params(mu,sd,rng,p);current=[]
                    for j,v in enumerate(theta):current.append(branch(pass_index,1+gen*12+j,controls(v,nominal,p),snap,'candidate'))
                    elite=sorted(range(12),key=lambda j:(current[j]['score'],j))[:p['elite']]
                    item=dict(pass_index=pass_index,generation=gen,mu=mu.tolist(),sd=sd.tolist(),theta=theta.tolist(),scores=[v['score'] for v in current],elite=elite)
                    mu=theta[elite].mean(0);sd=np.maximum(theta[elite].std(0),np.asarray(p['min_sd']));item.update(next_mu=mu.tolist(),next_sd=sd.tolist());cem.append(item);save(OUT/'cem.json',cem)
                s.close();s=None
            live=[r for r in records if r['pass_index']==0];best=min(live,key=lambda r:(r['score'],r['index']));selected=best['index'];u=pred['controls'][selected];phase='SELECTED_COMPLETE_ORIGIN_REPLAY';before=physical;commands+=1;assert physical+40000<=p['max_physics'];q=E.safe_initial(E.N.science(p),p).eval().requires_grad_(False);s=E.L.Session(p['job'],native,create,w,guard,charge,q);E.L.install_stop_handlers()
            for i in range(300):s.step(source['actions'][i],owner=int(source['ownership'][i]),actor_active=True)
            for action in u:
                s.step(action,owner=1,actor_active=True)
                if s.done():break
            a=E.payload(s)
            for key in a:assert E.equal(a[key],first_arrays[selected][key]),('Selected origin replay mismatch',key)
            a['control_source']=np.r_[np.zeros(17,np.uint8),np.ones(283,np.uint8),np.full(len(a['actions'])-300,2,np.uint8)]
            E.N.write_npz(OUT/'selected_mixed.npz',E.N.S.packed(a),p['query_full_bytes']);command(phase,before)
            result=dict(selected_index=selected,selected_branch=best,selected_path=str(OUT/'selected_mixed.npz'),selected_sha256=sha(OUT/'selected_mixed.npz'),selected_array_digest=digest(a),valid_recovery_label=bool(best['success'] and not best['collision']),mixed_offpolicy=True,labels_actual_executed_bounded_actions_only=True,student_policy_hash=q.phash(),frozen_student_unchanged=q.phash()==p['student_policy_hash'],old275=None,wide54=None,fresh16=None,reload78=None,no_six_metre_acceptance=True)
            s.close();s=None
        E.N.write_npz(OUT/'predictions.npz',pred,1024*1024);save(OUT/'snapshots.json',snapshot_records);state='COMPLETE'
    except BaseException:
        save(OUT/'failure.json',dict(error=traceback.format_exc(),phase=phase,physics=physical,commands=commands,no_training_restart=True))
        if s is not None and s.rows:
            try:E.N.write_npz(OUT/'partial.npz',E.N.S.packed(E.payload(s)),p['query_full_bytes'])
            except BaseException:save(OUT/'partial_error.json',dict(error=traceback.format_exc()))
    finally:
        if s:s.close()
        if w:w.close()
        save(OUT/'summary.json',dict(state=state,pid=os.getpid(),elapsed_s=time.time()-start,physics=physical,main_commands=commands,preflight_commands=2,all_commands=commands+2,Actor_updates=0,optimizer_steps=0,Actor_Adam_steps=0,control_search_generations=2,control_search_proposals=24,native_search_passes_completed=len(snapshot_records),all_sessions_closed=True,result=result));save(OUT/'status.json',dict(state=state,epoch=time.time(),physics=physical,main_commands=commands,Actor_updates=0,Actor_Adam_steps=0));E.N.manifest(OUT);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert state=='COMPLETE';print('T421_TERMINAL_CAPTURE_COMPLETE',result.get('valid_recovery_label'),flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
