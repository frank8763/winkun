"""One-time root registration and actual-path zero-native offline contracts."""
from day_core import *
import ast,importlib.util,types,dis,builtins,fcntl,socket,subprocess
from zoneinfo import ZoneInfo
from datetime import datetime

def prepare():
    assert not ROOT_OUT.exists();assert read(ROOT/'stage/runs/t409_cold_npz_reserve/audit.json')['state']=='PASS';assert read(ROOT/'stage/runs/t408_joint_wide/audit.json')['state']=='PASS'
    for name in ['t408_joint_wide','t409_cold_npz_reserve']:
        s=read(ROOT/'stage/runs'/name/'summary.json');assert not Path('/proc',str(s['pid'])).exists()
    previous=read(ROOT/'stage/runs/t408_joint_wide/protocol.json');oldrows=[json.loads(x) for x in (ROOT/'stage/runs/t408_joint_wide/cases.jsonl').read_text().splitlines()];old=[r['path'] for r in oldrows if r['role'].startswith('initial_old8_')];assert len(old)==8
    common={k:previous[k] for k in ['source_protocol','raw_sigma','lr','epochs','ppo_clip','max_logratio','old8_jobs','left5_jobs','right_job','left_history_files','right_history_file','main_jobs','fresh_jobs','reload_jobs','rl_jobs','deployment']}
    common.update(old_history_files=old,old_weight=1.,broad_old_weight=1.,left_weight=.5,right_weight=.5,correction_weight=.5,consolidation_steps=600,correction_steps=200,full_bytes=400*1024,teacher_first=20,teacher_advance_holds=[0,2],cycles=16,reject_stop=2,minimum_cycle_seconds=240,seed=20261107410)
    freshids={j['id'] for j in common['fresh_jobs']};ledger={}
    for f in sorted((ROOT/'stage/runs').rglob('*.jsonl')):
        if f.name not in ['cases.jsonl','replays.jsonl','evaluations.jsonl']:continue
        ledger[str(f)]=sha(f)
        for line in f.read_text().splitlines():
            try:v=json.loads(line)
            except json.JSONDecodeError:continue
            if isinstance(v,dict):assert not any(isinstance(s,str) and any(s==i or s.endswith('_'+i) for i in freshids) for s in [v.get(k) for k in ['id','job_id','role','task_id']]),f
    common['fresh_ledger']=ledger
    stages=[dict(id='A',folder='lateral',kind='lateral',final_bytes=9*2**20,max_physics=8000000,max_commands=350,main_seconds=4500,learning_seconds=3600,audit_seconds=900),dict(id='B',folder='projected',kind='projected',final_bytes=9*2**20,max_physics=8000000,max_commands=350,main_seconds=4500,learning_seconds=3600,audit_seconds=900),dict(id='C',folder='recovery',kind='recovery',final_bytes=16*2**20,max_physics=14000000,max_commands=700,main_seconds=4500,learning_seconds=3600,audit_seconds=900),dict(id='V',folder='validation',kind='validation',final_bytes=6*2**20,max_physics=12000000,max_commands=600,main_seconds=4500,learning_seconds=0,audit_seconds=300)]
    paths=[*HERE.glob('*.py'),DOC/'PLAN.md',*map(Path,old),*map(Path,common['left_history_files']),Path(common['right_history_file']),PARENT,ROOT/'stage/runs/t408_joint_wide/summary.json',ROOT/'stage/runs/t408_joint_wide/audit.json',ROOT/'stage/runs/t409_cold_npz_reserve/audit.json',Path(common['source_protocol'])]
    paths.extend(Path(f) for f in previous['inputs'] if str(f).endswith('.py') or 'candidate.pt.gz' in str(f))
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    tz=ZoneInfo('Asia/Shanghai');epoch=lambda h,m:datetime(2026,11,7,h,m,tzinfo=tz).timestamp()
    p=dict(id='T410_OFFLINE_DAY_20261107',common=common,stages=stages,initial_model=str(PARENT),root_bytes=41*2**20,root_metadata_bytes=2**20,inputs={str(f):sha(f) for f in sorted(set(paths))},joint_ids=[j['id'] for j in [*common['old8_jobs'],*common['left5_jobs'],common['right_job']]],selection_groups=dict(old=[j['id'] for j in common['old8_jobs']],left=[j['id'] for j in common['left5_jobs']],right=[common['right_job']['id']]),learning_deadline=epoch(18,45),native_deadline=epoch(19,30),all_work_deadline=epoch(20,0),all_seconds=28500,limits=dict(physics=50000000,commands=2600,main_wall_s=20000),prior_actual=dict(physics=639303,commands=120,main_wall_s=205.7174723148346),authorization='2026-11-07 user explicitly offline unsupervised multi-mechanism experimentation until20:00, no network/LLM dependency, preserve Stageonly and endtoendsingleActor',selection='A/B independent fixedT404; C chooses audited retained A/B byLEFTcount,oldcount,RIGHTcount then armorder; V selects similarly acrossA/B/C, nofreshselection',network_required=False,remote_backup='PENDING_AFTER_NETWORK_RETURN; never block next registered science on GitHub',no_seven_eight_code_or_launch=True,stop='shared contractfailure stopschain, normalscientificrejection allowsnextexactpreprogrammedbranch; no retries/grids/seedchanges/oldbudgetextension')
    cap=lambda js:sum(j['case']['step_cap'] for j in js);joint=cap([*common['old8_jobs'],*common['left5_jobs'],common['right_job']]);base=read(common['source_protocol']);worstAB=5*joint+16*40000;worstC=2*joint+16*(joint+11*40000);worstV=2*joint+cap(common['main_jobs'])+cap(base['final_old_jobs'])+cap(common['fresh_jobs'])+cap(common['reload_jobs'])
    used=N.K.S.used(ROOT/'stage/runs');assert used+p['root_bytes']<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    assert worstAB<8000000 and worstC<14000000 and worstV<12000000;assert 2*worstAB+worstC+worstV+639303<50000000
    ROOT_OUT.mkdir();(ROOT_OUT/'preflight').mkdir();save(HERE/'protocol.json',p);save(ROOT_OUT/'protocol.json',p)
    budget=dict(state='FULL_ROOT_WORST_CASE_RESERVED',global_before=used,root_reserved=p['root_bytes'],global_full_worst=used+p['root_bytes'],limit=2**30,minimum_free=40*2**30,branch_physics=dict(A=worstAB,B=worstAB,C=worstC,V=worstV),preflight_native=0,preflight_commands_reserved=100,preflight_Adam=16,max_live=dict(A=616,B=616,C=16*208,V=0),shadow_separate=True,audit_historical_same_max=True,C_conditional_storage_blocks_not_promised16=True,full400KiB_verified_beforelaunch=True,model_bytes=512*1024,compact_bytes=5*1024,root1MiBmetadata=True)
    save(DOC/'BUDGET.json',budget);print('T410_REGISTERED_NOT_STARTED',budget['global_full_worst'],flush=True)

def preflight():
    N.offline();root=read(ROOT_OUT/'protocol.json');assert not (ROOT_OUT/'preflight.json').exists()
    for f,h in root['inputs'].items():assert sha(f)==h,f
    for name in ['run.py','audit.py','driver.py']:
        spec=importlib.util.spec_from_file_location('stage410_preflight_'+Path(name).stem,HERE/name);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        def check(code):
            missing={v.argval for v in dis.get_instructions(code) if v.opname=='LOAD_GLOBAL' and v.argval not in vars(mod) and not hasattr(builtins,v.argval)}
            for v in code.co_consts:
                if isinstance(v,types.CodeType):missing|=check(v)
            return missing
        for fn in vars(mod).values():
            if isinstance(fn,types.FunctionType) and fn.__module__==mod.__name__:assert not check(fn.__code__),(name,check(fn.__code__))
    folder=ROOT_OUT/'preflight';set_out(folder);p=dict(root['common'],initial_model=root['initial_model'],out=str(folder),native_stop_epoch=time.time()+300,max_physics=1000000,final_bytes=2**20);base=N.science(p);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    q,opt=learner(base,p);parent=STAGE404.actor(base,p,PARENT);a=N.S.arrays(p['right_history_file']);h=ph=None;locs=[]
    with torch.no_grad():
        for x in a['states']:
            u,l,h,_=q.step(torch.tensor(x[None]),h);v,r,ph,_=parent.step(torch.tensor(x[None]),ph);assert equal(u,v) and equal(l,r) and equal(h,ph);locs.append(l[0])
        x=torch.tensor(a['states']);seq=q.sequence(x,q.base_actions(x));assert float((seq-torch.stack(locs)).abs().max())<1e-10
    broad=N.old_data(q,base,lambda:None);g=groups(q,[N.S.arrays(f) for f in p['old_history_files']],p,broad);before=L.state(q,opt);shadow,so=learner(base,p);calls=0
    for fn in [fit_A,fit_B]:
        L.load(q,opt,before);L.load(shadow,so,before);one=fn(q,opt,g,p,lambda:None,lambda *_:None,steps=2);two=fn(shadow,so,g,p,lambda:None,lambda *_:None,steps=2);calls+=4;assert one==two and L.statehash(q,opt)==L.statehash(shadow,so);L.load(q,opt,before);assert digest(L.state(q,opt))==digest(before)
    # Nonzero action labels also exercise the exact correction-only prefix mask.
    g['corrections']=[dict(x=x,b=q.base_actions(x),target=torch.tensor(a['actions'],dtype=torch.float64),first=20)]
    q.reset_noise();q.rng=np.random.default_rng(410);q.sample=True;h=None
    with torch.no_grad():
        for row in x:_,_,h,_=q.step(row[None],h)
    synthetic=copy.deepcopy(a);synthetic.update({k:np.asarray([v[k] for v in q.tape]) for k in q.tape[0]});items=[]
    for k in range(4):v=copy.deepcopy(synthetic);v['rewards'][-1]+=.1*k;items.append(v)
    short=dict(p,epochs=2)
    for projected in [False,True]:
        L.load(q,opt,before);L.load(shadow,so,before);one=ppo_fit(q,opt,items,g,short,0,lambda:None,lambda *_:None,projected);two=ppo_fit(shadow,so,items,g,short,0,lambda:None,lambda *_:None,projected);calls+=4;assert one==two and L.statehash(q,opt)==L.statehash(shadow,so)
    L.load(q,opt,before);assert digest(L.state(q,opt))==digest(before) and original_hash(q)==original_hash(parent);export(q,opt,folder/'clone.pt.gz')
    rng=np.random.default_rng(410);padded={}
    for k,v in N.S.packed(synthetic).items():
        shape=(401,*v.shape[1:]) if k.startswith('history_') else (400,*v.shape[1:]) if v.ndim and len(v)==len(x) else v.shape;padded[k]=rng.normal(size=shape).astype(v.dtype) if np.issubdtype(v.dtype,np.floating) else rng.integers(0,100,size=shape,dtype=v.dtype)
    padded.update(ownership=rng.integers(0,2,400,np.uint8),hold_sha256=rng.integers(0,256,(400,32),np.uint8),task_goal_xy=np.zeros(2),scene_sha256=np.zeros(32,np.uint8));full=len(N.R.blob(padded));assert full<=p['full_bytes']
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    denied=False
    try:socket.create_connection(('example.com',443),timeout=.01)
    except AssertionError:denied=True
    assert denied
    counts=mp.get_context('spawn').RawArray('q',5);pool=Pool(p,counts,lambda:None)
    try:
        v=pool.request([dict(model=str(folder/'clone.pt.gz'),job=p['right_job'],first=0,seed=None,cancel_probe=True)]*4);assert sum(counts)==0 and all(r['state']=='CANCEL_PASS' for r in v)
    finally:pool.close()
    # The supervisor's actual group termination path must work while offline.
    spec=importlib.util.spec_from_file_location('stage410_stop_probe',HERE/'driver.py');driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver);driver.ACTIVE=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True);driver.stop();assert driver.ACTIVE.poll() is not None
    result=dict(state='PASS',native=0,discarded_Adam=calls,zero_physics_commands=8,parameter_isolation4450=True,parent_function_actions_locations_hidden_exact=True,sequence_serial_density=True,actual_A_B_PPO_correction_shadow_and_mature_rollback=True,actual4workers_READY_cancel_exit=True,actual_writer_lock=True,external_network_rejected=True,actual_driver_process_group_stop=True,recursive_main_audit_driver_globals=True,incompressible_full_bytes=full,models_limit512KiB=True,source_committed_and_remote_required_beforelaunch=True)
    save(ROOT_OUT/'preflight.json',result);save(DOC/'READINESS.json',result);N.manifest(folder);print('T410_PREFLIGHT_PASS_0_NATIVE_16_DISCARDED_ADAM',flush=True)

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        DOC.mkdir(parents=True,exist_ok=True);f=DOC/(sys.argv[1]+'_ERROR.json');save(f,dict(error=traceback.format_exc(),native='zero for registration/preflight',no_automatic_retry=True));raise
