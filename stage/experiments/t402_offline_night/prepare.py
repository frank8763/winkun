"""All scientific branches, budgets, and offline child probes locked before launch."""
from night import *
import ast,datetime,zoneinfo

def register():
    assert not OUT.exists();assert read(ROOT/'stage/runs/t401_cold_delta_reserve/audit.json')['state']=='PASS'
    t=read(ROOT/'stage/runs/t398_warm_recovery_rl/protocol.json');old=read(ROOT/'stage/runs/t393_safe_demo_pg/protocol.json')
    paths=[*HERE.glob('*.py'),SAFE_PARENT,WARM_PARENT,REFERENCE,RECOVERY,KNOWN_BRIDGE,Path(t['source_protocol']),DOC/'PLAN.md',DOC/'METHODS_AND_LIMITS.md',ROOT/'stage/docs/review_20261106/T399_FINDINGS.md',ROOT/'stage/runs/t399_rejected_warm_diagnostic/episodes/recover53.npz',ROOT/'stage/runs/t399_rejected_warm_diagnostic/episodes/recover54.npz',ROOT/'stage/runs/t399_rejected_warm_diagnostic/audit.json',ROOT/'stage/runs/t401_cold_delta_reserve/audit.json',ROOT/'stage/experiments/t398_warm_recovery_rl/t398_learning.py',ROOT/'stage/experiments/t396_iterative_recovery/t396_learning.py',ROOT/'stage/experiments/t395_frontier_recovery/t395_learning.py',ROOT/'stage/experiments/t393_safe_demo_pg/t393_learning.py',ROOT/'stage/experiments/t392_demo_closed_loop/t392_learning.py',ROOT/'stage/experiments/t387_full_horizon_night/night_common.py',ROOT/'stage/experiments/t381_native_mppi_teacher/run.py']
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    tz=zoneinfo.ZoneInfo('Asia/Shanghai');epoch=lambda h,m:datetime.datetime(2026,11,7,h,m,tzinfo=tz).timestamp()
    p={k:copy.deepcopy(t[k]) for k in ['source_protocol','centres','centre_left','shift_x_m','raw_sigma','lr','epochs','ppo_clip','max_logratio','demo_weight','old_weight','teacher_start_hold','bridge_holds','warm_steps','deployment']}
    p.update(id='T402_NETWORK_INDEPENDENT_CURRENT_HISTORY_AND_REVERSE_RL_NIGHT',user_authorization='2026-11-06 offline unattended night, all work stops2026-11-07 08:30 confirmed',inputs={str(f):sha(f) for f in sorted(set(paths))},seed=20261106402,rounds=6,prefix_holds=[265,260,250,225,150,0],learning_seconds=28800,main_seconds=32400,learning_stop_epoch=epoch(7,0),native_stop_epoch=epoch(8,0),all_stop_epoch=epoch(8,30),audit_seconds=1500,final_bytes=20*1024**2,full_bytes=400*1024,max_commands=1200,max_physics=40000000,full_gate_additional_bytes=34*1024**2,old275=None,main54=None,fresh16=None,reload78=None,seven_eight=False,scientific_scope='Joint engineerpilot, historical T398control reused notmatchedcausal; nojournalnoveltyclaim',stop='Fixedfinite1/2/4probe, A200 warm, conditionalB200consolidation, conditional6PPOrounds;2rejectionsor2zero-frontier-successroundstop;absolute deadlines',full_gate_policy='Notpreimplemented inthisnight; complete53recovery/old8/centresonlypilot, neverstableclaim or7/8')
    worst=24*p['full_bytes']+8*p['full_bytes']+4*p['full_bytes']+12*224*1024+130*5*1024+12*64*1024+512*1024
    used=N.K.S.used(ROOT/'stage/runs');assert worst<=p['final_bytes'] and used+p['final_bytes']<=1024**3 and shutil.disk_usage(ROOT).free>=40*1024**3
    OUT.mkdir();(OUT/'episodes').mkdir();(OUT/'blocks').mkdir();(OUT/'probes').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'BUDGET_AUDIT.json',dict(state='FULL_WORST_RESERVATION_PASS',global_before=used,full_reserved=used+p['final_bytes'],global_limit=1024**3,own=p['final_bytes'],worst_bytes=worst,train24full=24*p['full_bytes'],own8full=8*p['full_bytes'],partial4full=4*p['full_bytes'],models10=12*224*1024,compact130=130*5*1024,probe12full=12*64*1024,metadata512KiB=512*1024,max_physics=40000000,max_commands=1200,worst_main_commands=210,worst_main_physics=8000000,supervised_live_max400_shadow400_audit400=True,RL_live_max48_shadow48_audit48=True,preflight_clones=12,absolute_learning=p['learning_stop_epoch'],absolute_native=p['native_stop_epoch'],absolute_all=p['all_stop_epoch']))
    print('T402_REGISTERED_NOT_LAUNCHED',worst,flush=True)

def preflight():
    N.offline();p=load_plan();base=verify(p);assert not (OUT/'preflight.json').exists();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    begin=time.time();guard=lambda:(_ for _ in ()).throw(TimeoutError('T402 preflight300s')) if time.time()>begin+300 else None
    q,opt=new_learner(base,p,SAFE_PARENT);refs=L.groups(q,p,[KNOWN_BRIDGE]);a=N.S.arrays(ROOT/'stage/runs/t399_rejected_warm_diagnostic/episodes/recover53.npz')
    x=torch.tensor(a['states']);old=targets(q,[(a,0)]);initial=state(q,opt);sq,so=new_learner(base,p,SAFE_PARENT)
    pp=dict(p,warm_steps=2);live=L.warm_fit(q,opt,refs,old,pp,guard,lambda *_:None);other=L.warm_fit(sq,so,L.groups(sq,p,[KNOWN_BRIDGE]),old,pp,guard,lambda *_:None)
    assert live==other and statehash(q,opt)==statehash(sq,so);mature=state(q,opt);load(q,opt,initial);assert digest(state(q,opt))==digest(initial);load(q,opt,mature)
    g=targets(q,[(a,265)]);live=consolidate_fit(q,opt,old,g,p,guard,lambda *_:None,steps=2);other=consolidate_fit(sq,so,old,g,p,guard,lambda *_:None,steps=2)
    assert live==other and statehash(q,opt)==statehash(sq,so)
    # Exercise the real probability path with deterministic innovations and four
    # synthetic rewards; synthetic success is never a navigation result.
    q.sample=True;q.rng=np.random.default_rng(402);q.reset_noise();h=None
    with torch.no_grad():
        for row in x:_,_,h,_=q.step(row[None],h)
    aa=copy.deepcopy(a);aa.update({k:np.asarray([v[k] for v in q.tape]) for k in q.tape[0]});aa['stochastic'][:265]=0
    batch=[]
    for i in range(4):
        b=copy.deepcopy(aa);b['rewards'][-1]+=.1*i;batch.append(b)
    q.sample=False;sq.sample=False;live=L.fit(q,opt,batch,265,dict(p,epochs=2),refs,old,guard,lambda *_:None);other=L.fit(sq,so,batch,265,dict(p,epochs=2),L.groups(sq,p,[KNOWN_BRIDGE]),old,guard,lambda *_:None)
    assert live==other and statehash(q,opt)==statehash(sq,so);load(q,opt,initial);assert digest(state(q,opt))==digest(initial)
    # Actual independent CPU0..3 native setup/READY/cancel/EXIT code path. Zero
    # physical budget, not just a same-process tensor or mocked child test.
    pool=Pool(p,4,guard,'preflight_workers')
    try:
        values=pool.request([dict(cancel_probe=True) for _ in range(4)]);assert all(v['state']=='CANCEL_PROBE_PASS' and v['physics']==0 for v in values)
        childpids=[c.pid for c in pool.children]
    finally:pool.close()
    assert all(not Path('/proc',str(pid)).exists() for pid in childpids)
    try:
        import socket
        socket.create_connection(('example.com',443),timeout=.01);raise AssertionError('Offline network guard absent')
    except AssertionError as error:assert 'external network disabled' in str(error)
    # Cancellation of a future physical step is tested in every real worker.
    save(OUT/'preflight.json',dict(state='PASS',native=0,geometry=4,native_setup_commands=4,discarded_Adam=12,clones=dict(warm4=True,consolidation4=True,RL4=True),all_four_real_children_ready_cancel_exit=True,childpids=childpids,shadow_and_mature_rollback_exact=True,full_current_parameter_history=True,offline_external_socket_refused=True,elapsed_s=time.time()-begin))
    save(DOC/'LOCAL_READINESS.json',dict(state='PREFLIGHT_PASS_NOT_YET_LAUNCHED',offline=True,noLLM=True,no_network=True,no_install=True,all_worker_setup_probe_PASS=True,native_steps=0,optimizer_discarded=12,deadline='2026-11-07T08:30:00+08:00',source_sync_required_before_launch=True));print('T402_PREFLIGHT12_ADAM_8_GEOMETRY_ZERO_NATIVE_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
