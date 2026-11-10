"""Register and launch the explicitly renewed wide8 window, never resume old runs."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import copy,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t476_runtime as T
L,M,P,E,R,torch,np=T.L,T.M,T.P,T.E,T.R,T.torch,T.np
read,save,sha,digest=T.read,T.save,T.sha,T.digest
OUT,DOC=T.OUT,ROOT/'stage/docs/overnight_20261110'
def extend(j,d):
    j=copy.deepcopy(j);j.update(id=f"wide117_{d:g}m_{j['source_reset_id']}",radial_distance_m=d)
    j['case']['step_cap']=int(np.ceil(40000*d/6));j['case']['distance_m']=d;return j

def prepare():
    assert not OUT.exists(),'Never restart';source=ROOT/'stage/runs/t449_offline_joint_recovery';old=read(source/'protocol.json');result=next(v['candidate'] for v in read(source/'summary.json')['phases'] if v['phase']=='recovery');assert result['wide54']['success']==54 and result['old275']['success']==274
    stop=datetime(2026,11,11,9,0,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp();assert time.time()+6000<stop
    keys=['source_protocol','input_mean','input_scale','actor_seed','value_seed','raw_sigma','gamma_native','lambda_native','value_lr','target_kl']
    p={k:copy.deepcopy(old[k]) for k in keys};p.update(id='T476_WIDE8_FINITE_OFFLINE_RELAY',out=str(OUT),seed=20261110476,bc_lr=3e-5,actor_lr=3e-5,epochs=16,rounds=4,
        initial=result['model'],historical_best=result,old275_jobs=old['old275_jobs'],old8_jobs=old['old8_jobs'],six_jobs=old['main_jobs'],wide8_jobs=[extend(j,8.) for j in old['main_jobs']],
        max_bytes=50*2**20,full_bytes=640*1024,compact_bytes=4096,model_bytes=1792*1024,max_physics=15000000,max_commands=520,main_seconds=9000,learning_seconds=6000,audit_seconds=1200,
        root_max_physics=50000000,root_max_commands=1800,root_max_main_seconds=32400,
        absolute_learning_stop=stop-3600,absolute_native_stop=stop-1800,absolute_all_stop=stop,
        modes=['direct','curriculum','longcredit'],original_night_closed_physics=27172741,original_night_closed_commands=1892,
        renewed_window_not_expanding_old_budget=True,user_authorization='Nov10 latest: use best to start wide8, skip7; multiple distinct Stage lines untilNov11 09:00; autonomously research and modify. Supersedes previous Stage-stop and08:30 only for this new window.',
        inherited_known_failure='T449 dev_task6_00 contact; wide6 54/54 old274/275 is research best, not full six gate.',
        task='Same exact base6 initial integration and1.17m aperture; extend single final goal radially before initial observation. All4history goals replaced consistently. No runtime waypoint or target switch. capceil40000*d/6;8m53334native=106.668s, true task timeout.',
        mechanisms='direct physical-GAE; fixed6.5/7/7.5/8 finalgoal course; lambda_native1 completeepisode credit. Same initialActor/V and fresh optimizers. No BC/teacher/planner/runtimeQ.',
        course_advance='Accepted update, at least1/2 safe new training, both currentlevel deterministic centers safe; otherwise stay, max4rounds. After2consecutive retention-panel rejects endarm.',
        acceptance='Preserve all initial/current retained success IDs on fixed old8+six2+eight2 panel. Final383wide8/6/old assessed separately. Panel acceptance is not complete regression guarantee.',
        final_gate='Development pilot only: no fresh16/reload78 allocated; always report null, do not claim eightstable or sixgate. No automatic new experiment after finite3arms.',
        storage_budget='24full train640KiB=15MiB;12joint snapshots1792KiB=21MiB;1467compact4096B<5.74MiB;remaining8.26MiB forinitial/metadata/partial. Exact replays reuse original path only after digest equality. No uniqueevidence deletion.',
        no_Phase=True,preflight_commands=0)
    p['courses']={str(d):[extend(old['main_jobs'][i],d) for i in [13,40]] for d in [6.5,7.0,7.5,8.0]}
    p['panel_jobs']=p['old8_jobs']+[p['six_jobs'][i] for i in [13,40]]+[p['wide8_jobs'][i] for i in [13,40]]
    files={str(f) for f in HERE.glob('*.py')}|{str(DOC/'T476_PLAN.md'),str(DOC/'T476_PRIMARY_SOURCES.md'),str(source/'protocol.json'),str(source/'summary.json'),result['model']['path']}
    files|={j['exact_snapshot'] for j in p['six_jobs']+p['old275_jobs'] if j.get('exact_snapshot')}
    base=read(p['source_protocol']);files.add(p['source_protocol']);files|=set(base['input_hashes'])
    for module in list(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if f and str(Path(f).resolve()).startswith(str(ROOT/'stage/experiments')) and Path(f).suffix=='.py':files.add(str(Path(f).resolve()))
    p['inputs']={f:sha(f) for f in sorted(files)};p['protected_files']={result['model']['path']:result['model']['sha256'],**{j['exact_snapshot']:sha(j['exact_snapshot']) for j in p['six_jobs'] if j.get('exact_snapshot')}}
    assert M.used()+p['max_bytes']+256*1024<=2**30
    OUT.mkdir();obj=L.make(p);p['initial']=P.export(obj,OUT/'initial_fresh_optimizers.pt.gz',p['model_bytes']);assert p['initial']['policy_hash']==result['model']['policy_hash'];p['protected_files'][p['initial']['path']]=p['initial']['sha256']
    save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);T.verify(p,True)
    caps=sum(j['case']['step_cap'] for j in p['wide8_jobs']+p['six_jobs']+p['old275_jobs']);per=4*(4*53334+sum(j['case']['step_cap'] for j in p['panel_jobs'])+2*53334)+sum(j['case']['step_cap'] for j in p['panel_jobs'])+caps
    assert per<=15000000 and 3*per+54*53334<=50000000
    save(DOC/'T476_BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',global_reserved=M.used()-M.used(OUT)+p['max_bytes'],max_bytes=p['max_bytes'],worst_per_arm_physics=per,worst_root_physics=3*per+54*53334,root_max_physics=50000000,root_max_commands=1800,prior_closed_window_separate=True,all_stop_epoch=stop,full24training_and12models_reserved=True,additional_full_validation_not_allocated=True))
    print('T476_REGISTERED',flush=True)

def preflight():
    p=read(OUT/'protocol.json');T.verify(p,True);assert not (OUT/'preflight.json').exists();E.N.offline();folder=OUT/'preflight_native';folder.mkdir()
    module,native,create,w,_=T.native_setup(folder/'native',p);geometry=[]
    try:
        for j in [*p['wide8_jobs'],*[j for group in p['courses'].values() for j in group]]:
            s=T.Session(j,native,create,w,lambda *_:None,lambda:(_ for _ in()).throw(AssertionError('zero physical preflight')))
            x=s.packet();hp,hg,ht=w.get(s.env);assert np.max(np.abs(E.N.K.H.rebuild(x,np.asarray(s.env._pose()[:3]),hp,s.env.goal_xy,hg)-x))<=2e-6
            geometry.append(dict(id=j['id'],cap=s.cap,goal=s.env.goal_xy.tolist(),pose=list(s.env._pose()[:3]),distance=float(np.linalg.norm(s.env.goal_xy-np.array(s.env._pose()[:2]))),scene=s.scene,initial=digest(s.initial),input120=digest(x),all_history_goals_consistent=bool(np.all(hg==s.env.goal_xy)),start_collision=False));s.close()
    finally:w.close()
    save(OUT/'geometry.json',geometry)
    # Actual CPU recurrent policy arrays provide density/gradient fixture only.
    fixtures=[]
    for i in [0,1]:
        q=P.frozen(p,p['initial']['path']);q.sample=True;q.rng=np.random.default_rng([p['seed'],999,i]);x=np.tile(np.array(p['input_mean'],np.float32),(5,1));h=None;actions=[]
        for v in x:
            with torch.no_grad():a,loc,h,_=q.step(torch.tensor(v[None]),h)
            actions.append(a[0].numpy())
        a=dict(states=x,next_states=x.copy(),actions=np.array(actions),rewards=np.array([0.,.1,-.1,.02,1.]),native_counts=np.full(5,100,np.uint8),ownership=np.zeros(5,np.uint8),end_flags=np.array([[0,0,0]]*4+[[0,1,0]],np.uint8))
        a.update({k:np.asarray([v[k] for v in q.tape]) for k in q.tape[0]});fixtures.append(a)
    modes={};total=0
    for mode in p['modes']:
        pp=dict(p,epochs=1,lambda_native=1. if mode=='longcredit' else p['lambda_native']);obj=L.make(pp);shadow=L.make(pp);before=P.state(*obj)
        aa=L.fit(obj,fixtures,pp,lambda:None,lambda *_:None);bb=L.fit(shadow,fixtures,pp,lambda:None,lambda *_:None);assert aa==bb and digest(P.state(*obj))==digest(P.state(*shadow));P.load(obj,before);P.load(shadow,before);assert digest(P.state(*obj))==p['initial']['joint_hash']==digest(P.state(*shadow));modes[mode]=aa;total+=4
    counts=R.M.mp.get_context('spawn').RawArray('q',3);pool=T.Pool(dict(p,out=str(folder),native_stop_epoch=time.time()+180,max_physics=1),counts,lambda:None)
    try:reply=pool.request([dict(model=p['initial']['path'],job=j,seed=None,cancel_probe=True) for j in [p['wide8_jobs'][13],p['wide8_jobs'][40]]]);assert all(v['state']=='CANCEL_PASS' for v in reply) and sum(counts)==0
    finally:pool.close()
    commands=len(geometry)+6;save(OUT/'preflight.json',dict(state='PASS',native=0,zero_step_geometry=len(geometry),goal_reachability_NOT_proven=True,commands=commands,discarded_synthetic_Actor_and_Value_Adam=total,modes=modes,two_worker_native_cancellation=True,full_Actor_V_Adam_shadow_rollback_exact=True,fixture_NOT_navigation_success=True))
    print('T476_PREFLIGHT_PASS',commands,flush=True)

def launch():
    p=read(OUT/'protocol.json');T.verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists();src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip() and not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    now=time.time();assert now+6000<p['absolute_all_stop']
    with (OUT/'driver_console.log').open('xb') as f:
        c=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T476','--why=Finite offlinewide8 untilNov11 09:00','timeout','--signal=TERM','--kill-after=2',str(int(p['absolute_all_stop']-now)),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=c.pid,epoch=now,all_stop=p['absolute_all_stop'],source_remote=src['remote_commit'],protocol_sha256=sha(OUT/'protocol.json'),offline_no_LLM_network_install=True));print('T476_ACTUALLY_LAUNCHED',c.pid,flush=True)
if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        if OUT.exists():save(OUT/f'error_control_{sys.argv[1]}_{os.getpid()}.json',dict(error=traceback.format_exc(),epoch=time.time()))
        raise
