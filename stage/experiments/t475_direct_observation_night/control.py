"""One final bounded Stage attempt; no automatic continuation after failure."""
from pathlib import Path
import copy,json,os,shutil,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t475_direct_observation_night';DOC=ROOT/'stage/docs/overnight_20261110'
sys.path.insert(0,str(HERE))
import t475_runtime as T
L,M,P,E,R,torch,np=T.L,T.M,T.P,T.E,T.R,T.torch,T.np
read,save,sha,digest=T.read,T.save,T.sha,T.digest

def prepare():
    assert not OUT.exists(),'Neverrestart'
    source=ROOT/'stage/runs/t474_verified_joint_night';old=read(source/'protocol.json');report=read(DOC/'T474_FINAL_REPORT.json')
    assert report['state']=='COMPLETE_AUDITED_LOCAL' and report['remote']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED' and not report['six_gate']
    assert all(not Path('/proc',str(v)).exists() for v in read(source/'SEALED.json')['scientific_pids'])
    for f in ['launch.json','driver_started.json']:assert not Path('/proc',str(read(source/f)['pid'])).exists()
    p=copy.deepcopy(old['science']);raw=read(source/'verified/audit.json');assert raw['state']=='PASS'
    p.update(id='T475_FINAL_DIRECT_OBSERVATION_REPAIR',out=str(OUT),mode='direct',verified_raw15=raw['verified_raw15'],
        observation_seed=20261110475,observation_lr=3e-5,epochs=512,
        frozen327_actor_digest=digest(P.checkpoint(p['learning_initial']['path'])['residual_state']['actor']),
        max_bytes=56*2**20,main_partition_bytes=13*2**20,gate_partition_bytes=43*2**20,
        max_physics=8000000,max_commands=333,preflight_commands=6,worker_count=2,
        model_bytes=1024*1024,full_bytes=448*1024,compact_bytes=4096,main_seconds=4200,learning_seconds=1200,audit_seconds=1200,
        overall_physics_max=11000000,overall_commands_max=441,gate_commands_max=102,gate_max_physics=3600000,
        gate_main_seconds=2400,gate_audit_seconds=600,backup_seconds=900,all_seconds=10800,
        absolute_learning_stop=old['absolute_learning_stop'],absolute_native_stop=old['absolute_native_stop'],absolute_all_stop=old['absolute_all_stop'],
        night_prior_physics=report['night_physics'],night_prior_commands=report['night_commands'],night_prior_main_seconds=old['night_prior_main_seconds']+report['main_elapsed_s'],
        night_max_physics=40000000,night_max_commands=2000,night_max_main_seconds=14400,
        original_retained=old['initial_candidate'],
        mechanism='Freeze all53940 original327 parameters; add3938 new currentnormalized120->32tanh->2logit path, lastlayerzero, originalGRU64carry stays causal. No taskID/router/planner/teacher atdeployment. Newpath hasfreshAdam, originalActorAdamarchiveunchanged.',
        learning_semantics='Onefixed512BC+512shadow using same4verifiedmixedrecoveries andfinitepreservation asT474; raw15onlyexecutedownershipsuffix labels. Newarchitecture/mask/freshoptimizer/stepbudget jointlychanged, not capacitycausalproof or novelRL.',
        decision='Onlysamecandidate329allSAFE permits16fresh then78newmodelpath/new2workersnative fullarrayexact. Noadaptationafterfresh. FailedoriginalsuccessIDsrejectmodel. Ifstrictsixgatefails, stopStage line; do not prepare/launch furtherexperiments or8m.',
        budget='Existingnightonly: remaining447commands, thisfinalattempt441including6pref+333full329+102conditionalgates. Two workers are serial admission/cost change, same exactnativeprotocol. Noextra seed/grid/teacher/round.',
        user_authorization='Nov10 user: stop repeatedchecking/waste, continue finalattempt, ifstillnotthrough abandonthisStage line. SameconfirmedNov11 08:30absolutehardstop. Noautomaticnewnumberafterthis.',
        final_attempt=True,stop_stage_if_not_passed=True,eight_started=False)
    p['worst_main_physics']=sum(j['case']['step_cap'] for j in p['main_jobs']+p['old275_jobs'])
    assert p['worst_main_physics']<=p['max_physics'] and p['worst_main_physics']+3580002<=p['overall_physics_max']
    assert p['night_prior_commands']==1553 and p['night_prior_commands']+441<=2000
    assert p['night_prior_physics']+p['overall_physics_max']<=40000000
    assert p['night_prior_main_seconds']+p['main_seconds']+p['gate_main_seconds']<=14400
    assert time.time()+p['all_seconds']<p['absolute_all_stop'] and time.time()+4200+1200+2400<p['absolute_native_stop']
    assert M.used()+p['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    files=set(old['inputs'])|{str(f) for f in HERE.glob('*.py')}|{str(DOC/'T475_PLAN.md'),str(DOC/'T474_FINAL_REPORT.json'),str(source/'summary.json'),str(source/'verified/summary.json'),str(source/'verified/audit.json'),p['verified_raw15']['path'],str(source/'SEALED.json'),str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T474.json')}
    p['inputs']={f:sha(f) for f in sorted(files)};p['constructor_reads']=sorted(files)
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);T.verify(p,True)
    save(DOC/'T475_BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',root_bytes=p['max_bytes'],full_global=M.used()-M.used(OUT)+p['max_bytes'],max_physics=p['overall_physics_max'],worst_physics=p['worst_main_physics']+3580002,actual_main333_pref6_gate102_total441=True,night_prior_commands=p['night_prior_commands'],night_max_actual_commands=1994,night_command_limit=2000,main_seconds4200_audit1200_gate2400_gateaudit600=True,absolute_all_stop=p['absolute_all_stop'],last_attempt_no_more_experiments=True))
    print('T475_REGISTERED',flush=True)

def preflight():
    p=read(OUT/'protocol.json');T.verify(p,True);assert not (OUT/'preflight.json').exists();q,op=L.make(p);shadow=L.make(p);initial=L.state(q,op);anchor=L.B.frozen(p,p['learning_initial']['path'])
    CPU=L.Actor(p).eval();maxerr=0.
    for row in p['seed_occupancy_rows']:
        a=E.N.S.arrays(row['path']);x=torch.tensor(a['states'])
        with torch.no_grad():
            loc=CPU.sequence(x);base=anchor.sequence(x);assert torch.equal(loc,base)
            hidden=None
            for i,value in enumerate(x):
                command,point,hidden,_=CPU.step(value[None],hidden)
                assert np.array_equal(command[0].numpy(),a['actions'][i]) and np.array_equal((np.zeros(64) if i==0 else prior_hidden),a['hidden_before'][i]);prior_hidden=hidden[0,0].numpy().copy()
                maxerr=max(maxerr,float((point[0]-base[i]).abs().max()))
            assert np.array_equal(hidden[0,0].numpy(),a['last_hidden_after'])
    assert maxerr<1e-10
    batch=L.data(p);a=L.step((q,op),batch);b=L.step(shadow,batch);assert a==b and digest(L.state(q,op))==digest(L.state(*shadow))
    assert a['reference']<1e-24 and a['own']<1e-24 and a['extra']==0
    model=L.export((q,op),OUT/'preflight_trained.pt.gz',p['model_bytes']);assert L.frozen(p,model['path']).phash()==q.phash()
    L.load((q,op),initial);L.load(shadow,initial);assert digest(L.state(q,op))==digest(initial)==digest(L.state(*shadow));zero=L.export((q,op),OUT/'zero_initialized.pt.gz',p['model_bytes'])
    counts=R.M.mp.get_context('spawn').RawArray('q',3);folder=OUT/'preflight_native';folder.mkdir();pool=T.Pool(dict(p,out=str(folder),native_stop_epoch=time.time()+180,max_physics=1),counts,lambda:None)
    try:
        result=pool.request([dict(model=zero['path'],job=j,seed=None,cancel_probe=True) for j in p['old8_jobs'][:2]]);assert all(r['state']=='CANCEL_PASS' for r in result) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',native=0,commands=6,discarded_new_Adam=2,zero_initial_full6_actual_history_actions_hidden_exact=True,sequence_step_error=maxerr,full_frozen327_digest_unchanged=True,new3938parameters=True,actual_shadow_serialization_and_rollback_exact=True,two_worker_cancel_before_physics=True))
    print('T475_PREFLIGHT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');T.verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip() and not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    now=time.time();assert now+p['all_seconds']<p['absolute_all_stop']
    with (OUT/'driver_console.log').open('xb') as f:
        c=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T475','--why=Final bounded Stage attempt until08:30, no retry',
            'timeout','--signal=TERM','--kill-after=2',str(min(p['all_seconds'],int(p['absolute_all_stop']-now))),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=c.pid,epoch=now,all_stop=min(now+p['all_seconds'],p['absolute_all_stop']),source_remote=src['remote_commit'],protocol_sha256=sha(OUT/'protocol.json'),offline_no_LLM_or_network_needed=True,last_attempt_no_automatic_continuation=True));print('T475_ACTUALLY_LAUNCHED',c.pid,flush=True)
if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        if OUT.exists():save(OUT/f'error_control_{sys.argv[1]}_{os.getpid()}.json',dict(error=traceback.format_exc(),epoch=time.time()))
        raise
