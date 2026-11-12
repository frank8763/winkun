"""Lock the offline night before the user leaves; 09:00 absolute all-stop."""
from pathlib import Path
import datetime,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t510_runtime as R
L,G,P,torch,np=R.L,R.G,R.P,R.torch,R.np
read,save,sha,digest=R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/overnight_20261112'
ARMS=['drive_only','steer_only','joint_keep_priority']
def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t509_two_goal_joint';p=read(prior/'protocol.json');assert read(prior/'pilot/audit.json')['state']=='PASS';G.stopped(prior/'pilot')
    assert not Path('/proc',str(read(prior/'launch.json')['pid'])).exists();assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T509.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    repair=[r for r in G.rows(prior/'pilot') if r['role']=='FRESH_TWO_GOAL_OWN_EXPLORATION8'];assert len(repair)==8 and all(r['success'] and not r['collision'] for r in repair)
    best=read(ROOT/'stage/runs/t508_frozen_completion/pilot/summary.json')['result']['model'];assert best['policy_hash']==p['initial']['policy_hash'] and best['joint_hash']==p['initial']['joint_hash']
    guard=p['guard_jobs'];rest=p['remaining_field_jobs'];physics=2*sum(j['case']['step_cap'] for j in guard)+sum(j['case']['step_cap'] for j in rest+p['field_jobs'])
    import math
    shape=sum(1352*math.ceil(j['case']['step_cap']/100)+4480+65536 for j in guard+rest)
    tz=datetime.timezone(datetime.timedelta(hours=8));end=datetime.datetime(2026,11,13,9,0,tzinfo=tz).timestamp();learn=end-3600;native=end-1800
    p.update(id='T510_OFFLINE_NIGHT_COMMAND_CREDIT',out=str(OUT),initial=best,parent_model=best,initial_joint_locked=True,repair_rows=repair,
             arms=ARMS,arm_bytes=28*2**20,max_bytes=52*2**20,arm_physics=physics,root_max_physics=3*physics,root_max_commands=249,root_max_main_seconds=10800,
             arm_main_seconds=3600,arm_learning_seconds=1200,arm_audit_seconds=600,backup_reserve_seconds=1200,preflight_commands=6,
             absolute_learning_stop=learn,absolute_native_stop=native,absolute_all_stop=end,fit_steps=32,actor_lr=1e-4,drive_initial_steps=1025,steer_initial_steps=385,
             Actor_Adam=96,shadow_Actor_Adam=96,audit_discarded_Actor_Adam=96,maximum_live_Actor_Adam=96,maximum_shadow_Actor_Adam=96,maximum_audit_discarded_Actor_Adam=96,preflight_discarded_Actor_Adam=12,Value_Adam=0,new_policy_gradient_Adam=0,
             scientific_scope='Threefixedindependentarms fromsameacceptedfield11:drive-only,steer-only,jointkeeppriorityone-sidedrawgradientprojection.Eachone32supervisedproposalwithmatureAdam,same8verifiedownGOALand19ownkeep.No extraexploration/grids/retry/rolloverfromfailedcandidate.21guard+21exact,conditional11+24exact. Perphasefullworstphysical/time/capacityadmission,skipreasonexplicit.',
             optimizer_contract='8parameterexistingAdam preserved. drive-only assignssteergradNoneandassertsallsteerparams/moments/countersimmutable;steer-onlysymmetrical. Jointrepairgradientprojectedagainstnegativekeepdotthensumhalf+half;rawprojectionnotactualAdamdisplacementguarantee;logboth. No newmodelcapacity.',
             acceptance='Everyarmprotected19IDs nativeGOAL/0invalidcensor+repairproxyimproves+newtask>=1 beforefullfield. Finalold8safealloriginal11retained,qualified>11/contact<=9;otherwisefullActorAdamrollback. Allarmsstartsameacceptedparent;selecthighestcompletefieldgoal,thencontact,thenfixedarmorder;notfreshholdout.',
             offline_contract='Allscience/audit/nextarmchoiceslocalpurePythonandinstalleddependencies,noLLM/network/install required. Networkbackup/sourceonlyafterallscienceended;networkfailureREMOTE_PENDINGneverblockpreprogrammedscience. No promisesfillclockwithduplicates. 08:00stoplearning08:30native09:00all.',
             authorization='Nov12useraskscheckandprepareovernight,explicitreplyNov13 09:00allstop. Finite3scientificallydifferentarms,notrepeatT509gridorcontinuefailedsamebudget.')
    for k in ['training_jobs','training_seeds','max_blocks','block_steps','learning_gate','retained18']:p.pop(k,None)
    # Keep only actual source dependencies and actual fixed input data; do not
    # claim unrelated historical datasets are current scientific inputs.
    p['inputs']={f:h for f,h in p['inputs'].items() if Path(f).suffix=='.py'}
    p['protected_files']={best['path']:best['sha256']}
    for f in [*HERE.glob('*.py'),DOC/'PLAN.md',DOC/'PRIMARY_SOURCES.md',prior/'pilot/cases.jsonl',prior/'pilot/summary.json',prior/'pilot/audit.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T509.json']:
        p['inputs'][str(f)]=sha(f)
    for r in repair+p['keep_rows']:p['inputs'][r['path']]=r['sha256']
    p['inputs'][best['path']]=best['sha256']
    assert shape+4*2**20<p['arm_bytes'] and G.M.used()+p['max_bytes']+512*1024<2**30,(shape,G.M.used())
    assert time.time()+3600+600+1200<end and physics*3<16_000_000
    OUT.mkdir();obj=L.make(p);assert digest(P.state(*obj))==best['joint_hash'];blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    save(DOC/'BUDGET.json',dict(max_physics=physics*3,max_commands=249,max_main_seconds=10800,own_root_bytes=p['max_bytes'],per_arm_bytes=p['arm_bytes'],per_arm_full_shape=shape,per_arm_models_metadata_partial=4*2**20,complete_global_reservation=G.M.used()-G.M.used(OUT)+p['max_bytes'],admission_uses_actual_previous_bytes_plus_complete_next_arm=True,per_arm_main=3600,per_arm_learning=1200,per_arm_audit=600,backup_reserve=1200,learning_stop=learn,native_stop=native,all_stop=end,arms=ARMS,live_supervised_Actor_Adam_max=96,shadow_Actor_Adam_max=96,audit_discarded_Actor_Adam_max=96,preflight_discarded_Actor_Adam=12,new_native_training=0,Value_Adam=0,CPU=[14,15],CPUQuota='100%',GPU=False));print('T510_NIGHT_REGISTERED_NOT_LAUNCHED',flush=True)
def preflight():
    p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'preflight.json').exists();data=L.groups(p);checks=[];q=L.make(p)[0]
    initial_keep=max(float((L.A.location(q,g)-g['target']).abs().max()) for g in data if g['kind']=='keep');assert initial_keep<1e-10
    for arm in p['arms']:
        pp=dict(p,arm=arm);obj=L.make(pp);shadow=L.make(pp);before=P.state(*obj)
        a=L.fit(obj,data,pp,lambda:None,lambda *_:None,steps=2);b=L.fit(shadow,data,pp,lambda:None,lambda *_:None,steps=2);assert a==b and digest(P.state(*obj))==digest(P.state(*shadow))
        model=L.A.export(obj,OUT/'preflight_model.pt.gz',p['model_bytes']);assert digest(P.checkpoint(model['path'])['joint_state'])==model['joint_hash']
        for o in [obj,shadow]:P.load(o,before);assert digest(P.state(*o))==p['initial']['joint_hash']
        checks.append(dict(arm=arm,cloned_discarded_Adam=4,live_shadow_exact=True,inactive_param_moments_preserved=a['frozen_inactive_parameters_and_mature_Adam_exact'],raw_and_Adam_displacement_diagnostic_finite=all(np.isfinite(float(v)) for t in a['trace'] for v in t.values())))
    folder=OUT/'preflight_native';folder.mkdir();counts=G.R.M.mp.get_context('spawn').RawArray('q',3);pool=R.F.Pool(dict(p,phase='student',out=str(folder),native_stop_epoch=time.time()+120,max_physics=1),counts,lambda:None)
    try:
        x=pool.request([dict(model=p['initial']['path'],job=j,seed=None,cancel_probe=True) for j in [p['guard_jobs'][-1],p['old8_jobs'][0]]]);assert all(y['state']=='CANCEL_PASS' for y in x) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,commands=6,discarded_historical_Actor_Adam=12,Value_Adam=0,initial_keep_max_abs_error=initial_keep,arms=checks,all_workers_stopped=True));print('T510_ALL_THREE_ARMS_PREFLIGHT_PASS',flush=True)
def launch():
    p=read(OUT/'protocol.json');R.verify(p,True);pre=read(OUT/'preflight.json');assert pre['state']=='PASS' and not Path('/proc',str(pre['pid'])).exists() and not (OUT/'launch.json').exists()
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert source['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time());assert seconds>5400
    args=['systemd-run','--user','--unit=stage-t510-offline-night','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T510','--why=Offline finite Stage night until09:00','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')]
    subprocess.run(args,check=True);print('T510_OFFLINE_NIGHT_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
