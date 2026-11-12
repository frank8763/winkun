"""Lock the offline night before the user leaves; 09:00 absolute all-stop."""
from pathlib import Path
import datetime,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t511_runtime as R
L,G,P,torch,np=R.L,R.G,R.P,R.torch,R.np
read,save,sha,digest=R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/overnight_20261112'
def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t510_overnight_command_credit';p=read(prior/'protocol.json');seal=read(prior/'SEALED.json');assert all(not Path('/proc',str(pid)).exists() for pid in seal['scientific_pids']) and not Path('/proc',str(read(prior/'launch.json')['pid'])).exists()
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T510.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    parent=p['initial'];old=P.checkpoint(parent['path'])['joint_state'];assert digest(old)==parent['joint_hash']
    prioraccount=read(DOC/'FINAL_REPORT.json')['accounting'];assert prioraccount['physics']==2805558
    guard=p['guard_jobs'];rest=p['remaining_field_jobs'];physics=12*sum(j['case']['step_cap'] for j in guard)+sum(j['case']['step_cap'] for j in rest+p['field_jobs']);commands=6*42+11+24+4
    import math
    shape=lambda js:sum(1352*math.ceil(j['case']['step_cap']/100)+4480+65536 for j in js)
    reserve=shape(guard+rest)+4*2**20
    p.update(id='T511_FROZEN_PARENT_NEW_GLOBAL_RESIDUAL',out=str(OUT),parent_model=parent,initial=parent,initial_joint_locked=False,residual_seed=20261112511,actor_lr=3e-4,
             max_blocks=6,block_steps=32,max_bytes=40*2**20,root_max_physics=physics,root_max_commands=commands+6,root_max_main_seconds=14400,pilot_max_physics=physics,pilot_commands=commands,preflight_commands=6,
             next_block_full_reserve=reserve,next_block_time_reserve=1800,
             Actor_Adam=192,shadow_Actor_Adam=192,audit_discarded_Actor_Adam=192,maximum_live_Actor_Adam=192,maximum_shadow_Actor_Adam=192,maximum_audit_discarded_Actor_Adam=192,preflight_discarded_Actor_Adam=4,Value_Adam=0,new_policy_gradient_Adam=0,
             parent_Actor_Adam_hash=digest(old['actor_opt']),prior_night_accounting=prioraccount,night_combined_physics_limit=16000000,night_combined_commands_limit=600,night_combined_main_seconds_limit=21600,
             scientific_scope='Newmechanism frozenentireacceptedparent incloldadapters,newglobal184->64tanh->2 head finalweightsbiaszero,output .1tanh boundedresidual afterparentlocation clipmean[-1.25,1.25].11970newparams freshAdam0 only,newPG0/Value0. Same8realownGOAL+19ownkeep;max6x32 with21native+21exact/block firstrejectstop andfullnextblock+finaladmission. No runtime taskID/router/teacher/map/planner.',
             optimizer_contract='EntireparentActor frozenandhashchecked everynewheadstep. OriginalmatureAdam1025/385 remainsimmutableinparentfile,neverloadedintonew11970paramoptimizer. FreshnewheadAdam0 lr3e-4;Value/ValueAdam immutable. FullnewActorAdamrollback onreject/interrupt.',
             acceptance='Eachblock all19originalprotectedIDs+newlypassed IDs mustnativesafeGOAL/0invalidcensor andrepairproxydecrease,otherwisewholeblockrollbackandstop. New>=1retainedcandidate thenremaining11+24exact,old8safeoriginalfield11allretained qualifiedfield>11contact<=9. Originalbest11 remainsuntouched. Partialretainnotfinalpromotion.',
             capacity_contract='Rootown40MiB/global1GiB/free40;beforeeachnextblock reserve fullguard21+remaining11 shapes+4MiB modelpartialmetadata. Insufficientbudget skipremainingblocks explicitly, no deleting uniqueevidence or enlarging cap.',
             authorization='UserNov12explicitcontinueovernightafterT510finitearmsfailed,untilNov13 09:00samehardstop. DistinctcapacityandfrozenparentmechanismnotredoT510projection or additsrounds.')
    p.pop('arms',None)
    p['inputs']={f:h for f,h in p['inputs'].items() if Path(f).suffix=='.py'};p['protected_files']={parent['path']:parent['sha256']}
    for f in [*HERE.glob('*.py'),DOC/'CONTINUATION_PLAN.md',DOC/'CONTINUATION_PRIMARY_SOURCES.md',DOC/'PREFLIGHT_CACHE_CLEANUP.json',prior/'SEALED.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T510.json']:
        p['inputs'][str(f)]=sha(f)
    for row in p['repair_rows']+p['keep_rows']:p['inputs'][row['path']]=row['sha256']
    assert G.M.used()+p['max_bytes']+512*1024<2**30 and reserve+2*2**20<p['max_bytes']
    assert physics+prioraccount['physics']<16000000 and commands+6+prioraccount['commands']<=600 and 14400+prioraccount['main_seconds']<21600
    assert time.time()+14400+1200+1200<p['absolute_all_stop'] and time.time()+14400<p['absolute_native_stop']
    OUT.mkdir();obj=L.make(p);q=obj[0];parentq=L.A.frozen(p,parent['path']);maxdiff=0.
    with torch.no_grad():
        for row in p['keep_rows']+p['repair_rows']:
            x=torch.tensor(G.E.N.S.arrays(row['path'])['states']);a=q.sequence(x);b=parentq.sequence(x);assert torch.equal(a,b);maxdiff=max(maxdiff,float((a-b).abs().max()))
    assert not obj[2].state and digest(P.checkpoint(parent['path'])['joint_state']['actor_opt'])==p['parent_Actor_Adam_hash']
    model=L.export(obj,OUT/'expanded_initial.pt.gz',p['model_bytes']);p['initial']=model;p['initial_joint_locked']=True;p['protected_files'][model['path']]=model['sha256'];p['inputs'][model['path']]=model['sha256']
    assert digest(P.state(*L.make(p)))==model['joint_hash'];blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    save(OUT/'zero_initial_function.json',dict(state='EXACT_PARENT_FUNCTION_ON_ALL27_FULL_HISTORIES',max_location_error=maxdiff,newheadAdam0=True,old_parent_matureAdam_hash=p['parent_Actor_Adam_hash'],parentpolicyhash=parent['policy_hash'],expandedpolicyhash=model['policy_hash'],newnative=0))
    save(DOC/'CONTINUATION_BUDGET.json',dict(max_physics=physics,max_commands=commands+6,max_main_seconds=14400,learning_seconds=10800,audit_seconds=1200,backup_reserve=1200,own_bytes=p['max_bytes'],nextblock_full_reserve=reserve,shape_guard=shape(guard),shape_remaining=shape(rest),global_reserved=G.M.used()-G.M.used(OUT)+p['max_bytes'],night_combined_max_physics=physics+prioraccount['physics'],night_combined_max_commands=commands+6+prioraccount['commands'],night_combined_max_main_seconds=14400+prioraccount['main_seconds'],maximum_blocks=6,block_steps=32,live_newhead_Adam_max=192,shadow_Adam_max=192,audit_historical_Adam_max=192,preflight_discarded_Adam=4,Value_Adam=0,learning_stop=p['absolute_learning_stop'],native_stop=p['absolute_native_stop'],all_stop=p['absolute_all_stop'],CPU=[14,15],CPUQuota='100%',GPU=False));print('T511_FROZEN_PARENT_REGISTERED_NOT_LAUNCHED',flush=True)

def preflight():
    p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'preflight.json').exists();obj=L.make(p);shadow=L.make(p);before=P.state(*obj);data=L.groups(p)
    with torch.no_grad():initial_keep=max(float((L.location(obj[0],g)-g['target']).abs().max()) for g in data if g['kind']=='keep');assert initial_keep<1e-10
    a=L.fit(obj,data,p,lambda:None,lambda *_:None,steps=2);b=L.fit(shadow,data,p,lambda:None,lambda *_:None,steps=2);assert a==b and digest(P.state(*obj))==digest(P.state(*shadow))
    model=L.export(obj,OUT/'preflight_model.pt.gz',p['model_bytes']);assert digest(P.checkpoint(model['path'])['joint_state'])==model['joint_hash']
    for o in [obj,shadow]:P.load(o,before);assert digest(P.state(*o))==p['initial']['joint_hash']
    folder=OUT/'preflight_native';folder.mkdir();counts=G.R.M.mp.get_context('spawn').RawArray('q',3);pool=R.Pool(dict(p,phase='student',out=str(folder),native_stop_epoch=time.time()+120,max_physics=1),counts,lambda:None)
    try:
        v=pool.request([dict(model=model['path'],job=j,seed=None,cancel_probe=True) for j in [p['guard_jobs'][-1],p['old8_jobs'][0]]]);assert all(z['state']=='CANCEL_PASS' for z in v) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,commands=6,discarded_newhead_Adam=4,Value_Adam=0,initial_keep_error=initial_keep,exact_parent_zero_function=True,newhead_parameter_count=11970,entireparent_parameters_and_matureAdam_immutable=True,shadow_and_fullrollback_exact=True,actual_new_actor_workers_load_cancel_pass=True,all_workers_stopped=True));print('T511_NEW_HEAD_PREFLIGHT_PASS',flush=True)
def launch():
    p=read(OUT/'protocol.json');R.verify(p,True);pre=read(OUT/'preflight.json');assert pre['state']=='PASS' and not Path('/proc',str(pre['pid'])).exists() and not (OUT/'launch.json').exists()
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert source['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time());assert seconds>17000
    args=['systemd-run','--user','--unit=stage-t511-residual-continuation','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T511','--why=Offline frozen-parent Stage continuation until09:00','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')]
    subprocess.run(args,check=True);print('T511_OFFLINE_NIGHT_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
