"""Register current-policy multi-start credit pilot, no unlimited relaunches."""
from pathlib import Path
import copy,json,math,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t507_runtime as R
G,L,P,torch,np,read,save,sha,digest=R.G,R.L,R.P,R.torch,R.np,R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/review_20261112'

def keep_record(row):return {k:row[k] for k in ['job_id','path','sha256','success','collision']}
def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t506_neargoal_curriculum';p=read(prior/'protocol.json');assert read(prior/'pilot/audit.json')['state']=='PASS';G.stopped(prior/'pilot')
    assert not Path('/proc',str(read(prior/'launch.json')['pid'])).exists();assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T506.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    rr=G.rows(prior/'pilot');repair=[r for r in rr if r['role']=='FRESH_OWN_POLICY_NEARGOAL12'];assert len(repair)==12 and all(r['success'] and not r['collision'] and r['loaded_model_sha256']==p['initial']['sha256'] for r in repair)
    safe=ROOT/'stage/runs/t505_joint_retention';sr=read(safe/'pilot/summary.json')['result'];assert sr['accepted'] and sr['old8']['success']==8 and sr['field_qualified_goals']==7
    keep=[r for r in G.rows(safe/'pilot') if r['role']=='BLOCK_1_NATIVE_GUARD15'];assert len(keep)==15 and all(r['success'] and not r['collision'] for r in keep)
    newids=p['selected_training_ids'];assert len(newids)==3 and len(set(newids))==3
    by={j['id']:j for j in p['field_jobs']};guard=p['guard_jobs']+[by[k] for k in newids];assert len(guard)==18 and len({j['id'] for j in guard})==18
    remaining=[j for j in p['field_jobs'] if j['id'] not in {z['id'] for z in guard}];assert len(remaining)==14
    physics=6*sum(j['case']['step_cap'] for j in guard)+sum(j['case']['step_cap'] for j in remaining+p['field_jobs'])
    shape=sum(1352*math.ceil(j['case']['step_cap']/100)+4480+65536 for j in guard*3+remaining);now=time.time()
    p.update(id='T507_JOINT_VERIFIED_SUCCESS_TRANSFER',out=str(OUT),parent_model=sr['model'],initial=sr['model'],initial_joint_locked=False,
             repair_rows=repair,keep_rows=keep,guard_jobs=guard,remaining_field_jobs=remaining,new_task_ids=newids,initial_protected_ids=sorted(r['job_id'] for r in keep),
             drive_initial_steps=993,steer_initial_steps=353,actor_lr=1e-4,near_goal_radius_m=1.,max_blocks=3,block_steps=32,
             max_bytes=64*2**20,preflight_commands=6,pilot_max_physics=physics,root_max_physics=physics,root_max_commands=156,root_max_main_seconds=4200,
             absolute_learning_stop=now+4200,absolute_native_stop=now+5100,absolute_all_stop=now+7200,
             Actor_Adam=96,shadow_Actor_Adam=96,audit_discarded_Actor_Adam=96,maximum_live_Actor_Adam=96,maximum_shadow_Actor_Adam=96,maximum_audit_discarded_Actor_Adam=96,preflight_discarded_Actor_Adam=4,Value_Adam=0,
             max_live_Actor_Adam=96,max_shadow_Actor_Adam=96,max_audit_discarded_Actor_Adam=96,live_Adam=96,shadow_Adam=96,audit_discarded_Adam=96,new_policy_gradient_Adam=0,new_preflight_discarded_Adam=4,
             scientific_scope='AcceptedT505 start.12actualarchived ownpolicy safesuccesses collectedT506 under identicalActor,nowexplicitlyoldbehaviordata. Executed-action feasibleboundary hinge BC on3taskfamilies equal/4episodes each.15actualownsuccesshistory keepold8/field7. Halfrepair/halfkeep;repairhalfphysicalmean/halfphysicalmeanatdistance<=1m,keephalfphysicalmean/halftop20risk;3fixedblocks32max,18native+18exact each,firstrejectstop. NewPG0/Value0/noextrateacher.',
             optimizer_contract='Keep T505maturedrive993/steer353Adam fullmoments; explicit supervisorylr1e-4 matchesacceptedT505original (T506collectioninitialsameActorbutlr3e-5notused). Freezeallbackbone/head/Value/ValueAdam;11906adapterparams.No new network/capacity.',
             loss_contract='0.5repair(3tasks equal,4realepisodes each)+0.5keep(old8/field7twofamilies equal,within equal),peraxis sigma normalized:repairhalfphysicalmean+halfphysicalmeanatdistance<=1m;keephalfphysicalmean+halfupper20risk. repair executedaction boundaryhinge,keep originalunclippedlocations. Completehistory frozenfeaturecache;not stochasticdensity/currentPPO.',
             replay_contract='12oldbehavior successes reusedwithoriginalraw/integration/arraySHA,0newtrainingnative. Each18developmentnative+18exact allphysicalcounted,final14+24exact onlyifnewIDsafe andinitial15protected. Identicalpolicyguardfield10 reusednotindependentdata.',
             acceptance='Eachblockallinitial15protectedIDs safeplusnewlypassedIDsneverlost,repairlossdecreases,0invalidcensor;otherwisewholeblockActorAdamrollbackandstop. All18successstopslearningearly. Finalfullfield whennewatleast1retained:old8safe,originalfield7 retained,qualifiedfield>7/contact<=10/0invalidcensor. IfnotfinalpasswholeexperimentrollbacktoacceptedT505. OriginalT505/T501bestindependent.',
             reference_contract='12realstochasticGOALs sourceT506initialActorhash exactlyT505accepted.15realT505autonomoussuccesshistories. No actioninventing/smoothing/newteacher/newonpolicy claim.',
             conditional='Atleastone newlypassedID under retainedblockandinitial15safe beforeadditional14field and24exact. Otherwisefullfieldmissingnull. No seed/noise/epoch grid or failedbudgetcontinuation.',
             authorization='Nov12usernextstepafter12exploratoryGOALsandT506retentionfailure. Independentjointsupervisionwithperblocknativeguard,notrepeatfailedPPO orrestartoldjobs.')
    for key in ['training_jobs','training_seeds','learning_gate','target_kl','epochs']:p.pop(key,None)
    p['inputs']=dict(p['inputs']);p['protected_files']=dict(p['protected_files']);p['protected_files'][sr['model']['path']]=sr['model']['sha256']
    for f in [*HERE.glob('*.py'),ROOT/'stage/docs/research_20261112_t507/PRIMARY_SOURCES.md',prior/'pilot/summary.json',prior/'pilot/audit.json',prior/'pilot/cases.jsonl',prior/'SEALED.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T506.json',safe/'pilot/summary.json',safe/'pilot/audit.json']:
        p['inputs'][str(f)]=sha(f)
    for row in keep+repair:p['inputs'][row['path']]=row['sha256']
    assert shape+14*2**20<p['max_bytes'] and G.M.used()+p['max_bytes']+2**20<2**30,(shape,G.M.used())
    OUT.mkdir();obj=L.make(p);model=L.A.export(obj,OUT/'joint_initial.pt.gz',p['model_bytes']);assert model['policy_hash']==p['parent_model']['policy_hash'];p['initial']=model;p['initial_joint_locked']=True
    p['protected_files'][model['path']]=model['sha256'];p['inputs'][model['path']]=model['sha256'];assert digest(P.state(*L.make(p)))==model['joint_hash']
    blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    save(DOC/'T507_BUDGET.json',dict(max_physics=physics,max_commands=156,max_main_seconds=4200,learning_seconds=2700,audit_seconds=900,all_seconds=7200,backup_sync_reserve=1500,own_bytes=p['max_bytes'],shape_bound=shape,metadata_models_partial=14*2**20,global_reserved=G.M.used()-G.M.used(OUT)+p['max_bytes'],learning_stop=p['absolute_learning_stop'],native_stop=p['absolute_native_stop'],all_stop=p['absolute_all_stop'],CPU=[14,15],CPUQuota='100%',GPU=False,maximum_blocks=3,block_steps=32,live_supervised_Actor_Adam=96,shadow_Actor_Adam=96,audit_discarded_Actor_Adam=96,preflight_discarded_Actor_Adam=4,Value_Adam=0,new_policy_gradient_updates=0,new_training_native=0))
    print('T507_REGISTERED_NOT_LAUNCHED',flush=True)

def preflight():
    p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'preflight.json').exists();obj=L.make(p);shadow=L.make(p);before=P.state(*obj);data=L.groups(p)
    with torch.no_grad():initial_keep=max(float((L.A.location(obj[0],g)-g['target']).abs().max()) for g in data if g['kind']=='keep');assert initial_keep<1e-10
    aa=L.fit(obj,data,p,lambda:None,lambda *_:None,steps=2);bb=L.fit(shadow,data,p,lambda:None,lambda *_:None,steps=2)
    assert aa==bb and digest(P.state(*obj))==digest(P.state(*shadow));model=L.A.export(obj,OUT/'preflight_model.pt.gz',p['model_bytes']);assert digest(P.checkpoint(model['path'])['joint_state'])==model['joint_hash']
    for o in [obj,shadow]:P.load(o,before);assert digest(P.state(*o))==p['initial']['joint_hash']
    folder=OUT/'preflight_native';folder.mkdir();counts=G.R.M.mp.get_context('spawn').RawArray('q',3);pool=R.F.Pool(dict(p,phase='student',out=str(folder),native_stop_epoch=time.time()+120,max_physics=1),counts,lambda:None)
    try:
        vals=pool.request([dict(model=model['path'],job=j,seed=None,cancel_probe=True) for j in [p['field_jobs'][15],p['old8_jobs'][0]]]);assert all(v['state']=='CANCEL_PASS' for v in vals) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,commands=6,discarded_historical_replay_Actor_Adam=4,Value_Adam=0,verified_own_success_joint_transfer_NOT_new_native_or_policygradient=True,initial_keep_max_abs_error=initial_keep,mature993_353_and_supervisory_lr_exact=True,live_shadow_and_fullAdam_rollback_exact=True,actual_success_arrays_and_joint_targets_verified=True,all_workers_stopped=True))
    print('T507_PREFLIGHT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');R.verify(p,True);pre=read(OUT/'preflight.json');assert pre['state']=='PASS' and not Path('/proc',str(pre['pid'])).exists();assert not (OUT/'launch.json').exists()
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert source['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time());assert seconds>6800
    args=['systemd-run','--user','--unit=stage-t507-joint-success-transfer','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T507','--why=Bounded joint transfer of own-policy successful trajectories','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')]
    subprocess.run(args,check=True);print('T507_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
