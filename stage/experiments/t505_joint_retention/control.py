"""Register current-policy multi-start credit pilot, no unlimited relaunches."""
from pathlib import Path
import copy,json,math,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t505_runtime as R
G,L,P,torch,np,read,save,sha,digest=R.G,R.L,R.P,R.torch,R.np,R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/review_20261112'

def keep_record(row):return {k:row[k] for k in ['job_id','path','sha256','success','collision']}
def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t504_success_replay';p=read(prior/'protocol.json')
    assert read(prior/'pilot/audit.json')['state']=='PASS';G.stopped(prior/'pilot');assert not Path('/proc',str(read(prior/'launch.json')['pid'])).exists()
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T504.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    result=read(prior/'pilot/summary.json')['result'];assert not result['accepted'] and result['field_qualified_goals']==7 and result['old8']['success']==6
    rows=G.rows(prior/'pilot');keep=[r for r in rows if r['success'] and not r['collision']];assert len(keep)==13
    failed={r['job_id'] for r in rows[:8] if not r['success'] or r['collision']};assert failed=={'dev_task6_13','dev_task7_13'}
    previous=G.rows(ROOT/'stage/runs/t503_fullfield_onpolicy/pilot');repair=[r for r in previous if r['role']=='AUTONOMOUS_OLD8_SENTINELS' and r['job_id'] in failed];assert len(repair)==2 and all(r['success'] and not r['collision'] for r in repair)
    field=[r for r in keep if r['job_id'].startswith('t502_')];assert len(field)==7
    by={j['id']:j for j in p['field_jobs']};guard=p['old8_jobs']+[by[r['job_id']] for r in field];remaining=[j for j in p['field_jobs'] if j['id'] not in {r['job_id'] for r in field}];assert len(guard)==15 and len(remaining)==17
    physics=8*sum(j['case']['step_cap'] for j in guard)+sum(j['case']['step_cap'] for j in remaining)+sum(j['case']['step_cap'] for j in p['field_jobs'])
    full=guard*4+remaining;shape=sum(1352*math.ceil(j['case']['step_cap']/100)+4480+65536 for j in full);now=time.time();steps=result['Actor_Adam'];assert steps==49
    p.update(id='T505_JOINT_REPAIR_PER_ID_NATIVE_GUARD',out=str(OUT),parent_model=result['model'],initial=result['model'],initial_joint_locked=False,
             repair_rows=repair,keep_rows=keep,guard_jobs=guard,remaining_field_jobs=remaining,initial_protected_ids=sorted(r['job_id'] for r in keep),initial_field_success_ids=sorted(r['job_id'] for r in field),
             drive_initial_steps=912+steps,steer_initial_steps=272+steps,actor_lr=1e-4,max_blocks=4,block_steps=16,
             max_bytes=64*2**20,preflight_commands=6,pilot_max_physics=physics,root_max_physics=physics,root_max_commands=171,root_max_main_seconds=3600,
             absolute_learning_stop=now+3600,absolute_native_stop=now+4500,absolute_all_stop=now+6600,
             Actor_Adam=64,shadow_Actor_Adam=64,audit_discarded_Actor_Adam=64,maximum_live_Actor_Adam=64,maximum_shadow_Actor_Adam=64,maximum_audit_discarded_Actor_Adam=64,preflight_discarded_Actor_Adam=4,Value_Adam=0,
             scientific_scope='Explicit unpromotedT504 researchcopy for different jointrepair mechanism. Two actualT503successful oldhistories locationtargets,13actualT504successful ownhistories keep. Halfrepair/halfkeep normalizedunclippedlocation,halfphysicalmean/halftop20risk. 4fixedblocks16steps max, native15+15exact aftereveryblock. Firstrejection stops, no identicalrollbackretrain. No new policygradient/Value/teacher.',
             optimizer_contract='Preserve full matureT504 drive961/steer321 Adam, same1e-4lr,11906adapterparameters; freeze backbone/head/Value/ValueAdam.',
             acceptance='Eachblock retain everyprotectedactualsuccessID(initial13 plusnewrepaired),0invalid/censor andrepairlossdecreases;otherwise fullblockActorAdamrollback andstop. Provisionalretention notdeploymentpromotion. Onceold8all restored, remaining17 plus24exact;field7reused samepolicy guard,notindependentnewdata. Finalretain onlyold8safe+initialfield7safe+fieldqualified>=7/contact<=10/0invalidcensor;otherwisefullexperimentActorAdamrollback toexplicitresearchinitial. BestacceptedT503 andT501independentlyunchanged.',
             reference_contract='2 actual successful T503 autonomous oldcenters,13 actual successful T504 autonomous histories. Truefrozenrecurrentfeatures;locations storedunclipped. No teacher routing or claim allold275safe.',
             authorization='Nov12 user asks analyze how to pass all andstartnext. Distinct jointdata andperblockactualclosedloopretention,not restartT504 or extenditsreplaybudget.')
    for key in ['replay_rows','epochs','target_kl','near_goal_radius_m','keep_weight']:p.pop(key,None)
    p['inputs']=dict(p['inputs']);p['protected_files']=dict(p['protected_files'])
    for f in [*HERE.glob('*.py'),ROOT/'stage/docs/research_20261112_t505/PRIMARY_SOURCES.md',prior/'pilot/summary.json',prior/'pilot/audit.json',prior/'pilot/diagnosis.json',prior/'pilot/cases.jsonl',prior/'SEALED.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T504.json']:
        p['inputs'][str(f)]=sha(f)
    for row in keep+repair:p['inputs'][row['path']]=row['sha256']
    assert shape+14*2**20<p['max_bytes'] and G.M.used()+p['max_bytes']+2**20<2**30,(shape,G.M.used())
    OUT.mkdir();obj=L.make(p);model=L.A.export(obj,OUT/'research_initial.pt.gz',p['model_bytes']);assert model['policy_hash']==p['parent_model']['policy_hash'];p['initial']=model;p['initial_joint_locked']=True
    p['protected_files'][model['path']]=model['sha256'];p['inputs'][model['path']]=model['sha256'];assert digest(P.state(*L.make(p)))==model['joint_hash']
    blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    save(DOC/'T505_BUDGET.json',dict(max_physics=physics,max_commands=171,max_main_seconds=3600,learning_seconds=2400,audit_seconds=900,all_seconds=6600,backup_sync_reserve=1500,own_bytes=p['max_bytes'],shape_bound=shape,metadata_models_partial=14*2**20,global_reserved=G.M.used()-G.M.used(OUT)+p['max_bytes'],learning_stop=p['absolute_learning_stop'],native_stop=p['absolute_native_stop'],all_stop=p['absolute_all_stop'],CPU=[14,15],CPUQuota='100%',GPU=False,maximum_blocks=4,block_steps=16,live_supervised_Actor_Adam=64,shadow_Actor_Adam=64,audit_discarded_Actor_Adam=64,preflight_discarded_Actor_Adam=4,Value_Adam=0,new_policy_gradient_updates=0))
    print('T505_REGISTERED_NOT_LAUNCHED',flush=True)

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
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,commands=6,discarded_historical_replay_Actor_Adam=4,Value_Adam=0,joint_history_supervision_NOT_new_native_or_policygradient=True,initial_keep_max_abs_error=initial_keep,mature961_321_and_same_lr_exact=True,live_shadow_and_fullAdam_rollback_exact=True,actual_success_arrays_and_joint_targets_verified=True,all_workers_stopped=True))
    print('T505_PREFLIGHT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');R.verify(p,True);pre=read(OUT/'preflight.json');assert pre['state']=='PASS' and not Path('/proc',str(pre['pid'])).exists();assert not (OUT/'launch.json').exists()
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert source['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time());assert seconds>6200
    args=['systemd-run','--user','--unit=stage-t505-joint-retention','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T505','--why=Bounded joint repair and per-ID native retention','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')]
    subprocess.run(args,check=True);print('T505_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
