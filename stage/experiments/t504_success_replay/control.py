"""Register current-policy multi-start credit pilot, no unlimited relaunches."""
from pathlib import Path
import copy,json,math,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t504_runtime as R
G,L,P,torch,np,read,save,sha,digest=R.G,R.L,R.P,R.torch,R.np,R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/review_20261112'

def keep_record(row):return {k:row[k] for k in ['job_id','path','sha256','success','collision']}
def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t503_fullfield_onpolicy';p=read(prior/'protocol.json')
    assert read(prior/'pilot/audit.json')['state']=='PASS';G.stopped(prior/'pilot')
    assert not Path('/proc',str(read(prior/'launch.json')['pid'])).exists()
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T503.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    result=read(prior/'pilot/summary.json')['result'];assert result['accepted'] and result['field_qualified_goals']==6
    rows=G.rows(prior/'pilot');replay=[r for r in rows if r['role']=='FRESH_NATIVE_FULLFIELD_TRAIN8' and r['job_id']=='t502_field_sector5_offset1' and r['success'] and not r['collision']]
    assert len(replay)==2
    field=[r for r in rows if r['role']=='AUTONOMOUS_FULLFIELD24_AFTER_RL' and r['success'] and not r['collision']];assert len(field)==6
    keep=[r for r in p['keep_rows'] if not r['job_id'].startswith('t502_')]+[keep_record(r) for r in field];assert len(keep)==385
    jobs=p['old8_jobs']+p['field_jobs'];physics=sum(j['case']['step_cap'] for j in p['old8_jobs'])+2*sum(j['case']['step_cap'] for j in p['field_jobs'])
    shape=sum(1352*math.ceil(j['case']['step_cap']/100)+4480+65536 for j in jobs);now=time.time()
    p.update(id='T504_TERMINAL_SUCCESS_LIKELIHOOD_REPLAY',out=str(OUT),parent_model=result['model'],initial=result['model'],initial_joint_locked=False,
             replay_rows=replay,keep_rows=keep,drive_initial_steps=912,steer_initial_steps=272,actor_lr=1e-4,keep_weight=1.,epochs=64,near_goal_radius_m=1.,target_kl=.005,
             initial_field_success_ids=sorted(r['job_id'] for r in field),max_bytes=48*2**20,preflight_commands=6,pilot_max_physics=physics,root_max_physics=physics,root_max_commands=66,root_max_main_seconds=2400,
             absolute_learning_stop=now+1800,absolute_native_stop=now+3600,absolute_all_stop=now+5400,
             Actor_Adam=64,shadow_Actor_Adam=64,audit_discarded_Actor_Adam=64,maximum_live_Actor_Adam=64,maximum_shadow_Actor_Adam=64,maximum_audit_discarded_Actor_Adam=64,preflight_discarded_Actor_Adam=4,Value_Adam=0,
             scientific_scope='Fixed two historical actual stochastic safeGOAL trajectories of remaining sector5offset1. Offpolicy censored-action likelihood supervised replay, not new policygradient nor exactSIL/AWR. Equal episodes; half completehistory mean plushalf mean at physicaldistance<=1m. Fixed64steps, no noise/seed/teacher/threshold/reward/time changes. Full recurrent history frozenfeaturecache.',
             optimizer_contract='Preserve T503 mature drive912/steer272 completeAdam moments; explicitly change RL3e-5 to replay1e-4. Backbone/head/Value/ValueAdam frozen; 11906adapterparams.',
             acceptance='Retain only if current6fieldsuccess IDs preserved, qualifiedfield24>6, contacts<=10, old8safe,0invalid/censor. Otherwise rollback entireActor/matureAdam. Full275/six54/eight54 not repeated and not claimed, originalT501 andT503 unchanged.',
             reference_contract='379 historical autonomous successful T501 trajectories plus6currentfield successes. Targets recomputed from CURRENTT503 initial untrimmed locations on completehistory, frozenfeaturecache. Historical old379 not fullclosedloop validated underT503. Four families equal/within equal;halfmean/halftop20risk. Softretention not proof.',
             authorization='Nov12 user explicitly requests next experiment after T503 completed report; independent changed objective replay pilot, no restart or extraT503 rollout budget.')
    for key in ['training_jobs','training_seeds','learning_gate']:p.pop(key,None)
    p['inputs']=dict(p['inputs']);p['protected_files']=dict(p['protected_files'])
    for f in [*HERE.glob('*.py'),DOC/'T504_PRIMARY_SOURCES.md',prior/'pilot/summary.json',prior/'pilot/audit.json',prior/'pilot/diagnosis.json',prior/'pilot/cases.jsonl',prior/'SEALED.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T503.json']:
        p['inputs'][str(f)]=sha(f)
    for row in keep+replay:p['inputs'][row['path']]=row['sha256']
    assert shape+14*2**20<p['max_bytes'] and G.M.used()+p['max_bytes']+2**20<2**30
    OUT.mkdir();obj=L.make(p);model=L.A.export(obj,OUT/'replay_initial.pt.gz',p['model_bytes']);assert model['policy_hash']==p['parent_model']['policy_hash'];p['initial']=model;p['initial_joint_locked']=True
    p['protected_files'][model['path']]=model['sha256'];p['inputs'][model['path']]=model['sha256'];assert digest(P.state(*L.make(p)))==model['joint_hash']
    blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    save(DOC/'T504_BUDGET.json',dict(max_physics=physics,max_commands=66,max_main_seconds=2400,learning_seconds=1200,audit_seconds=600,all_seconds=5400,backup_sync_reserve=1500,own_bytes=p['max_bytes'],shape_bound=shape,metadata_models_partial=14*2**20,global_reserved=G.M.used()-G.M.used(OUT)+p['max_bytes'],learning_stop=p['absolute_learning_stop'],native_stop=p['absolute_native_stop'],all_stop=p['absolute_all_stop'],CPU=[14,15],CPUQuota='100%',GPU=False,live_supervised_Actor_Adam=64,shadow_Actor_Adam=64,audit_discarded_Actor_Adam=64,preflight_discarded_Actor_Adam=4,Value_Adam=0,new_policy_gradient_updates=0))
    print('T504_REGISTERED_NOT_LAUNCHED',flush=True)


def preflight():
    p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'preflight.json').exists();obj=L.make(p);shadow=L.make(p);before=P.state(*obj);refs=L.references(p);data=L.training_data(p)
    with torch.no_grad():initial_keep=max(float((L.A.location(obj[0],g)-g['target']).abs().max()) for g in refs);assert initial_keep<1e-10
    aa=L.fit(obj,data,p,refs,lambda:None,lambda *_:None,steps=2);bb=L.fit(shadow,data,p,refs,lambda:None,lambda *_:None,steps=2)
    assert aa==bb and digest(P.state(*obj))==digest(P.state(*shadow));model=L.A.export(obj,OUT/'preflight_model.pt.gz',p['model_bytes']);assert digest(P.checkpoint(model['path'])['joint_state'])==model['joint_hash']
    for o in [obj,shadow]:P.load(o,before);assert digest(P.state(*o))==p['initial']['joint_hash']
    folder=OUT/'preflight_native';folder.mkdir();counts=G.R.M.mp.get_context('spawn').RawArray('q',3);pool=R.F.Pool(dict(p,phase='student',out=str(folder),native_stop_epoch=time.time()+120,max_physics=1),counts,lambda:None)
    try:
        vals=pool.request([dict(model=model['path'],job=j,seed=None,cancel_probe=True) for j in [p['field_jobs'][15],p['old8_jobs'][0]]]);assert all(v['state']=='CANCEL_PASS' for v in vals) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,commands=6,discarded_historical_replay_Actor_Adam=4,Value_Adam=0,historical_success_replay_NOT_new_native_or_policygradient=True,initial_keep_max_abs_error=initial_keep,mature912_272_and_explicit_lr_exact=True,live_shadow_and_fullAdam_rollback_exact=True,actual_success_arrays_and_near_goal_mask_verified=True,all_workers_stopped=True))
    print('T504_PREFLIGHT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');R.verify(p,True);pre=read(OUT/'preflight.json');assert pre['state']=='PASS' and not Path('/proc',str(pre['pid'])).exists();assert not (OUT/'launch.json').exists()
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert source['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time());assert seconds>5100
    args=['systemd-run','--user','--unit=stage-t504-success-replay','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T504','--why=Bounded successful fullfield history replay','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')]
    subprocess.run(args,check=True);print('T504_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
