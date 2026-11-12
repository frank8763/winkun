"""Register current-policy multi-start credit pilot, no unlimited relaunches."""
from pathlib import Path
import copy,json,math,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t509_runtime as R
G,L,P,torch,np,read,save,sha,digest=R.G,R.L,R.P,R.torch,R.np,R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/review_20261112'

def keep_record(row):return {k:row[k] for k in ['job_id','path','sha256','success','collision']}
def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t508_frozen_completion';p=read(prior/'protocol.json');assert read(prior/'pilot/audit.json')['state']=='PASS';G.stopped(prior/'pilot')
    assert not Path('/proc',str(read(prior/'launch.json')['pid'])).exists();assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T508.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    result=read(prior/'pilot/summary.json')['result'];assert result['accepted'] and result['field_qualified_goals']==11 and result['old8']['success']==8
    fieldrows=read(prior/'pilot/combined_field24_rows.json');field=[r for r in fieldrows if r['success'] and not r['collision']];keep=p['retained18'][:8]+field;assert len(keep)==19 and all(r['loaded_model_sha256']==result['model']['sha256'] for r in keep)
    detail=read(prior/'pilot/diagnosis.json');failed=sorted([z for z in detail if z['timeout'] and not z['collision'] and z['min_decision_distance_m']<=1.],key=lambda z:(z['min_decision_distance_m'],z['id']));ids=[z['id'] for z in failed];assert ids==['t502_field_sector1_offset1','t502_field_sector0_offset2']
    by={j['id']:j for j in p['field_jobs']};training=[by[k] for k in ids for _ in range(4)];guard=p['old8_jobs']+[by[r['job_id']] for r in field]+[by[k] for k in ids];assert len(guard)==21
    remaining=[j for j in p['field_jobs'] if j['id'] not in {g['id'] for g in guard}];assert len(remaining)==11
    physics=2*sum(j['case']['step_cap'] for j in training)+4*sum(j['case']['step_cap'] for j in guard)+sum(j['case']['step_cap'] for j in remaining+p['field_jobs'])
    shape=sum(1352*math.ceil(j['case']['step_cap']/100)+4480+65536 for j in training+guard*2+remaining);now=time.time()
    p.update(id='T509_TWO_NEARGOAL_FRESH_SUCCESS_JOINT',out=str(OUT),initial=result['model'],parent_model=result['model'],initial_joint_locked=False,
             training_jobs=training,training_seeds=[[20261112509,i//4,i%4] for i in range(8)],new_task_ids=ids,keep_rows=keep,guard_jobs=guard,remaining_field_jobs=remaining,initial_protected_ids=sorted(r['job_id'] for r in keep),initial_field_success_ids=sorted(r['job_id'] for r in field),
             drive_initial_steps=1025,steer_initial_steps=385,actor_lr=1e-4,near_goal_radius_m=1.,max_blocks=2,block_steps=16,
             max_bytes=60*2**20,preflight_commands=6,pilot_max_physics=physics,root_max_physics=physics,root_max_commands=145,root_max_main_seconds=3600,
             absolute_learning_stop=now+3600,absolute_native_stop=now+4500,absolute_all_stop=now+6600,
             Actor_Adam=32,shadow_Actor_Adam=32,audit_discarded_Actor_Adam=32,maximum_live_Actor_Adam=32,maximum_shadow_Actor_Adam=32,maximum_audit_discarded_Actor_Adam=32,preflight_discarded_Actor_Adam=4,Value_Adam=0,
             max_live_Actor_Adam=32,max_shadow_Actor_Adam=32,max_audit_discarded_Actor_Adam=32,live_Adam=32,shadow_Adam=32,audit_discarded_Adam=32,new_policy_gradient_Adam=0,new_preflight_discarded_Adam=4,
             scientific_scope='Fixedtwo remainingnear-goaltimeouts fromacceptedfield11;4freshiidownpolicy.2noiseeach+8exact. Both musthave>=1qualifiedGOAL beforejoint supervisedexecuted-action BC(allqualifiedcollectedsuccesses,twotaskfamiliesequal)+actual19historykeep(old8/field11equal).Max2x16 with21native+21exact perblock;protect19IDsandnewsuccessID. No newPG/Value/teacher/reward orarchitecture change.',
             optimizer_contract='PreserveacceptedT507 maturedrive1025/steer385Adam and1e-4supervisorylr exactly,11906existingadapterparams;Value/backbone/head frozen.',
             loss_contract='Halfrepairhalfkeep;repairhalfphysicalmeanhalfgoal<=1mphysicalmean ofexecutedboundaryhinge peraxis sigma-normalized,twotasks equal/qualifiedepisode equal. keepold8/field11equal halfphysicalmeanhalftop20unclippedlocationrisk.',
             learning_gate='Bothfixedtasks eachatleast1safequalifiedGOAL from4ownpolicyindependentnoise,otherwise0Actorandnoevaluation. Usesallqualifiedsuccesses,dropsnooriginalrawfailures. Data-dependentimitationselectionnotunbiasedRL.',
             acceptance='Eachblock allprotectedIDs(original19+newlypassed) nativesafe,0invalidcensor andrepairlossdecreases;firstrejectionfullActorAdamrollbackandstop. Finalwhen>=1newretained:remaining11+24exact,old8safeall11fieldIDsretained,qualified>11contact<=9. OtherwisefullexperimentActorAdamrollback originalaccepted11untouched.',
             replay_contract='8fresh+8exact,thenatmost2x21guard+21exact,then11remaining+24exact.Samecandidateguardfield13reuse notindependentrepeat.Allphysicscounted,newtrainingseed1not8replicates.',
             conditional='No training ifeither task4explorationslackqualifiedsafeGOAL,firstguardfailurestop,no retry/noisegrid/newround.',
             authorization='Nov12usernextafterfullfield11verified;independentnewtwo-task data curriculum withjointselfsuccesslearning,notrestartingendedT508orT507.')
    p['inputs']=dict(p['inputs']);p['protected_files']=dict(p['protected_files']);p['protected_files'][result['model']['path']]=result['model']['sha256']
    for f in [*HERE.glob('*.py'),ROOT/'stage/docs/research_20261112_t509/PRIMARY_SOURCES.md',prior/'pilot/summary.json',prior/'pilot/audit.json',prior/'pilot/diagnosis.json',prior/'pilot/combined_field24_rows.json',prior/'SEALED.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T508.json']:
        p['inputs'][str(f)]=sha(f)
    for row in keep:p['inputs'][row['path']]=row['sha256']
    assert shape+14*2**20<p['max_bytes'] and G.M.used()+p['max_bytes']+2**20<2**30,(shape,G.M.used())
    OUT.mkdir();obj=L.make(p);model=L.A.export(obj,OUT/'joint_initial.pt.gz',p['model_bytes']);assert model['policy_hash']==p['parent_model']['policy_hash'] and model['joint_hash']==p['parent_model']['joint_hash'];p['initial']=model;p['initial_joint_locked']=True
    p['protected_files'][model['path']]=model['sha256'];p['inputs'][model['path']]=model['sha256'];assert digest(P.state(*L.make(p)))==model['joint_hash']
    blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    save(DOC/'T509_BUDGET.json',dict(max_physics=physics,max_commands=145,max_main_seconds=3600,learning_seconds=2400,audit_seconds=600,all_seconds=6600,backup_sync_reserve=1200,own_bytes=p['max_bytes'],shape_bound=shape,metadata_models_partial=14*2**20,global_reserved=G.M.used()-G.M.used(OUT)+p['max_bytes'],learning_stop=p['absolute_learning_stop'],native_stop=p['absolute_native_stop'],all_stop=p['absolute_all_stop'],CPU=[14,15],CPUQuota='100%',GPU=False,maximum_blocks=2,block_steps=16,live_supervised_Actor_Adam=32,shadow_Actor_Adam=32,audit_discarded_Actor_Adam=32,preflight_discarded_Actor_Adam=4,Value_Adam=0,new_policy_gradient_updates=0,new_training_native=8,exact_training=8))
    print('T509_REGISTERED_NOT_LAUNCHED',flush=True)

def fixture(p):
    q=L.make(p)[0].requires_grad_(False);data=[]
    for i,row in enumerate(p['keep_rows']):
        a=G.E.N.S.arrays(row['path']);g=L.B.cached(q,a);assert float((g['target']-torch.tensor(a['locations'])).abs().max())<1e-10
        g.update(kind='keep',family=int(row['job_id'].startswith('t502_')),id=row['job_id'],record_id='keep_'+str(i),dt=torch.tensor(a['native_counts'].astype(float)));data.append(g)
    for i in range(2):
        a=G.E.N.S.arrays(p['keep_rows'][-1]['path']);x=np.asarray(a['states'][:8]).copy();qq=L.make(p)[0];qq.sample=True;qq.rng=np.random.default_rng([20261112509,999,i]);h=None;actions=[]
        with torch.no_grad():
            for xx in x:
                action,loc,h,_=qq.step(torch.tensor(xx[None]),h);actions.append(action[0].numpy())
        g=L.B.cached(q,dict(states=x));loc=qq.sequence(torch.tensor(x));assert float((loc-torch.tensor(np.asarray([t['locations'] for t in qq.tape]))).abs().max())<1e-10
        raw=torch.tensor(np.asarray([t['raw_actions'] for t in qq.tape]));atom=torch.tensor(np.asarray([t['atom'] for t in qq.tape]));old=torch.tensor(np.asarray([t['logp'] for t in qq.tape]));assert float((P.L.density(loc,raw,atom,p['raw_sigma'])-old).abs().max())<1e-9
        g.update(kind='repair',family=i,id=p['new_task_ids'][i],record_id='repair_'+str(i),dt=torch.full((len(x),),100.,dtype=torch.float64),target=torch.tensor(np.asarray(actions),dtype=torch.float64),near=torch.ones(len(x),dtype=torch.bool));data.append(g)
    return data

def preflight():
    p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'preflight.json').exists();obj=L.make(p);shadow=L.make(p);before=P.state(*obj);data=fixture(p)
    initial_keep=max(float((L.A.location(obj[0],g)-g['target']).abs().max()) for g in data if g['kind']=='keep');assert initial_keep<1e-10
    aa=L.fit(obj,data,p,lambda:None,lambda *_:None,steps=2);bb=L.fit(shadow,data,p,lambda:None,lambda *_:None,steps=2)
    assert aa==bb and digest(P.state(*obj))==digest(P.state(*shadow));model=L.A.export(obj,OUT/'preflight_model.pt.gz',p['model_bytes']);assert digest(P.checkpoint(model['path'])['joint_state'])==model['joint_hash']
    for o in [obj,shadow]:P.load(o,before);assert digest(P.state(*o))==p['initial']['joint_hash']
    folder=OUT/'preflight_native';folder.mkdir();counts=G.R.M.mp.get_context('spawn').RawArray('q',3);pool=R.F.Pool(dict(p,phase='student',out=str(folder),native_stop_epoch=time.time()+120,max_physics=1),counts,lambda:None)
    try:
        vals=pool.request([dict(model=model['path'],job=j,seed=None,cancel_probe=True) for j in [p['training_jobs'][0],p['old8_jobs'][0]]]);assert all(v['state']=='CANCEL_PASS' for v in vals) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,commands=6,discarded_synthetic_Actor_Adam=4,Value_Adam=0,synthetic_NOT_navigation_data=True,initial_keep_max_abs_error=initial_keep,mature1025_385_unchanged_same_lr=True,live_shadow_ActorAdam_rollback_exact=True,all_workers_stopped=True));print('T509_PREFLIGHT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');R.verify(p,True);pre=read(OUT/'preflight.json');assert pre['state']=='PASS' and not Path('/proc',str(pre['pid'])).exists();assert not (OUT/'launch.json').exists()
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert source['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time());assert seconds>6200
    args=['systemd-run','--user','--unit=stage-t509-two-goal-joint','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T509','--why=Bounded two-goal exploration and joint success consolidation','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')]
    subprocess.run(args,check=True);print('T509_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
