"""Register current-policy multi-start credit pilot, no unlimited relaunches."""
from pathlib import Path
import copy,json,math,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t506_runtime as R
G,L,P,torch,np,read,save,sha,digest=R.G,R.L,R.P,R.torch,R.np,R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/review_20261112'

def keep_record(row):return {k:row[k] for k in ['job_id','path','sha256','success','collision']}
def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t505_joint_retention';p=read(prior/'protocol.json');assert read(prior/'pilot/audit.json')['state']=='PASS';G.stopped(prior/'pilot')
    assert not Path('/proc',str(read(prior/'launch.json')['pid'])).exists();assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T505.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    result=read(prior/'pilot/summary.json')['result'];assert result['accepted'] and result['old8']['success']==8 and result['field_qualified_goals']==7
    keep=[r for r in G.rows(prior/'pilot') if r['role']=='BLOCK_1_NATIVE_GUARD15'];assert len(keep)==15 and all(r['success'] and not r['collision'] and r['loaded_model_sha256']==result['model']['sha256'] for r in keep)
    detail=read(prior/'pilot/diagnosis.json');failed=sorted([d for d in detail if d['timeout'] and not d['collision'] and d['min_decision_distance_m']<=1.],key=lambda d:(d['min_decision_distance_m'],d['id']))
    selected=[d['id'] for d in failed[:3]];assert selected==['t502_field_sector7_offset1','t502_field_sector7_offset2','t502_field_sector0_offset1']
    by={j['id']:j for j in p['field_jobs']};training=[by[k] for k in selected for _ in range(4)];guard=p['guard_jobs'];remaining=p['remaining_field_jobs'];assert len(remaining)==17 and len(guard)==15
    physics=2*sum(j['case']['step_cap'] for j in training+guard)+sum(j['case']['step_cap'] for j in remaining+p['field_jobs'])
    shape=sum(1352*math.ceil(j['case']['step_cap']/100)+4480+65536 for j in training+guard+remaining);now=time.time()
    p.update(id='T506_NEARGOAL_FRESH_OWN_POLICY_CURRICULUM',out=str(OUT),parent_model=result['model'],initial=result['model'],initial_joint_locked=False,
             training_jobs=training,training_seeds=[[20261112506,i//4,i%4] for i in range(12)],selected_training_ids=selected,keep_rows=keep,
             drive_initial_steps=993,steer_initial_steps=353,actor_lr=3e-5,keep_weight=1.,epochs=16,target_kl=.005,
             max_bytes=64*2**20,preflight_commands=6,pilot_max_physics=physics,root_max_physics=physics,root_max_commands=105,root_max_main_seconds=3600,
             absolute_learning_stop=now+3000,absolute_native_stop=now+4500,absolute_all_stop=now+6600,
             Actor_Adam=16,shadow_Actor_Adam=16,audit_discarded_Actor_Adam=16,maximum_live_Actor_Adam=16,maximum_shadow_Actor_Adam=16,maximum_audit_discarded_Actor_Adam=16,preflight_discarded_Actor_Adam=4,Value_Adam=0,
             max_live_Actor_Adam=16,max_shadow_Actor_Adam=16,max_audit_discarded_Actor_Adam=16,live_Adam=16,shadow_Adam=16,audit_discarded_Adam=16,new_policy_gradient_Adam=16,new_preflight_discarded_Adam=4,preflight_discarded_Adam_max=4,preflight_discarded_Adam=4,
             scientific_scope='Freshfixed3nearestdecisionendpointmissed-goal timeouttasks,4iidownpolicyeach.12sample+12exact, at leastone qualifiedsafeGOAL before one max16clippedActorupdate withphysicaltime3peerRTGwithin each4/noCritic. Originalreward/gamma/sigma/time/geometry. NoSIL/BC/teacher/newreward. Guardold8+field7with15exact before remaining17+24exact; samepolicyfield7reuse notindependentrepeat.',
             loss_contract='Originalreward physicaltime peerRTGwithin4,discountedgamma^tcoefficients/pertaskRMS,clippedratio.8/1.2+ownhistorylocationkeep+actual15successkeep.Max16/targetKL.005poststop.Notclaimedunbiasedorvarianceproof.',
             replay_contract='12freshownpolicy episodes+12exact beforeoptimization;guard15+15exact thenremaining17+24exact;allrawarrays/integrationSHA andallphysicalrepetitions counted.',
             conditional='QualifiedGOALgate thenoneproposal then15protectednativeguard thenconditionalfullfield24;otherwiseboundedstopandrollback.',
             optimizer_contract='Preserve matureT505 drive993/steer353 fullAdam state, explicitlylowerlr1e-4 to3e-5 forRL;11906adapterparams,freezeencoder/core/head/ValueandValueAdam. WholeActorAdamrollback.',
             acceptance='Onecandidate; retainresearch only ifsamepolicy old8+field7allactualsafeGOAL thenfield24qualified>7,alloriginal7IDsretained,contacts<=10,0invalidcensor;otherwisefullActorAdamrollback. Fullold275/six54/eight54/fresh16notexecuted or claimed. BestT505andT501independentlylocked.',
             reference_contract='15actualacceptedT505successful completehistories,old8andfield7familiesequal,withinfamilyequal;halfmean/halftop20unclippednormalizedlocationrisk. These15notold379/275proof.',
             learning_gate='Atleastone trainingepisode rawqualifiedGOAL withnoobstaclecontact/nofieldboundary crossing;otherwise0ActorValueandnoevaluation. Data-dependentsuccessgate notunbiasedclaim. Noextra noise/seed/roundwhenfailed.',
             authorization='Nov12userexplicitcontinueafterT505acceptedjointold8field7. Independentnewtaskcurriculum ownpolicyRL,notrestartorcontinuefailedoldbudget.')
    for key in ['repair_rows','max_blocks','block_steps']:p.pop(key,None)
    p['inputs']=dict(p['inputs']);p['protected_files']=dict(p['protected_files']);p['protected_files'][result['model']['path']]=result['model']['sha256']
    for f in [*HERE.glob('*.py'),ROOT/'stage/docs/research_20261112_t506/PRIMARY_SOURCES.md',prior/'pilot/summary.json',prior/'pilot/audit.json',prior/'pilot/diagnosis.json',prior/'pilot/cases.jsonl',prior/'SEALED.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T505.json']:
        p['inputs'][str(f)]=sha(f)
    for row in keep:p['inputs'][row['path']]=row['sha256']
    assert shape+14*2**20<p['max_bytes'] and G.M.used()+p['max_bytes']+2**20<2**30,(shape,G.M.used())
    OUT.mkdir();obj=L.make(p);model=L.A.export(obj,OUT/'rl_initial.pt.gz',p['model_bytes']);assert model['policy_hash']==p['parent_model']['policy_hash'];p['initial']=model;p['initial_joint_locked']=True
    p['protected_files'][model['path']]=model['sha256'];p['inputs'][model['path']]=model['sha256'];assert digest(P.state(*L.make(p)))==model['joint_hash']
    blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    save(DOC/'T506_BUDGET.json',dict(max_physics=physics,max_commands=105,max_main_seconds=3600,learning_seconds=1800,audit_seconds=900,all_seconds=6600,backup_sync_reserve=1500,own_bytes=p['max_bytes'],shape_bound=shape,metadata_models_partial=14*2**20,global_reserved=G.M.used()-G.M.used(OUT)+p['max_bytes'],learning_stop=p['absolute_learning_stop'],native_stop=p['absolute_native_stop'],all_stop=p['absolute_all_stop'],CPU=[14,15],CPUQuota='100%',GPU=False,live_Actor_Adam=16,shadow_Actor_Adam=16,audit_discarded_Actor_Adam=16,preflight_discarded_Actor_Adam=4,Value_Adam=0,training_unique=12,exact_training=12))
    print('T506_REGISTERED_NOT_LAUNCHED',flush=True)

def fixture(p):
    row=p['keep_rows'][-1];a=G.E.N.S.arrays(row['path']);x=np.asarray(a['states'][:8]).copy();result=[]
    for i in range(12):
        q=L.make(p)[0];q.sample=True;q.rng=np.random.default_rng([20261112506,999,i]);h=None;acts=[]
        with torch.no_grad():
            for xx in x:action,loc,h,_=q.step(torch.tensor(xx[None]),h);acts.append(action[0].numpy())
        arrays=dict(states=x,next_states=np.concatenate([x[1:],x[-1:]]),actions=np.asarray(acts),native_counts=np.full(len(x),100,np.uint8),rewards=np.array([0.]*7+[1. if i%2==0 else -.5]),ownership=np.zeros(len(x),np.uint8),end_flags=np.array([[0,0,0]]*7+[[0,int(i%2==0),int(i%2==1)]],np.uint8))
        arrays.update({k:np.asarray([t[k] for t in q.tape]) for k in q.tape[0]});result.append(arrays)
    return result

def preflight():
    p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'preflight.json').exists();obj=L.make(p);shadow=L.make(p);before=P.state(*obj);refs=L.references(p);arrays=fixture(p)
    items,diagnostic=L.peer_batch(obj,arrays,p);assert any(torch.count_nonzero(b['adv']) for b in items)
    with torch.no_grad():initial_keep=max(float((L.A.location(obj[0],g)-g['target']).abs().max()) for g in refs);assert initial_keep<1e-10
    aa=L.fit(obj,arrays,p,refs,lambda:None,lambda *_:None,steps=2);bb=L.fit(shadow,arrays,p,refs,lambda:None,lambda *_:None,steps=2)
    assert aa==bb and digest(P.state(*obj))==digest(P.state(*shadow));model=L.A.export(obj,OUT/'preflight_model.pt.gz',p['model_bytes']);assert digest(P.checkpoint(model['path'])['joint_state'])==model['joint_hash']
    for o in [obj,shadow]:P.load(o,before);assert digest(P.state(*o))==p['initial']['joint_hash']
    folder=OUT/'preflight_native';folder.mkdir();counts=G.R.M.mp.get_context('spawn').RawArray('q',3);pool=R.F.Pool(dict(p,phase='student',out=str(folder),native_stop_epoch=time.time()+120,max_physics=1),counts,lambda:None)
    try:
        vals=pool.request([dict(model=model['path'],job=j,seed=None,cancel_probe=True) for j in [p['training_jobs'][0],p['old8_jobs'][0]]]);assert all(v['state']=='CANCEL_PASS' for v in vals) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,commands=6,discarded_synthetic_Actor_Adam=4,Value_Adam=0,synthetic_fixture_NOT_navigation_training=True,actual_raw_density_errors=[b['density_error'] for b in items],initial_keep_max_abs_error=initial_keep,mature993_353_and_explicit_lr_exact=True,live_shadow_and_fullAdam_rollback_exact=True,three_peer_physical_time_coefficient_finite=True,all_workers_stopped=True))
    print('T506_PREFLIGHT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');R.verify(p,True);pre=read(OUT/'preflight.json');assert pre['state']=='PASS' and not Path('/proc',str(pre['pid'])).exists();assert not (OUT/'launch.json').exists()
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert source['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time());assert seconds>6200
    args=['systemd-run','--user','--unit=stage-t506-neargoal-curriculum','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T506','--why=Bounded near-goal actual own-policy curriculum','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')]
    subprocess.run(args,check=True);print('T506_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
