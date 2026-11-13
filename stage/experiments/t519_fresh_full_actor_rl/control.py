"""Register current-policy multi-start credit pilot, no unlimited relaunches."""
from pathlib import Path
import copy,json,math,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t519_runtime as R
G,L,P,torch,np,read,save,sha,digest=R.G,R.L,R.P,R.torch,R.np,R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/day_20261113'

def keep_record(row):return {k:row[k] for k in ['job_id','path','sha256','success','collision']}
def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t518_active_tangent_recovery'
    assert read(prior/'fit/audit.json')['state']=='PASS'
    assert all(not Path('/proc',str(i)).exists() for i in read(prior/'SEALED.json')['scientific_pids']) and not Path('/proc',str(read(prior/'launch.json')['pid'])).exists()
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T518.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p=read(ROOT/'stage/runs/t509_two_goal_joint/protocol.json');target='t502_field_sector1_offset1';job=next(j for j in p['field_jobs'] if j['id']==target)
    evaluation=p['old8_jobs']+p['field_jobs'];physics=8*job['case']['step_cap']+2*sum(j['case']['step_cap'] for j in evaluation);now=time.time();parent=p['parent_model']
    p.update(id='T519_FRESH_FULL_ACTOR_CLOSED_LOOP_RL',out=str(OUT),initial=parent,parent_model=parent,initial_joint_locked=False,
        training_jobs=[job]*4,training_seeds=[[20261113519,i] for i in range(4)],evaluation_jobs=evaluation,epochs=16,actor_lr=3e-5,target_kl=.01,keep_weight=1.,
        max_bytes=9*2**20,model_bytes=3*2**20,preflight_commands=6,pilot_commands=76,pilot_max_physics=physics,root_max_physics=physics,root_max_commands=82,root_max_main_seconds=1800,
        absolute_learning_stop=now+1200,absolute_native_stop=now+2700,absolute_all_stop=now+5400,
        Actor_Adam=16,shadow_Actor_Adam=16,audit_discarded_Actor_Adam=16,preflight_discarded_Actor_Adam=4,Value_Adam=0,new_policy_gradient_Adam=16,
        scientific_scope='Onefreshactual4iidfullclosedloopepisodes+4exactfromaccepted11Actor ononefailedtask. Atleast1qualifiedGOAL beforemax16clippedpolicygradient fullActor updates+shadow;criticfreephysicaltime3peerRTGbaseline,originalreward/dynamics/sigma. AllcurrentfullhistoryBPTT numerator/density fromh0. No teacheractionsinPG or .002proxyhardgate.',
        optimizer_contract='Original8adapterparameterAdam1025/385momentsretained,addexistingencoder/core/headfreshAdamgroup0,explicit3e-5allgroups. WholeexistingActortrainable,noaddednetwork;Valueindependentseparateencoder/core/head andAdamfrozen/unused. Post-stepKL>.01stop onlynottrustguarantee. FullActorAdamrollbackifnativegatefails.',
        acceptance='Evaluatecandidatefullold8+field24all32nativeand32exactevenifoldIDsfail,compactaction/count/flagsstored/fullarrayshashescomparedbeforediscard. Acceptonlyold8safeall11fieldIDsretained/fieldqualified>11/contact<=9/0invalidcensor. OtherwisewholeActorAdambackoriginal. old275/six54/eight54/fresh16/78null.',
        evaluation_evidence='Allfull120history/hidden/noise/densitygeneratedinnativeworkers,firstandsecondfullarraysSHA/integrationexact;onlyactions/nativecounts/endflagscompactstoredforevaluation. IndependentofflineauditdoesNOTreconstructevaluationfull120,training4fullhistorydoes. No evidenceclaimbeyondthis.',
        authorization='Nov13userrequestsanalternateactualtrainingapproachafterofflinecompatibilitystalls. Researchskillprimarysourcesreview; boundedonebatchfullActorRLnotrepeatT516-518imitationconstraintornewcritic/SACclaim.')
    p['inputs']={f:h for f,h in p['inputs'].items() if Path(f).suffix=='.py'};p['protected_files']={parent['path']:parent['sha256']}
    for f in [*HERE.glob('*.py'),DOC/'T519_PLAN.md',DOC/'T519_PRIMARY_SOURCES.md',prior/'fit/summary.json',prior/'fit/audit.json',prior/'SEALED.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T518.json']:
        p['inputs'][str(f)]=sha(f)
    for row in p['keep_rows']:p['inputs'][row['path']]=row['sha256']
    p['inputs'].update(p['protected_files']);shape=4*(1352*math.ceil(job['case']['step_cap']/100)+4480+65536)
    assert G.M.used()+p['max_bytes']+128*1024<2**30
    OUT.mkdir();obj=L.make(p);new_parameters=[z for z in obj[0].parameters() if z not in obj[2].state]
    assert len(obj[2].param_groups)==2 and all(z.requires_grad for z in obj[0].parameters())
    model=L.A.export(obj,OUT/'rl_initial.pt.gz',p['model_bytes']);p['initial']=model;p['initial_joint_locked']=True;p['protected_files'][model['path']]=model['sha256'];p['inputs'][model['path']]=model['sha256']
    assert model['policy_hash']==parent['policy_hash'] and digest(P.state(*L.make(p)))==model['joint_hash']
    reserve=shape+32*p['compact_bytes']+p['model_bytes']+model['bytes']+1536*1024
    assert reserve<p['max_bytes'],(reserve,p['max_bytes'])
    blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    save(DOC/'T519_BUDGET.json',dict(max_physics=physics,max_commands=82,main_seconds=1800,learning_seconds=600,audit_seconds=600,backup_reserve=1200,all_stop=p['absolute_all_stop'],own_bytes=p['max_bytes'],full_training_shape=shape,all_data_model_partial_reserve=reserve,global_reserved=G.M.used()-G.M.used(OUT)+p['max_bytes'],trainable_Actor_parameters=sum(z.numel() for z in obj[0].parameters()),new_group_parameters=sum(z.numel() for z in new_parameters),max_live_PG=16,max_shadow_PG=16,max_audit_historical_Adam=16,preflight_discarded_max=4,ValueAdam=0,CPU=[14,15],CPUQuota='100%',GPU=False))
    print('T519_FULL_ACTOR_RL_REGISTERED_NOT_LAUNCHED',flush=True)

def preflight():
    p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'preflight.json').exists();obj=L.make(p);shadow=L.make(p);before=P.state(*obj);refs=L.references(p);arrays=[]
    source=G.E.N.S.arrays(p['keep_rows'][0]['path']);x=source['states'][:8]
    for i in range(4):
        q=L.make(p)[0].eval().requires_grad_(False);q.sample=True;q.rng=np.random.default_rng([20261113519,999,i]);h=None
        with torch.no_grad():
            for state in x:_,_,h,_=q.step(torch.tensor(state[None]),h)
        tape=q.tape;end=np.zeros((len(x),3),np.uint8);end[-1,1 if i%2==0 else 2]=1;reward=np.zeros(len(x));reward[-1]=1. if i%2==0 else -.5
        arrays.append(dict(states=x.copy(),next_states=np.concatenate([x[1:],x[-1:]]),ownership=np.zeros(len(x),np.uint8),stochastic=np.ones(len(x),np.uint8),end_flags=end,rewards=reward,native_counts=np.full(len(x),100),raw_actions=np.asarray([t['raw_actions'] for t in tape]),atom=np.asarray([t['atom'] for t in tape]),logp=np.asarray([t['logp'] for t in tape]),locations=np.asarray([t['locations'] for t in tape])))
    aa=L.fit(obj,arrays,p,refs,lambda:None,lambda *_:None,steps=2);bb=L.fit(shadow,arrays,p,refs,lambda:None,lambda *_:None,steps=2)
    assert aa==bb and digest(P.state(*obj))==digest(P.state(*shadow))
    assert any(not torch.equal(before['actor'][k],P.state(*obj)['actor'][k]) for k in before['actor'] if k.startswith(('encoder.','core.')))
    model=L.A.export(obj,OUT/'discarded_preflight_clone.pt.gz',p['model_bytes']);assert digest(P.checkpoint(model['path'])['joint_state'])==model['joint_hash']
    for o in [obj,shadow]:P.load(o,before);assert digest(P.state(*o))==p['initial']['joint_hash']
    folder=OUT/'preflight_native';folder.mkdir();counts=G.R.M.mp.get_context('spawn').RawArray('q',3)
    pool=R.F.Pool(dict(p,phase='student',out=str(folder),native_stop_epoch=time.time()+120,max_physics=1),counts,lambda:None)
    try:
        values=pool.request([dict(model=model['path'],job=j,seed=None,cancel_probe=True) for j in [p['training_jobs'][0],p['old8_jobs'][0]]]);assert all(v['state']=='CANCEL_PASS' for v in values) and sum(counts)==0
    finally:pool.close()
    # Serialization scratch is exactly reconstructible from this registered
    # synthetic fixture; not a scientific candidate or unique navigation data.
    clone_receipt=dict(path=model['path'],bytes=model['bytes'],sha256=model['sha256'],joint_hash=model['joint_hash'],recipe='preflight fixture seeds20261113519/999/i,actual2maxAdam;exactlive-shadow')
    Path(model['path']).unlink()
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,commands=6,discarded_synthetic_Actor_Adam=aa['Actor_Adam']+bb['Actor_Adam'],synthetic_fixture_NOT_navigation=True,full_actor_encoder_core_really_updates=True,independent_Value_weights_and_Adam_unchanged=True,live_shadow_ActorAdam_rollback_serialization_exact=True,all_worker_load_cancel_PASS=True,discarded_nonunique_serialization_scratch=clone_receipt))
    print('T519_FULL_ACTOR_ONPOLICY_PREFLIGHT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');R.verify(p,True);pre=read(OUT/'preflight.json');assert pre['state']=='PASS' and not Path('/proc',str(pre['pid'])).exists();assert not (OUT/'launch.json').exists()
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert source['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time());assert seconds>3800
    args=['systemd-run','--user','--unit=stage-t519-fresh-full-actor-rl','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T519','--why=Bounded fresh on-policy full-Actor navigation learning','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')]
    subprocess.run(args,check=True);print('T519_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
