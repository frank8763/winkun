"""Register current-policy multi-start credit pilot, no unlimited relaunches."""
from pathlib import Path
import copy,json,math,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t520_runtime as R
G,L,P,torch,np,read,save,sha,digest=R.G,R.L,R.P,R.torch,R.np,R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/day_20261113'

def keep_record(row):return {k:row[k] for k in ['job_id','path','sha256','success','collision']}
def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t519_fresh_full_actor_rl'
    assert read(prior/'pilot/audit.json')['state']=='PASS'
    assert all(not Path('/proc',str(i)).exists() for i in read(prior/'SEALED.json')['scientific_pids']) and not Path('/proc',str(read(prior/'launch.json')['pid'])).exists()
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T519.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    previous=read(prior/'pilot/summary.json')['result'];assert not previous['accepted'] and previous['field_qualified_goals']==12 and previous['old8']['success']==7
    p=read(prior/'protocol.json');locked=p['parent_model'];parent=previous['model'];target='dev_task6_13';job=next(j for j in p['old8_jobs'] if j['id']==target)
    evaluation=p['old8_jobs']+p['field_jobs'];physics=8*job['case']['step_cap']+2*sum(j['case']['step_cap'] for j in evaluation);now=time.time()
    p.update(id='T520_RESEARCH12_OLD_CENTER_REAL_RL_REPAIR',out=str(OUT),initial=parent,parent_model=parent,locked_best=locked,initial_joint_locked=True,
        research_initial_NOT_promoted=True,initial_field_success_ids=previous['field_success_ids'],training_jobs=[job]*4,training_seeds=[[20261113520,i] for i in range(4)],evaluation_jobs=evaluation,
        epochs=16,actor_lr=3e-5,target_kl=.01,keep_weight=1.,max_bytes=6*2**20,model_bytes=2*2**20,
        preflight_commands=6,pilot_commands=76,pilot_max_physics=physics,root_max_physics=physics,root_max_commands=82,root_max_main_seconds=1800,
        absolute_learning_stop=now+1200,absolute_native_stop=now+2700,absolute_all_stop=now+5400,
        scientific_scope='ExplicitunpromotedT519field12/old7candidate andentireActorAdamresearchinitial. Oldcenterdev_task6_13 only4freshiidfromoriginalstart+4exact,atleast1qualifiedGOAL beforemax16fullActorPG+shadow. Softhistorical19inputtargets refreshedtoTHISinitialfunction,notnewverifiedsuccesses/teacher;all12fieldnativeIDsretainedbyfinalbehaviorgate. No newarchitecture/critic/reward/thresholdgrid.',
        optimizer_contract='ReusecompleteT519ActorandbothAdamgroups unchanged at3e-5. Expectedadaptersteps1041/401,newencoder/core/head16;checkactual source, do notresetmoments or appendanothergroup. SeparateValueandAdamimmutable/unused. RejectedproposalentireActorAdambackT519researchinitial;acceptedbest11sourceindependentlyprotected.',
        acceptance='Samecandidateold8all8safeGOAL,all12T519fieldsuccessIDsretained/qualified>=12/contact<=9/0invalidcensor. Full32native+32exactallcompletedsubjecthardbudgets. OtherwisewholeActorAdamrollbackresearchinitial,no bestpromotion. old275/six54/eight54/fresh16/reload78null.',
        authorization='Nov13user explicitlynextstart afterreviewanddraft. Newtargetedold-centerrestorationresearchfromT519notrestart/extendoldrun. OriginalacceptedT507model remainsunchanged.')
    p['inputs']={f:h for f,h in p['inputs'].items() if Path(f).suffix=='.py'};p['protected_files']={parent['path']:parent['sha256'],locked['path']:locked['sha256']}
    for f in [*HERE.glob('*.py'),DOC/'T520_PLAN.md',DOC/'T520_PRIMARY_SOURCES.md',prior/'protocol.json',prior/'pilot/summary.json',prior/'pilot/audit.json',prior/'SEALED.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T519.json',DOC/'T519_REVIEW_AND_NEXT.json']:
        p['inputs'][str(f)]=sha(f)
    for row in p['keep_rows']:p['inputs'][row['path']]=row['sha256']
    p['inputs'].update(p['protected_files']);shape=4*(1352*math.ceil(job['case']['step_cap']/100)+4480+65536)
    global_before=G.M.used();assert global_before+p['max_bytes']+64*1024<2**30
    obj=L.make(p);state=P.checkpoint(parent['path'])['joint_state'];assert digest(P.state(*obj))==parent['joint_hash'] and digest(P.state(*obj))==digest(state)
    assert len(obj[2].param_groups)==2 and all(g['lr']==3e-5 for g in obj[2].param_groups)
    group_steps=[[int(obj[2].state[z]['step']) for z in g['params']] for g in obj[2].param_groups]
    assert group_steps[0]==[1041]*4+[401]*4 and set(group_steps[1])=={16}
    reserve=shape+32*p['compact_bytes']+p['model_bytes']+1536*1024
    assert reserve<p['max_bytes'],(reserve,p['max_bytes'])
    OUT.mkdir();blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    save(OUT/'initial_model_contract.json',dict(state='EXACT_T519_RESEARCH_ACTOR_AND_ALL_ADAM_GROUPS',initial=parent,accepted_best_locked=locked,optimizer_group_steps=group_steps,no_model_copy=True,no_optimizer_reset=True,research_initial_NOT_promoted=True,native=0,Adam=0))
    save(DOC/'T520_BUDGET.json',dict(max_physics=physics,max_commands=82,main_seconds=1800,learning_seconds=600,audit_seconds=600,backup_reserve=1200,all_stop=p['absolute_all_stop'],own_bytes=p['max_bytes'],full_training_shape=shape,all_data_model_partial_reserve=reserve,global_reserved=global_before+p['max_bytes'],trainable_Actor_parameters=sum(z.numel() for z in obj[0].parameters()),max_live_PG=16,max_shadow_PG=16,max_audit_historical_Adam=16,preflight_discarded_max=4,ValueAdam=0,CPU=[14,15],CPUQuota='100%',GPU=False))
    print('T520_RESEARCH12_OLD_CENTER_REGISTERED_NOT_LAUNCHED',flush=True)

def preflight():
    p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'preflight.json').exists();obj=L.make(p);shadow=L.make(p);before=P.state(*obj);refs=L.references(p);arrays=[]
    source=G.E.N.S.arrays(p['keep_rows'][0]['path']);x=source['states'][:8]
    for i in range(4):
        q=L.make(p)[0].eval().requires_grad_(False);q.sample=True;q.rng=np.random.default_rng([20261113520,999,i]);h=None
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
    clone_receipt=dict(path=model['path'],bytes=model['bytes'],sha256=model['sha256'],joint_hash=model['joint_hash'],recipe='preflight fixture seeds20261113520/999/i,actual2maxAdam;exactlive-shadow')
    Path(model['path']).unlink()
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,commands=6,discarded_synthetic_Actor_Adam=aa['Actor_Adam']+bb['Actor_Adam'],synthetic_fixture_NOT_navigation=True,full_actor_encoder_core_really_updates=True,independent_Value_weights_and_Adam_unchanged=True,live_shadow_ActorAdam_rollback_serialization_exact=True,all_worker_load_cancel_PASS=True,discarded_nonunique_serialization_scratch=clone_receipt))
    print('T520_FULL_ACTOR_ONPOLICY_PREFLIGHT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');R.verify(p,True);pre=read(OUT/'preflight.json');assert pre['state']=='PASS' and not Path('/proc',str(pre['pid'])).exists();assert not (OUT/'launch.json').exists()
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert source['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time());assert seconds>3800
    args=['systemd-run','--user','--unit=stage-t520-old-center-repair','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T520','--why=Bounded research12 single old-center navigation repair','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')]
    subprocess.run(args,check=True);print('T520_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
