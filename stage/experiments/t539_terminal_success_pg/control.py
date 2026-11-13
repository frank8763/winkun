"""Register one exact trajectory-success score step, zero extra sampling."""
from pathlib import Path
import copy,os,sys,time
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t539_runtime as R
ROOT,OUT,DOC,L,read,save,sha,digest=R.ROOT,R.OUT,R.DOC,R.L,R.read,R.save,R.sha,R.digest
def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t538_native_direction_persistence';old=read(prior/'protocol.json');end=old['absolute_all_stop']
    assert read(prior/'audit.json')['state']=='PASS' and all(not Path('/proc',str(i)).exists() for i in read(prior/'SEALED.json')['scientific_pids']+[read(prior/'launch.json')['pid']])
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T538.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert time.time()+1600<end and time.time()+1000<end-900,'Full evaluation cannot fit current cutoff'
    configfile=ROOT/'stage/runs/t531_joint_success_distillation/protocol.json';base=read(configfile);initial=base['neural_teacher'];trainfile=ROOT/'stage/runs/t519_fresh_full_actor_rl/pilot/cases.jsonl'
    baseline_cases=ROOT/'stage/runs/t508_frozen_completion/pilot/cases.jsonl';baseline=next(r for r in R.G.rows(baseline_cases.parent) if r['job_id']=='t502_field_sector1_offset1')
    assert baseline['policy_hash']==initial['policy_hash'] and not baseline['success'] and not baseline['administrative_censor']
    p=dict(id='T539_WHOLE_EPISODE_SUCCESS_SINGLE_PG',actor_configuration=str(configfile),training_cases=str(trainfile),deterministic_baseline_cases=str(baseline_cases),initial=initial,deterministic_success_baseline=0.,learning_rate=.001,max_mean_KL=.005,max_point_KL=.05,max_bytes=int(1.375*2**20),model_bytes=640*1024,compact_bytes=12*1024,main_seconds=1000,learning_seconds=120,audit_seconds=240,transport_seconds=180,source_seconds=120,absolute_learning_stop=end-1800,absolute_native_stop=end-900,absolute_all_stop=end,stage_cpus=[14,15],raw_sigma=base['raw_sigma'],max_commands=68,max_physics=2*sum(j['case']['step_cap'] for j in base['old8_jobs']+base['field_jobs']),preflight_commands=5,scope='Exactly1wholeActorSGD success-probability REINFORCE step,fulltrajectory sum score,deterministic baseline0,original4samebehavior-policy T519episodes reused at identicalT507initial hash,3success1timeout. No newtraining samples,no teacherBC/noPPO/noCritic/nohistoricalMSE,noAdam. FreshSGD vs matureAdam plus estimator/objective/one-step changes are a joint pilot,not singlefactor causal attribution or original algorithm. Then onecandidatefullold8/field24+32exact,retainallprior11andold8/newfield>11elsewholeActorSGDrollback. No automatic extra steps/seeds.',authorization='User explicitly 下一步 after T538 complete. New bounded learning hypothesis within20:00window; no rerun ended dirs,no extension of25k diagnostic budget. New maximum3.201204M evaluation physics separately counted; no extra training collection.')
    p['inputs']={f:h for f,h in old['inputs'].items() if Path(f).suffix=='.py'}
    for f in [*HERE.glob('*.py'),configfile,trainfile,baseline_cases,DOC/'T539_PLAN.md',prior/'audit.json']:p['inputs'][str(f)]=sha(f)
    p['protected_files']={initial['path']:initial['sha256']}
    for r in R.G.rows(trainfile.parent)[:4]:assert r['policy_hash']==initial['policy_hash'];p['inputs'][r['path']]=r['sha256']
    for r in base['keep_rows']:
        if not r['job_id'].startswith('t502_'):p['inputs'][r['path']]=r['sha256']
    p['inputs'].update(p['protected_files'])
    assert R.used(ROOT/'stage/runs')+p['max_bytes']+65536<2**30
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);R.verify(p,True)
    c=R.configuration(p);items=L.arrays(c);assert [L.labels(a) for a in items]==[1.,1.,1.,0.]
    save(DOC/'T539_BUDGET.json',dict(new_training_native=0,reused_episodes=4,new_policy_gradient_SGD=1,live_SGD=1,shadow_SGD=1,audit_discarded_SGD=1,preflight_discarded_SGD=2,ActorAdam=0,ValueUpdates=0,max_new_evaluation_physics=p['max_physics'],main_commands=68,preflight_commands=5,own_bytes=p['max_bytes'],global_reserved=R.used(ROOT/'stage/runs')-R.used(OUT)+p['max_bytes'],model_bytes=p['model_bytes'],compact_each=p['compact_bytes'],main_seconds=1000,audit_seconds=240,backup_seconds=180,source_seconds=120,all_stop=end,native_stop=end-900))
    print('T539_REGISTERED_NOT_LAUNCHED',flush=True)
def preflight():
    root=read(OUT/'protocol.json');p=R.configuration(root);R.verify(p,True);assert not (OUT/'preflight.json').exists();obj=L.make(p);shadow=L.make(p);before=L.state(obj)
    # Real-input synthetic stochastic fixture; not a navigation training batch.
    source=R.E.N.S.arrays(p['old_success_rows'][0]['path']);x=source['states'][:8];items=[]
    for i in range(4):
        q=L.make(p)[0].eval().requires_grad_(False);q.sample=True;q.rng=R.np.random.default_rng([539,999,i]);h=None
        with R.torch.no_grad():
            for z in x:_,_,h,_=q.step(R.torch.tensor(z[None]),h)
        tape=q.tape;end=R.np.zeros((8,3),R.np.uint8);end[-1,1 if i%2==0 else 2]=1
        items.append(dict(states=x.copy(),next_states=R.np.concatenate([x[1:],x[-1:]]),ownership=R.np.zeros(8,R.np.uint8),stochastic=R.np.ones(8,R.np.uint8),end_flags=end,raw_actions=R.np.asarray([t['raw_actions'] for t in tape]),atom=R.np.asarray([t['atom'] for t in tape]),locations=R.np.asarray([t['locations'] for t in tape]),logp=R.np.asarray([t['logp'] for t in tape])))
    a=L.fit(obj,items,p,lambda:None,lambda *_:None);b=L.fit(shadow,items,p,lambda:None,lambda *_:None);assert a==b and digest(L.state(obj))==digest(L.state(shadow))
    model=L.export(obj,p,OUT/'discarded_preflight.pt.gz');assert L.A.frozen(p,model['path']).phash()==model['policy_hash'];Path(model['path']).unlink()
    for o in [obj,shadow]:L.load(o,before);assert digest(L.state(o))==digest(before)
    folder=OUT/'preflight_native';folder.mkdir();counts=R.G.R.M.mp.get_context('spawn').RawArray('q',3)
    pool=R.F.Pool(dict(p,out=str(folder),phase='student',native_stop_epoch=time.time()+60,max_physics=1),counts,lambda:None)
    try:
        vals=pool.request([dict(model=p['initial']['path'],job=p['old8_jobs'][0],seed=None,cancel_probe=True)]);assert vals[0]['state']=='CANCEL_PASS' and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),new_native=0,commands=5,discarded_synthetic_SGD=2,live_shadow_serialize_rollback_exact=True,actor_only_plus_SGD_checkpoint_loader_pass=True,worker_zero_native_cancel=True,scientific_training_NOT_executed=True));print('T539_PREFLIGHT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
