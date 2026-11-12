"""Register current-policy multi-start credit pilot, no unlimited relaunches."""
from pathlib import Path
import copy,json,math,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t503_runtime as R
G,L,P,torch,np,read,save,sha,digest=R.G,R.L,R.P,R.torch,R.np,R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/review_20261112'

def keep_record(row):return {k:row[k] for k in ['job_id','path','sha256','success','collision']}
def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t502_fullfield_probe';p=read(prior/'protocol.json');assert read(prior/'probe/audit.json')['state']=='PASS';G.stopped(prior/'probe')
    assert not Path('/proc',str(read(prior/'launch.json')['pid'])).exists();assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T502.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    field=G.rows(prior/'probe');success=[r for r in field if r['success'] and not r['collision']];assert len(success)==5
    detail=read(prior/'probe/diagnosis.json');failed=sorted([d for d in detail if not d['native_success']],key=lambda d:(d['min_decision_distance_m'],d['id']));selected=[d['id'] for d in failed[:2]]
    assert selected==['t502_field_sector5_offset1','t502_field_sector2_offset1']
    by={j['id']:j for j in p['field_jobs']};training=[by[k] for k in selected for _ in range(4)]
    joint=ROOT/'stage/runs/t501_joint_command_repair';oldplan=read(joint/'protocol.json');own=[r for r in G.rows(joint/'joint') if r['success'] and not r['collision']];assert len(own)==379
    keep=own+success;old8=oldplan['old8_jobs'];assert len(old8)==8 and all(j['id'] in {r['job_id'] for r in own} for j in old8)
    now=time.time();physics=2*sum(j['case']['step_cap'] for j in training)+sum(j['case']['step_cap'] for j in old8)+2*sum(j['case']['step_cap'] for j in p['field_jobs'])
    shape=sum(1352*math.ceil(j['case']['step_cap']/100)+4480+65536 for j in training+old8+p['field_jobs'])
    p.update(id='T503_NATIVE_MULTI_START_CLIPPED_PEER_CREDIT',out=str(OUT),parent_model=p['initial'],initial_joint_locked=False,
             training_jobs=training,training_seeds=[[20261112503,i//4,i%4] for i in range(8)],keep_rows=[keep_record(r) for r in keep],old8_jobs=old8,
             initial_field_success_ids=sorted(r['job_id'] for r in success),actor_lr=3e-5,keep_weight=1.,epochs=16,target_kl=.005,
             max_bytes=48*2**20,model_bytes=2*2**20,preflight_commands=6,pilot_max_physics=physics,root_max_physics=physics,root_max_commands=82,root_max_main_seconds=3600,
             absolute_learning_stop=now+1800,absolute_native_stop=now+4500,absolute_all_stop=now+6000,
             maximum_live_Actor_Adam=16,maximum_shadow_Actor_Adam=16,maximum_audit_discarded_Actor_Adam=16,preflight_discarded_Actor_Adam=4,Value_Adam=0,
             scientific_scope='One eight-episode fresh own-policy pilot on two fixed near-goal-miss fullfield tasks,4independentnoiseeach,originaliidrawsigma.2. Three peer physicaltime rewardtogo baseline withterminalcontinuation0,physicalgamma^t weight,per-taskRMS,clippedratio.8/1.2max16updates,targetKL.005. NoCritic/teacher/newreward/threshold/actuator change, not claimedunbiased/novel/variance reduction.',
             optimizer_contract='Full current Actor/mature8parameterAdam drive896/steer256 moments preserved. Explicit newRL learningrate3e-5 replaces supervised.001; initialRLcheckpoint differsonlyoptimizerlr,policyweightsidentical. Freeze backbone/head/Value/ValueAdam,11906adapterparams trainable.',
             learning_gate='Atleastone safeGOAL amongall8samples before updating,otherwise0newActor/ValueAdam andno24retest. Gate isdata-dependent engineeringchoice,notunbiased policygradient claim. No addednoiseseed/round/budget whenfailed.',
             acceptance='Afteroneproposal,old8safe plus all24actualfield24+24exact. Retainresearchonlyif initial5fieldsuccessIDs preserved,fieldqualified>5,contacts<=11 and0invalid/censor. OtherwisefullActor/matureAdam rollback. Fullold275/six54/eight54 andfresh16 NOTexecuted or claimedpreserved onnewcandidate; bestT501 remainsfrozen.',
             authorization='Nov12 user asks analysis then next experiment afterfullfield24transferresult; bounded newrealclosedloopRL mechanism,notrepeatingfrozenT502.')
    p['inputs']=dict(p['inputs']);p['protected_files']=dict(p['protected_files'])
    for f in [*HERE.glob('*.py'),prior/'probe/summary.json',prior/'probe/audit.json',prior/'probe/diagnosis.json',prior/'probe/cases.jsonl',prior/'SEALED.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T502.json',joint/'joint/summary.json',joint/'joint/audit.json']:
        p['inputs'][str(f)]=sha(f)
    for row in keep:p['inputs'][row['path']]=row['sha256']
    for key in ['registration_zero_physical_commands','preflight_zero_physical_commands']:p.pop(key,None)
    p.update(Actor_Adam=16,shadow_Actor_Adam=16,audit_discarded_Actor_Adam=16,preflight_discarded_Actor_Adam=4,Value_Adam=0)
    assert shape+14*2**20<p['max_bytes'] and G.M.used()+p['max_bytes']+2**20<2**30
    OUT.mkdir();obj=L.make(p);model=L.A.export(obj,OUT/'rl_initial.pt.gz',p['model_bytes']);assert model['policy_hash']==p['parent_model']['policy_hash'];p['initial']=model;p['initial_joint_locked']=True
    p['protected_files'][model['path']]=model['sha256'];p['inputs'][model['path']]=model['sha256'];assert digest(P.state(*L.make(p)))==model['joint_hash']
    blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    save(DOC/'T503_BUDGET.json',dict(max_physics=physics,max_commands=82,max_main_seconds=3600,learning_seconds=1200,audit_seconds=600,all_seconds=6000,backup_sync_reserve=1500,own_bytes=p['max_bytes'],shape_bound=shape,metadata_models_partial=14*2**20,global_reserved=G.M.used()-G.M.used(OUT)+p['max_bytes'],learning_stop=p['absolute_learning_stop'],native_stop=p['absolute_native_stop'],all_stop=p['absolute_all_stop'],CPU=[14,15],CPUQuota='100%',GPU=False,live_Actor_Adam=16,shadow_Actor_Adam=16,audit_discarded_Actor_Adam=16,preflight_discarded_Actor_Adam=4,Value_Adam=0))
    print('T503_REGISTERED_NOT_LAUNCHED',flush=True)

def fixture(p):
    row=p['keep_rows'][-1];a=G.E.N.S.arrays(row['path']);x=np.asarray(a['states'][:8]).copy();result=[]
    for i in range(8):
        q=L.make(p)[0];q.sample=True;q.rng=np.random.default_rng([20261112503,999,i]);h=None;acts=[]
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
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,commands=6,discarded_synthetic_Actor_Adam=4,Value_Adam=0,synthetic_fixture_NOT_navigation_training=True,actual_raw_density_errors=[b['density_error'] for b in items],initial_keep_max_abs_error=initial_keep,mature896_256_and_explicit_lr_exact=True,live_shadow_and_fullAdam_rollback_exact=True,three_peer_physical_time_coefficient_finite=True,all_workers_stopped=True))
    print('T503_PREFLIGHT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');R.verify(p,True);pre=read(OUT/'preflight.json');assert pre['state']=='PASS' and not Path('/proc',str(pre['pid'])).exists();assert not (OUT/'launch.json').exists()
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert source['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time());assert seconds>5500
    args=['systemd-run','--user','--unit=stage-t503-fullfield-onpolicy','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T503','--why=Bounded fresh closed-loop fullfield policy learning','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')]
    subprocess.run(args,check=True);print('T503_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
