"""Register current-policy multi-start credit pilot, no unlimited relaunches."""
from pathlib import Path
import copy,json,math,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t508_runtime as R
G,L,P,torch,np,read,save,sha,digest=R.G,R.L,R.P,R.torch,R.np,R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/review_20261112'

def keep_record(row):return {k:row[k] for k in ['job_id','path','sha256','success','collision']}
def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t507_joint_success_transfer';p=read(prior/'protocol.json')
    review=ROOT/'stage/runs/t507_powerloss_review';audit=read(review/'audit.json');assert audit['state']=='INTERRUPTED_BY_POWER_LOSS_COMPLETED_BLOCK_AUDIT_PASS' and not audit['original_final_gate_complete']
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T507_powerloss.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert all(not Path('/proc',str(pid)).exists() for pid in read(review/'protocol.json')['old_pids'])
    rows=G.rows(prior/'pilot');assert len(rows)==18 and all(r['success'] and not r['collision'] for r in rows)
    model=audit['final_model'];assert all(r['loaded_model_sha256']==model['sha256'] for r in rows) and sha(model['path'])==model['sha256']
    physics=sum(j['case']['step_cap'] for j in p['remaining_field_jobs']+p['field_jobs']);shape=sum(1352*math.ceil(j['case']['step_cap']/100)+4480+65536 for j in p['remaining_field_jobs']);now=time.time()
    end=min(p['absolute_all_stop'],now+4800)
    p.update(id='T508_FROZEN_POWERLOSS_VALIDATION_COMPLETION',out=str(OUT),initial=model,parent_model=model,retained18=rows,
             max_bytes=32*2**20,preflight_commands=6,pilot_max_physics=physics,root_max_physics=physics,root_max_commands=48,root_max_main_seconds=2400,
             absolute_learning_stop=now,absolute_native_stop=end-1200,absolute_all_stop=end,
             Actor_Adam=0,shadow_Actor_Adam=0,audit_discarded_Actor_Adam=0,maximum_live_Actor_Adam=0,maximum_shadow_Actor_Adam=0,maximum_audit_discarded_Actor_Adam=0,preflight_discarded_Actor_Adam=0,Value_Adam=0,
             max_live_Actor_Adam=0,max_shadow_Actor_Adam=0,max_audit_discarded_Actor_Adam=0,live_Adam=0,shadow_Adam=0,audit_discarded_Adam=0,new_policy_gradient_Adam=0,new_preflight_discarded_Adam=0,
             scientific_scope='Frozen sameT507block0model only:14missingcases+complete24exact;importold18samepolicyresults auditedafterpowerloss.0training/Actor/ValueAdam,nochangedscientificmechanism or successrules. Originalpowerlossdir unchangedneverrestart.',
             optimizer_contract='ModelincludingActor/Adam/Value/ValueAdam bytesfrozen no update or optimizerconstruction.',
             loss_contract='No training loss,nooptimization/noextraPPOorBC.',
             acceptance='ExactoriginalT507finalgate:old8allactualsafeGOAL,originalfield7retained,fieldqualified>7,contacts<=10,0invalidcensor.32distinctfinalcases composed imported18+new14 sameexactmodel. Complete24exact. No full275/six54/eight54/fresh16/78.',
             replay_contract='Original18native18exactalreadycompletedandforensicallyaudited. New14missingoriginalcases+24exact,including10oldfield exactagain explicitphysicalrepeat. Powerloss2unknownrequestsremainunknownnotclaimedcompleted;new14areadditionalchargedphysicalwork.',
             conditional='No training; unconditionalremaining14then24exact once. No autoretryorresumestaledirectory.',
             authorization='Nov12userexplicitcontinueafterpowerlossreview;boundedfrozencompletiononlynotnewlearningbudget.')
    for key in ['repair_rows','max_blocks','block_steps','keep_rows','near_goal_radius_m']:p.pop(key,None)
    p['inputs']=dict(p['inputs']);p['protected_files']=dict(p['protected_files']);p['protected_files'][model['path']]=model['sha256']
    for f in [*HERE.glob('*.py'),prior/'pilot/blocks.jsonl',prior/'pilot/cases.jsonl',review/'audit.json',review/'original_manifest.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T507_powerloss.json']:
        p['inputs'][str(f)]=sha(f)
    for row in rows:p['inputs'][row['path']]=row['sha256']
    assert shape+8*2**20<p['max_bytes'] and G.M.used()+p['max_bytes']+2**20<2**30,(shape,G.M.used())
    assert now+2400+300+1200<end
    OUT.mkdir();blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    save(DOC/'T508_BUDGET.json',dict(max_physics=physics,max_commands=48,max_main_seconds=2400,audit_seconds=300,all_seconds=end-now,backup_sync_reserve=1200,own_bytes=p['max_bytes'],shape_bound=shape,metadata_partial=8*2**20,global_reserved=G.M.used()-G.M.used(OUT)+p['max_bytes'],native_stop=p['absolute_native_stop'],all_stop=end,CPU=[14,15],CPUQuota='100%',GPU=False,Actor_Adam=0,Value_Adam=0,preflight_Adam=0,new_validation_cases=14,exact_native_replays=24,original_powerloss_physics_exact=None,original_powerloss_physics_lower=747195,original_powerloss_physics_upper=831302))
    print('T508_REGISTERED_FROZEN_NOT_LAUNCHED',flush=True)

def preflight():
    p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'preflight.json').exists();q=L.frozen(p,p['initial']['path']);assert q.phash()==p['initial']['policy_hash'] and digest(P.checkpoint(p['initial']['path'])['joint_state'])==p['initial']['joint_hash']
    for r in p['retained18']:
        assert r['loaded_model_sha256']==p['initial']['sha256'] and sha(r['path'])==r['sha256'];a=G.E.N.S.arrays(r['path']);assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
    folder=OUT/'preflight_native';folder.mkdir();counts=G.R.M.mp.get_context('spawn').RawArray('q',3);pool=R.F.Pool(dict(p,phase='student',out=str(folder),native_stop_epoch=time.time()+120,max_physics=1),counts,lambda:None)
    try:
        vals=pool.request([dict(model=p['initial']['path'],job=j,seed=None,cancel_probe=True) for j in [p['remaining_field_jobs'][0],p['old8_jobs'][0]]]);assert all(v['state']=='CANCEL_PASS' for v in vals) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,commands=6,Actor_Adam=0,Value_Adam=0,all_imported18_same_model_full_array_SHA_verified=True,frozen_model_joint_SHA_verified=True,actual_worker_cancel_pass=True,all_workers_stopped=True))
    print('T508_FROZEN_PREFLIGHT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');R.verify(p,True);pre=read(OUT/'preflight.json');assert pre['state']=='PASS' and not Path('/proc',str(pre['pid'])).exists();assert not (OUT/'launch.json').exists()
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert source['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time());assert seconds>4400
    args=['systemd-run','--user','--unit=stage-t508-frozen-completion','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T508','--why=Frozen verification of interrupted Stage candidate','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')]
    subprocess.run(args,check=True);print('T508_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
