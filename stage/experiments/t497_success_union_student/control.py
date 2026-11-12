"""Bounded success-source completion and the previously validated student fit."""
from pathlib import Path
import json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t497_runtime as R
G,S,read,save,sha,digest=R.G,R.S,R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/review_20261112'

def prepare():
    assert not OUT.exists()
    prior=ROOT/'stage/runs/t496_neural_speed_profile';p=read(prior/'protocol.json')
    assert read(prior/'teacher/audit.json')['state']=='PASS';G.stopped(prior/'teacher')
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T496.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert not Path('/proc',str(read(prior/'launch.json')['pid'])).exists()
    right=[r for r in G.rows(prior/'teacher') if r['success'] and not r['collision']]
    left=p['left_teacher_rows'];known=left+right;assert len(left)==27 and len(right)==22
    old=G.rows(ROOT/'stage/runs/t493_wide8_night/raw_ppo_02');old={r['job_id']:r for r in old if r['role']=='FULL54_EIGHT'}
    missing={j['id'] for j in p['wide8_jobs']}-{r['job_id'] for r in known}
    assert missing=={'wide117_8m_dev6_RIGHT_'+x for x in ['09','18','21','22','24']}
    references={k:old[k] for k in sorted(missing)};assert all(r['success'] and not r['collision'] for r in references.values())
    for k in ['left_teacher_rows','teacher_jobs','pulse_decisions','teacher_first','drive_logit_increment','Actor_Adam','Value_Adam']:p.pop(k,None)
    now=time.time()
    p.update(id='T497_SUCCESS_SOURCES_TO_ONE_NEURAL_DRIVE_STUDENT',out=str(OUT),
             verified_success_rows=known,reconstruction_references=references,
             reconstruction_jobs=[j for j in p['wide8_jobs'] if j['id'] in missing],
             max_bytes=72*2**20,root_max_physics=6293412,root_max_commands=136,root_max_main_seconds=4500,
             absolute_learning_stop=now+2400,absolute_native_stop=now+6000,absolute_all_stop=now+7200,
             adapter_seed=20261112496,bc_steps=512,preflight_commands=6,
             scientific_scope='One new finite data-completion/student experiment, not continuation of the failed single-teacher gate. Reconstruct exactly five previously successful plain40 RIGHT histories, verify all hashes against T493 and exact replay. Offline training data:27 T495 LEFT+22 T496 RIGHT+5 historical RIGHT. Source selection is never present at deployment.',
             student_scope='Reuse byte-identical T496 Actor and fit functions:5953 trainable drive-adapter parameters only,512live+512shadow and512 independent discarded audit Adam,0new policy-gradient/Value Adam. Both sides27 and all54 unique actual successful histories. New source coverage, not algorithm novelty.',
             acceptance='Same unchanged T496 student rule: full54 all valid, preserve every initial40 success ID, success>40 and zero contact; otherwise rollback fullActorAdam. All54safe then54 independent native model-copy exact; otherwise14 original failure exact. Old275/six/fresh16/under8 not executed.',
             authorization='Nov12 user continued after seeing T496 stop and proposal to complete success sources then learn oneActor; independent bounded pilot, no grid or ended-directory restart.')
    p['inputs']=dict(p['inputs']);p['protected_files']=dict(p['protected_files'])
    for f in [*HERE.glob('*.py'),prior/'teacher/audit.json',prior/'teacher/cases.jsonl',prior/'preflight.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T496.json']:
        p['inputs'][str(f)]=sha(f)
    for row in known:p['inputs'][row['path']]=row['sha256']
    for f in [ROOT/'stage/experiments/t496_neural_speed_profile'/name for name in ['speed_actor.py','speed_runtime.py']]:
        p['inputs'][str(f)]=sha(f)
    assert G.M.used()+p['max_bytes']+2**20<2**30
    assert 59*p['full_bytes']+4*p['model_bytes']+10*2**20<p['max_bytes']
    OUT.mkdir();blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    save(DOC/'T497_BUDGET.json',dict(max_physics=6293412,max_commands=136,max_main_seconds=4500,source_main=900,source_audit=120,student_main=3600,student_audit=600,backup_and_sync_reserve=900,own_bytes=p['max_bytes'],global_reserved=G.M.used()-G.M.used(OUT)+p['max_bytes'],learning_stop=p['absolute_learning_stop'],native_stop=p['absolute_native_stop'],all_stop=p['absolute_all_stop'],CPU=[14,15],CPUQuota='100%',GPU=False,live_Adam=512,shadow_Adam=512,audit_discarded_Adam=512,preflight_new_Adam=0))
    print('T497_REGISTERED_NOT_LAUNCHED',flush=True)

def preflight():
    p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'preflight.json').exists()
    # No repeat synthetic optimizer test: exact same Actor/fit/runtime functions
    # already passed T496 four-clone numerical/Adam rollback contracts.
    previous=ROOT/'stage/runs/t496_neural_speed_profile';old=read(previous/'protocol.json');proof=read(previous/'preflight.json')
    assert proof['state']=='PASS' and proof['discarded_Actor_Adam']==4
    for name in ['speed_actor.py','speed_runtime.py']:
        f=ROOT/'stage/experiments/t496_neural_speed_profile'/name;assert sha(f)==old['inputs'][str(f)]
    obj=S.make(p);baseline=S.PLAIN_FROZEN(p,p['initial']['path']);row=p['verified_success_rows'][0];a=G.E.N.S.arrays(row['path']);x=S.torch.tensor(a['states'])
    with S.torch.no_grad():assert S.torch.equal(obj[0].sequence(x),baseline.sequence(x))
    model=S.export(obj,OUT/'preflight_model.pt.gz',p['model_bytes']);frozen=S.frozen(p,model['path']);h=None;values=[]
    with S.torch.no_grad():
        for xx in x[:24]:_,loc,h,_=frozen.step(xx[None],h);values.append(loc[0])
        error=float((S.torch.stack(values)-frozen.sequence(x[:24])).abs().max());assert error<1e-10
    # Spawn actual unchanged CPU/software-GL workers, abort before native0.
    folder=OUT/'preflight_native';folder.mkdir();counts=G.R.M.mp.get_context('spawn').RawArray('q',3)
    pp=dict(p,phase='student',out=str(folder),native_stop_epoch=time.time()+120,max_physics=1)
    pool=R.R.Pool(pp,counts,lambda:None)
    try:
        values=pool.request([dict(model=model['path'],job=j,seed=None,cancel_probe=True) for j in p['reconstruction_jobs'][:2]])
        assert all(v['state']=='CANCEL_PASS' for v in values) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,commands=6,new_Adam=0,reused_previous_discarded_Adam_proof=str(previous/'preflight.json'),reused_proof_sha256=sha(previous/'preflight.json'),zero_readout_preserves_plain40=True,sequence_step_error=error,all_workers_stopped=True,source_reconstruction_NOT_yet_executed=True))
    print('T497_PREFLIGHT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');R.verify(p,True);pre=read(OUT/'preflight.json');assert pre['state']=='PASS' and not Path('/proc',str(pre['pid'])).exists();assert not (OUT/'launch.json').exists()
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert source['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time());assert seconds>6300
    args=['systemd-run','--user','--unit=stage-t497-success-union','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T497','--why=Verified success-source completion and single neural student','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')]
    subprocess.run(args,check=True);print('T497_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
