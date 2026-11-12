"""Register and launch finite frozen-model regression with complete raw proof."""
from pathlib import Path
import json,math,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t499_runtime as R
G,S,P,read,save,sha,digest=R.G,R.S,R.P,R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/review_20261112'

def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t498_early_error_repair';p=read(prior/'protocol.json');s=read(prior/'repair/summary.json')
    assert s['state']=='COMPLETE' and s['result']['accepted'] and s['result']['independent54_exact']
    assert read(prior/'repair/audit.json')['state']=='PASS';G.stopped(prior/'repair')
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T498.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert not Path('/proc',str(read(prior/'launch.json')['pid'])).exists()
    jobs=p['old275_jobs']+p['six_jobs'];assert len(jobs)==len({j['id'] for j in jobs})==329
    assert sum(j['case']['step_cap'] for j in jobs)==6966735
    # packed removes only exact redundant next_states and hidden_after chains;
    # all original arrays are reconstructed byte-exact. ZIP overhead is reserved.
    storage=sum(1352*math.ceil(j['case']['step_cap']/100)+4480+65536 for j in jobs)
    now=time.time()
    for key in ['keep_rows','repair_row','repair_steps','replay_failure_ids','verified_success_rows','teacher_rows']:p.pop(key,None)
    p.update(id='T499_FROZEN_SINGLE_ACTOR_FULL329',out=str(OUT),initial=s['result']['model'],eight_summary=str(prior/'repair/summary.json'),
             max_bytes=128*2**20,root_max_physics=6966735,root_max_commands=339,root_max_main_seconds=6000,
             absolute_learning_stop=now,absolute_native_stop=now+6600,absolute_all_stop=now+8400,
             preflight_commands=6,Actor_Adam=0,Value_Adam=0,preflight_discarded_Adam=0,
             storage_shape_bound=storage,main_seconds=6000,learning_seconds=1,audit_seconds=600,
             scientific_scope='Frozen T498 singleActor old275+wide6(0.70m80s)54 original Stage task definitions, no teacher/planner/map/router,0optimizer. Full329 histories saved. Historical eight54 allpass retained separately; no model selection, threshold or time change.',
             acceptance='All329 valid safe GOAL required before any independent future heldout protocol. Full329 completion despite scientific failures; missing/censored separately, no automatic training/repair. fresh16/reload78 NOT executed or reserved here.',
             authorization='Nov12 user approved next frozen joint validation after reviewing T498 full54 and plan. Independent finite validation budget, not extension of T498.')
    for key in ['live_Adam','shadow_Adam','audit_discarded_Adam','new_preflight_discarded_Adam']:p[key]=0
    p['inputs']=dict(p['inputs']);p['protected_files']=dict(p['protected_files']);p['protected_files'][p['initial']['path']]=p['initial']['sha256']
    for f in [*HERE.glob('*.py'),prior/'repair/summary.json',prior/'repair/audit.json',prior/'SEALED.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T498.json']:
        p['inputs'][str(f)]=sha(f)
    assert storage+10*2**20<p['max_bytes'] and G.M.used()+p['max_bytes']+2**20<2**30
    OUT.mkdir();blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    save(DOC/'T499_BUDGET.json',dict(max_physics=6966735,max_commands=339,max_main_seconds=6000,audit_seconds=600,all_seconds=8400,backup_sync_reserve=1500,own_bytes=p['max_bytes'],full329_uncompressed_shape_zip_bound=storage,metadata_reserve=10*2**20,global_reserved=G.M.used()-G.M.used(OUT)+p['max_bytes'],native_stop=p['absolute_native_stop'],all_stop=p['absolute_all_stop'],CPU=[14,15],CPUQuota='100%',GPU=False,Adam=0,fresh16_reload78_NOT_reserved=True))
    print('T499_REGISTERED_NOT_LAUNCHED',flush=True)

def preflight():
    p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'preflight.json').exists()
    q=S.frozen(p,p['initial']['path']);assert q.phash()==p['initial']['policy_hash'] and digest(P.checkpoint(p['initial']['path'])['joint_state'])==p['initial']['joint_hash']
    row=G.rows(ROOT/'stage/runs/t498_early_error_repair/repair')[0];a=G.E.N.S.arrays(row['path']);packed=G.E.N.S.packed(a)
    assert sum(v.nbytes for v in packed.values())==1352*len(a['states'])+4480
    x=R.torch.tensor(a['states'][:24]);h=None;vals=[]
    with R.torch.no_grad():
        for xx in x:_,loc,h,_=q.step(xx[None],h);vals.append(loc[0])
        error=float((R.torch.stack(vals)-q.sequence(x)).abs().max());assert error<1e-10
    folder=OUT/'preflight_native';folder.mkdir();counts=G.R.M.mp.get_context('spawn').RawArray('q',3)
    pool=R.R.Pool(dict(p,phase='student',out=str(folder),native_stop_epoch=time.time()+120,max_physics=1),counts,lambda:None)
    try:
        jobs=[p['old275_jobs'][0],p['six_jobs'][0]]
        result=pool.request([dict(model=p['initial']['path'],job=j,seed=None,cancel_probe=True) for j in jobs])
        assert all(v['state']=='CANCEL_PASS' for v in result) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,commands=6,Adam=0,actual_model_and_joint_SHA_exact=True,sequence_step_error=error,full_packed_shape_bytes_contract=True,old_and_six_native_cancel_before_physical=True,all_workers_stopped=True))
    print('T499_PREFLIGHT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');R.verify(p,True);pre=read(OUT/'preflight.json');assert pre['state']=='PASS' and not Path('/proc',str(pre['pid'])).exists();assert not (OUT/'launch.json').exists()
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert source['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time());assert seconds>8100
    args=['systemd-run','--user','--unit=stage-t499-frozen-joint','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T499','--why=Frozen singleActor full329 validation','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')]
    subprocess.run(args,check=True);print('T499_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
