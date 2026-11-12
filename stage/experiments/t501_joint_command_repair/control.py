"""Register one finite two-command residual learning and full383 evaluation."""
from pathlib import Path
import json,math,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t501_runtime as R
G,L,P,torch,read,save,sha,digest=R.G,R.L,R.P,R.torch,R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/review_20261112'

def training_row(r):return {k:r[k] for k in ['job_id','path','sha256','array_hashes','success','collision']}
def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t500_joint_repair_sources';p=read(ROOT/'stage/runs/t499_frozen_joint_validation/protocol.json')
    for child in ['ablation','sources']:assert read(prior/child/'audit.json')['state']=='PASS';G.stopped(prior/child)
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T500.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert not Path('/proc',str(read(prior/'launch.json')['pid'])).exists()
    repair=read(prior/'sources/summary.json')['result']['successful_repair_sources'];old=[r for r in G.rows(ROOT/'stage/runs/t499_frozen_joint_validation/regression') if r['success'] and not r['collision']];eight=G.rows(ROOT/'stage/runs/t498_early_error_repair/repair');keep=old+eight
    assert len(repair)==43 and len(keep)==340 and len({r['job_id'] for r in repair+keep})==383
    jobs=p['old275_jobs']+p['six_jobs']+p['wide8_jobs'];assert len(jobs)==383
    storage=sum(1352*math.ceil(j['case']['step_cap']/100)+4480+65536 for j in jobs);assert sum(j['case']['step_cap'] for j in jobs)==9846771
    now=time.time()
    p.update(id='T501_JOINT_NEURAL_COMMAND_REPAIR_WITH340_OWN_SUCCESS_KEEP',out=str(OUT),
             repair_rows=[training_row(r) for r in repair],keep_rows=[training_row(r) for r in keep],initial_success_ids=sorted(r['job_id'] for r in keep),
             steer_seed=20261112501,fit_steps=256,max_bytes=180*2**20,model_bytes=2*2**20,
             root_max_physics=9846771,root_max_commands=393,root_max_main_seconds=7200,
             absolute_learning_stop=now+1800,absolute_native_stop=now+8100,absolute_all_stop=now+10800,
             preflight_commands=6,Actor_Adam=256,shadow_Actor_Adam=256,audit_discarded_Actor_Adam=256,preflight_discarded_Actor_Adam=4,Value_Adam=0,
             scientific_scope='One finite actual-success supervised repair of existing RL Actor. New zero-output steer184x32x1 neural residual + retained drive residual, each bounded.3logit;11906trainable parameters, fixed backbone/core/head/Value. Same single120-input64carryActor deployment; no side/time/task-ID threshold, planner/map/waypoint/teacher/router.',
             optimizer_contract='Preserve existing drive640-stepAdam state exactly; add four new steer parameter slots with absent moments(step0). One expandedAdam; no silent reset/mature steering history fabrication. Independent256-step historical clone reconstruction.',
             loss_contract='Fixed256 updates;0.5repair+0.5keep. Repair43 grouped task4(7)/task6(35)/six(1), threefamilies equal, episodes equal;0.5mean+0.5upper20percent state risk, drive/steer equal, existing clipped-action boundary hinge. Keep340actual student own histories groupsold233/six53/eight54 equal; original unclipped locations initial zero, not new expert or onpolicy.',
             acceptance='Complete383 valid; research acceptance preserves all initial340success IDs and gains>=1. Otherwise fullActor/mapped matureAdam rollback. Stable joint development only if383/383safe. fresh16/reload78 not reserved/consumed; no automatic extra epoch/seed/grid or phase-router.',
             authorization='Nov12 user explicitly starts next round after reviewing dual-command residual repair+340success retention and complete383 plan. Separate bounded new loss/architecture mechanism, not extension of T498/500.')
    for k in ['ablation_jobs','source_jobs','source_references','failed_references','existing_recovery_row','success_source_model','ablation_model','ablation_max_physics','sources_max_physics']:p.pop(k,None)
    p['inputs']=dict(p['inputs']);p['protected_files']=dict(p['protected_files'])
    for f in [*HERE.glob('*.py'),prior/'sources/summary.json',prior/'sources/audit.json',prior/'SEALED.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T500.json',DOC/'T501_DATA_AND_ACTION_SUPPORT.json']:
        p['inputs'][str(f)]=sha(f)
    for r in repair+keep:p['inputs'][r['path']]=r['sha256']
    assert G.M.used()+p['max_bytes']+2**20<2**30 and storage+20*2**20<p['max_bytes']
    obj=L.make(p);before=P.state(*obj);oldstate=P.checkpoint(p['initial']['path'])['joint_state'];p['expanded_initial_joint_hash']=digest(before)
    assert all(torch.equal(before['actor'][k],v) for k,v in oldstate['actor'].items())
    assert not torch.count_nonzero(obj[0].steer_adapter[-1].weight) and not torch.count_nonzero(obj[0].steer_adapter[-1].bias)
    assert len(json.dumps(p).encode())<2**21
    OUT.mkdir();blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    save(DOC/'T501_BUDGET.json',dict(max_physics=9846771,max_commands=393,max_main_seconds=7200,learning_seconds=1200,audit_seconds=1200,all_seconds=10800,backup_sync_reserve=1500,own_bytes=p['max_bytes'],full383_shape_bound=storage,model_metadata_partial_reserve=20*2**20,global_reserved=G.M.used()-G.M.used(OUT)+p['max_bytes'],learning_stop=p['absolute_learning_stop'],native_stop=p['absolute_native_stop'],all_stop=p['absolute_all_stop'],CPU=[14,15],CPUQuota='100%',GPU=False,live_Actor_Adam=256,shadow_Actor_Adam=256,audit_discarded_Actor_Adam=256,preflight_discarded_Actor_Adam=4,Value_Adam=0,fresh16_reload78_NOT_reserved=True))
    print('T501_REGISTERED_NOT_LAUNCHED',flush=True)

def preflight():
    p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'preflight.json').exists();data=L.groups(p);obj=L.make(p);shadow=L.make(p);before=P.state(*obj)
    assert digest(before)==p['expanded_initial_joint_hash'];metrics=L.measure(obj[0],data);assert metrics['keep']['max_episode_MSE']<1e-20
    q=L.LEGACY_FROZEN(p,p['initial']['path']);row=p['keep_rows'][-1];a=G.E.N.S.arrays(row['path']);x=torch.tensor(a['states']);h=hh=None
    with torch.no_grad():
        assert torch.equal(q.sequence(x),obj[0].sequence(x))
        for xx in x[:24]:
            aa,la,h,_=q.step(xx[None],h);bb,lb,hh,_=obj[0].step(xx[None],hh);assert torch.equal(aa,bb) and torch.equal(la,lb) and torch.equal(h,hh)
    start=time.monotonic();aa=L.fit(obj,data,p,lambda:None,lambda *_:None,steps=2);bb=L.fit(shadow,data,p,lambda:None,lambda *_:None,steps=2);elapsed=time.monotonic()-start
    assert aa==bb and digest(P.state(*obj))==digest(P.state(*shadow));estimate=elapsed*128;assert estimate<1000,(estimate,elapsed)
    model=L.export(obj,OUT/'preflight_model.pt.gz',p['model_bytes']);assert digest(P.checkpoint(model['path'])['joint_state'])==digest(P.state(*obj))
    changed=P.state(*obj)
    for o in [obj,shadow]:
        P.load(o,before);assert digest(P.state(*o))==p['expanded_initial_joint_hash']
        try:P.load(o,changed);raise RuntimeError('T501_SIMULATED_INTERRUPTION')
        except RuntimeError as e:assert str(e)=='T501_SIMULATED_INTERRUPTION';P.load(o,before)
        assert digest(P.state(*o))==p['expanded_initial_joint_hash']
    frozen=L.frozen(p,model['path']);h=None;locs=[]
    with torch.no_grad():
        for xx in x[:24]:_,loc,h,_=frozen.step(xx[None],h);locs.append(loc[0])
        err=float((torch.stack(locs)-frozen.sequence(x[:24])).abs().max());assert err<1e-10
    folder=OUT/'preflight_native';folder.mkdir();counts=G.R.M.mp.get_context('spawn').RawArray('q',3);pool=R.R.Pool(dict(p,phase='student',out=str(folder),native_stop_epoch=time.time()+120,max_physics=1),counts,lambda:None)
    try:
        vals=pool.request([dict(model=model['path'],job=j,seed=None,cancel_probe=True) for j in [p['old275_jobs'][0],p['six_jobs'][0]]]);assert all(v['state']=='CANCEL_PASS' for v in vals) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,commands=6,discarded_Actor_Adam=4,Value_Adam=0,initial_actual_policy_actions_and_hidden_exact=True,drive_mature640_newsteer0_mapped=True,initial_metrics=metrics,clone_updates_exact=True,full_mapped_ActorAdam_rejection_interruption_restore=True,sequence_step_error=err,clone4_step_seconds=elapsed,estimated_live_shadow512_seconds=estimate,all_workers_stopped=True,preflight_fit=aa))
    print('T501_PREFLIGHT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');R.verify(p,True);pre=read(OUT/'preflight.json');assert pre['state']=='PASS' and not Path('/proc',str(pre['pid'])).exists();assert not (OUT/'launch.json').exists()
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert source['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time());assert seconds>10100
    args=['systemd-run','--user','--unit=stage-t501-joint-command-repair','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T501','--why=SingleActor joint repair and full383 validation','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')]
    subprocess.run(args,check=True);print('T501_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
