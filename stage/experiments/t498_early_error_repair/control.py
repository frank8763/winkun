"""Register, numerically verify, then launch exactly one bounded repair."""
from pathlib import Path
import json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t498_runtime as R
G,S,L,P,read,save,sha,digest=R.G,R.S,R.L,R.P,R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/review_20261112'

def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t497_success_union_student';p=read(prior/'protocol.json');s=read(prior/'student/summary.json')
    assert s['state']=='COMPLETE' and read(prior/'student/audit.json')['state']=='PASS';G.stopped(prior/'student')
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T497.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert not Path('/proc',str(read(prior/'launch.json')['pid'])).exists()
    rows=G.rows(prior/'student');keep=[r for r in rows if r['success'] and not r['collision']];failed=[r for r in rows if not r['success']]
    assert len(keep)==53 and len(failed)==1 and failed[0]['job_id']=='wide117_8m_dev6_RIGHT_18'
    repair=next(r for r in G.rows(prior/'teacher') if r['job_id']==failed[0]['job_id']);assert repair['success'] and not repair['collision']
    now=time.time()
    for key in ['verified_success_rows','reconstruction_references','reconstruction_jobs','bc_steps']:p.pop(key,None)
    p.update(id='T498_EARLY_WORST_ERROR_AND_OWN53_RETENTION',out=str(OUT),initial=s['result']['model'],initial_success_ids=sorted(r['job_id'] for r in keep),
             keep_rows=keep,repair_row=repair,repair_steps=128,replay_failure_ids=sorted(set(p['original_failure_ids'])|{repair['job_id']}),
             max_bytes=64*2**20,root_max_physics=5760072,root_max_commands=122,root_max_main_seconds=3600,
             absolute_learning_stop=now+1800,absolute_native_stop=now+4800,absolute_all_stop=now+6000,
             preflight_commands=6,new_preflight_discarded_Adam=4,live_Adam=128,shadow_Adam=128,audit_discarded_Adam=128,
             scientific_scope='One new predeclared repair mechanism from explicitly UNPROMOTED53/54 research model and its mature512-step Adam. Preserve53own successful unclipped-location histories, repair verified RIGHT18 successful control history. No runtime task/side/time branch.',
             student_scope='Same5953parameter neural longitudinal adapter, fixed backbone/GRU/steering/Value. Per update0.5repair+0.5mean53keep; repair0.5all+0.25first20s+0.25top20percent first20 error.128live128shadow plus128 independent discarded audit. No policy-gradient/Value updates, no weight/seed/epoch grid or novelty claim.',
             acceptance='Exactly54/54 safe valid including RIGHT18 and all previous53. Otherwise rollback fullActor plus matureAdam and preserve rejected proposal. Allpass newprocess modelcopy full54 exact, else fixedoriginal14plusRIGHT18 exact15. old275/six/freshnull; not fullmap stability.',
             authorization='Nov12 user explicitly requests launch next round after inspecting T498 repair+53keep draft; new bounded loss mechanism not T497 budget extension.')
    p['inputs']=dict(p['inputs']);p['protected_files']=dict(p['protected_files']);p['protected_files'][p['initial']['path']]=p['initial']['sha256']
    for f in [*HERE.glob('*.py'),prior/'student/summary.json',prior/'student/audit.json',prior/'student/cases.jsonl',prior/'SEALED.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T497.json']:
        p['inputs'][str(f)]=sha(f)
    for row in keep+[repair]:p['inputs'][row['path']]=row['sha256']
    assert G.M.used()+p['max_bytes']+2**20<2**30
    assert 54*p['full_bytes']+4*p['model_bytes']+10*2**20<p['max_bytes']
    OUT.mkdir();blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    save(DOC/'T498_BUDGET.json',dict(max_physics=5760072,max_commands=122,max_main_seconds=3600,audit_seconds=600,all_seconds=6000,backup_sync_reserve=900,own_bytes=p['max_bytes'],global_reserved=G.M.used()-G.M.used(OUT)+p['max_bytes'],learning_stop=p['absolute_learning_stop'],native_stop=p['absolute_native_stop'],all_stop=p['absolute_all_stop'],CPU=[14,15],CPUQuota='100%',GPU=False,live_Adam=128,shadow_Adam=128,audit_discarded_Adam=128,preflight_discarded_Adam=4))
    print('T498_REGISTERED_NOT_LAUNCHED',flush=True)

def preflight():
    p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'preflight.json').exists()
    data=L.groups(p);obj=L.make(p);shadow=L.make(p);before=P.state(*obj);assert digest(before)==p['initial']['joint_hash']
    initial=L.measure(obj[0],data);assert initial['keep_max_episode_MSE']<1e-20
    events=[];fn=lambda *x:events.append(list(x));aa=L.fit(obj,data,p,lambda:None,fn,steps=2);bb=L.fit(shadow,data,p,lambda:None,lambda *_:None,steps=2)
    assert aa==bb and digest(P.state(*obj))==digest(P.state(*shadow))
    model=S.export(obj,OUT/'preflight_model.pt.gz',p['model_bytes']);assert digest(P.checkpoint(model['path'])['joint_state'])==model['joint_hash']
    for o in [obj,shadow]:P.load(o,before);assert digest(P.state(*o))==p['initial']['joint_hash']
    frozen=S.frozen(p,model['path']);a=G.E.N.S.arrays(p['repair_row']['path']);x=L.torch.tensor(a['states'][:24]);h=None;values=[]
    with L.torch.no_grad():
        for xx in x:_,loc,h,_=frozen.step(xx[None],h);values.append(loc[0])
        error=float((L.torch.stack(values)-frozen.sequence(x)).abs().max());assert error<1e-10
    # Both refusal and simulated interruption restore the entire mature state.
    for o in [obj,shadow]:
        P.load(o,P.checkpoint(model['path'])['joint_state']);P.load(o,before);assert digest(P.state(*o))==p['initial']['joint_hash']
    folder=OUT/'preflight_native';folder.mkdir();counts=G.R.M.mp.get_context('spawn').RawArray('q',3)
    pool=R.R.Pool(dict(p,phase='student',out=str(folder),native_stop_epoch=time.time()+120,max_physics=1),counts,lambda:None)
    try:
        vals=pool.request([dict(model=model['path'],job=j,seed=None,cancel_probe=True) for j in p['wide8_jobs'][:2]])
        assert all(v['state']=='CANCEL_PASS' for v in vals) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,commands=6,discarded_Actor_Adam=4,Value_Adam=0,initial_mature_joint_exact=True,initial_53_keep=initial,clone_updates_exact=True,full_rejection_and_interruption_rollback_exact=True,sequence_step_error=error,all_workers_stopped=True,preflight_fit=aa))
    print('T498_PREFLIGHT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');R.verify(p,True);pre=read(OUT/'preflight.json');assert pre['state']=='PASS' and not Path('/proc',str(pre['pid'])).exists();assert not (OUT/'launch.json').exists()
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert source['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time());assert seconds>5100
    args=['systemd-run','--user','--unit=stage-t498-early-error-repair','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T498','--why=Bounded early-error repair preserving53 own successes','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')]
    subprocess.run(args,check=True);print('T498_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
