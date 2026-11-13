"""Finite learning from a truly successful recovery; no success-before-learning gate."""
from pathlib import Path
import json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t527_runtime as R
G,L,P,torch,np,read,save,sha,digest=R.G,R.L,R.P,R.torch,R.np,R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/day_20261113'
def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t526_early_feedback_qualification';assert read(prior/'pilot/audit.json')['state']=='PASS'
    assert all(not Path('/proc',str(i)).exists() for i in read(prior/'SEALED.json')['scientific_pids']+[read(prior/'launch.json')['pid']])
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T526.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p=read(ROOT/'stage/runs/t525_student_state_recovery/protocol.json');row=G.rows(prior/'pilot')[0];assert row['success'] and not row['collision'] and row['teacher'] and sha(row['path'])==row['sha256']
    rr=read(prior/'pilot/summary.json')['result'];assert rr['recovery_teacher_qualified'] and not rr['handoff']['detail']['field_qualified_success'];now=time.time();assert now+1170<p['absolute_all_stop']
    ids=[r['job_id'] for r in p['keep_rows']];jobs={j['id']:j for j in p['old8_jobs']+p['field_jobs']};panel=[jobs[k] for k in ids]+[p['probe_job']];assert len(panel)==20 and len(set(j['id'] for j in panel))==20
    physics=2*sum(j['case']['step_cap'] for j in panel)
    p.update(id='T527_VALIDATED_FULL_RECOVERY_SUPERVISION',out=str(OUT),recovery_row=row,epochs=16,evaluation_jobs=panel,initial_protected_ids=ids,max_bytes=int(3.5*2**20),preflight_commands=0,pilot_commands=44,pilot_max_physics=physics,root_max_physics=physics,root_max_commands=44,root_max_main_seconds=780,
        absolute_learning_stop=p['absolute_all_stop']-600,absolute_native_stop=p['absolute_all_stop']-300,
        scientific_scope='First learning from T526 actuallysafeGOAL teacher recovery on acceptedstudentstate; failed untrainedhandoff NOT a ban on valid successful labels. Reuseactual19sameActorGOALfunctions,16fullhistory BC+shadow,0PG/Value. Onlyold8+field11+target20panel+exact,fullfield24null evenifpanelpass.',
        acceptance='All20samecandidateautonomoussafeGOAL protects19andaddsone target ->researchcandidateonly pending full24. AnyfailurewholeActorAdamrollbackexpandedT507traininginitial. Originalacceptedbestneveroverwritten.',
        authorization='User continueprogress after applying methods within15:30window. Newlimitedsupervisedlearningmechanism usesnowprovenrecovery;originalT526falseeligibility/error/historyretained. Notrestart/extendearlierRLandnoadditionalgrids.')
    p['inputs']={f:h for f,h in p['inputs'].items() if Path(f).suffix=='.py'}
    for f in [*HERE.glob('*.py'),DOC/'T527_PLAN.md',prior/'pilot/summary.json',prior/'pilot/audit.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T526.json']:
        p['inputs'][str(f)]=sha(f)
    p['inputs'][row['path']]=row['sha256']
    for r in p['keep_rows']:p['inputs'][r['path']]=r['sha256']
    p['inputs'].update(p['protected_files']);obj=L.make(p);assert digest(P.state(*obj))==p['training_initial_joint_hash']
    reserve=p['model_bytes']+20*p['compact_bytes']+768*1024;used=G.M.used();assert reserve<p['max_bytes'] and used+p['max_bytes']+65536<2**30,(used,reserve)
    OUT.mkdir();blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    save(DOC/'T527_BUDGET.json',dict(max_physics=physics,max_commands=44,main_seconds=780,learning_seconds=300,audit_seconds=150,backup_reserve=180,all_stop=p['absolute_all_stop'],own_bytes=p['max_bytes'],data_metadata_reserve=reserve,global_reserved=used+p['max_bytes'],live_supervised_Adam=16,shadow_supervised_Adam=16,audit_discarded_Adam=16,preflight_discarded_Adam=4,new_PG=0,ValueAdam=0,CPU=[14,15],CPUQuota='100%',GPU=False))
    print('T527_REGISTERED_NOT_LAUNCHED',flush=True)
def preflight():
    p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'preflight.json').exists();refs=L.references(p);obj=L.make(p);shadow=L.make(p);before=P.state(*obj)
    source=G.E.N.S.arrays(p['recovery_row']['path']);k=int(np.flatnonzero(source['ownership'])[0]);a={key:v for key,v in source.items()};a.update(states=source['states'][:k+8],teacher_actions=source['teacher_actions'][:k+8],ownership=source['ownership'][:k+8],stochastic=source['stochastic'][:k+8])
    aa=L.fit(obj,a,p,refs,lambda:None,lambda *_:None,steps=2);bb=L.fit(shadow,a,p,refs,lambda:None,lambda *_:None,steps=2);assert aa==bb and digest(P.state(*obj))==digest(P.state(*shadow));model=L.A.export(obj,OUT/'discarded_preflight_clone.pt.gz',p['model_bytes']);assert digest(P.checkpoint(model['path'])['joint_state'])==model['joint_hash']
    for z in [obj,shadow]:P.load(z,before);assert digest(P.state(*z))==p['training_initial_joint_hash']
    Path(model['path']).unlink();save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,commands=0,discarded_Actor_Adam=4,actual_recovery_prefix_and_labels_truncated_fixture_NOT_navigation=True,teacher_owner_mask_and_all19references=True,live_shadow_ActorAdam_rollback_serialization_exact=True,native_worker_contract_reused_from_actual_T526=True));print('T527_PREFLIGHT_PASS',flush=True)
def launch():
    p=read(OUT/'protocol.json');R.verify(p,True);pre=read(OUT/'preflight.json');assert pre['state']=='PASS' and not Path('/proc',str(pre['pid'])).exists() and not (OUT/'launch.json').exists()
    src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip() and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time());assert seconds>1170
    subprocess.run(['systemd-run','--user','--unit=stage-t527-validated-recovery-learning','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T527','--why=First validated actual recovery supervised learning with19successretention','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],check=True);print('T527_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
