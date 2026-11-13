"""Register current-policy multi-start credit pilot, no unlimited relaunches."""
from pathlib import Path
import copy,json,math,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t523_runtime as R
G,L,P,torch,np,read,save,sha,digest=R.G,R.L,R.P,R.torch,R.np,R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/day_20261113'

def keep_record(row):return {k:row[k] for k in ['job_id','path','sha256','success','collision']}
def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t522_lunch_joint_relay'
    assert read(prior/'joint_pg/audit.json')['state']=='PASS' and read(prior/'joint_pg/summary.json')['result']['Actor_Adam']==0
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T522.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert all(not Path('/proc',str(i)).exists() for i in read(prior/'SEALED.json')['scientific_pids']+[read(prior/'launch.json')['pid']])
    p=read(prior/'protocol.json');parent=p['initial'];rows=G.rows(prior/'joint_pg');assert len(rows)==8
    assert all(r['policy_hash']==parent['policy_hash'] and r['loaded_model_sha256']==parent['sha256'] and r['full'] and sha(r['path'])==r['sha256'] for r in rows)
    assert [r['job_id'] for r in rows]==[j['id'] for j in p['training_jobs']]
    physics=2*sum(j['case']['step_cap'] for j in p['evaluation_jobs']);now=time.time();end=p['absolute_all_stop'];assert now+2460<end and now+1500<p['absolute_native_stop']
    p.update(id='T523_FIRST_JOINT_PG_WITH_FAILURE_SIGNAL',out=str(OUT),imported_rows=rows,epochs=16,preflight_commands=0,pilot_commands=68,pilot_max_physics=physics,root_max_physics=physics,root_max_commands=68,root_max_main_seconds=1500,max_bytes=6*2**20,model_bytes=2*2**20,
        scientific_scope='First Actor use of exact T522 unchanged T519 behavior batch,0 new navigation samples. All8 success/contact/timeout retained; no per-task success prerequisite. Two tasks equal-weight PG with within-task physicalRTG peers/fullcurrenthistory. Shared-gradient firstupdate cosine descriptive only. Same source Actor/Adam,sigma,lr,softreferences,reward and native retention gate.',
        acceptance='Samecandidateold8all8safeGOAL and all12T519fieldIDsretained/qualified>=12/contact<=9/0invalidcensor. Full32native+32exact,otherwisewholeActorAdamrollbackresearchinitial. Originalaccepted11untouched. old275/six54/eight54/fresh16/78null.',
        authorization='Nov13 user explicitly nextstart and asks forgetting. Independent bounded first actual jointPG, not restart of T522 or extrapolation of previous no-update arm. Existing13:17 all-stop unchanged, new68allcommands/0freshsamples/4MiB own.')
    p['inputs']={f:h for f,h in p['inputs'].items() if Path(f).suffix=='.py'};p['protected_files']={parent['path']:parent['sha256'],p['locked_best']['path']:p['locked_best']['sha256']}
    for f in [*HERE.glob('*.py'),DOC/'T523_PLAN.md',DOC/'T523_FORGETTING_PRIMARY_SOURCES.md',prior/'protocol.json',prior/'joint_pg/summary.json',prior/'joint_pg/audit.json',prior/'joint_pg/cases.jsonl',prior/'joint_pg/replays.jsonl',ROOT/'stage/docs/long_distance_20261101/release_receipt_T522.json']:
        p['inputs'][str(f)]=sha(f)
    for row in p['keep_rows']+rows:p['inputs'][row['path']]=row['sha256']
    p['inputs'].update(p['protected_files']);obj=L.make(p);state=P.checkpoint(parent['path'])['joint_state'];assert digest(P.state(*obj))==parent['joint_hash']==digest(state)
    reserve=p['model_bytes']+32*p['compact_bytes']+1280*1024;used=G.M.used();assert reserve<p['max_bytes'] and used+p['max_bytes']+65536<2**30
    OUT.mkdir();blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    save(DOC/'T523_BUDGET.json',dict(max_physics=physics,max_commands=68,main_seconds=1500,learning_seconds=600,audit_seconds=300,backup_reserve=600,all_stop=end,own_bytes=p['max_bytes'],all_data_model_partial_reserve=reserve,global_reserved=used+p['max_bytes'],max_live_PG=16,max_shadow_PG=16,max_audit_historical_Adam=16,preflight_discarded_max=8,ValueAdam=0,new_training_navigation_samples=0,imported_T522_eight_previous_updates_zero=True,CPU=[14,15],CPUQuota='100%',GPU=False))
    print('T523_REGISTERED_NOT_LAUNCHED',flush=True)

def preflight():
    p=read(OUT/'protocol.json');R.verify(p,True);P.frozen=L.A.frozen;assert not (OUT/'preflight.json').exists();obj=L.make(p);shadow=L.make(p);before=P.state(*obj);refs=L.references(p);arrays=[]
    source=G.E.N.S.arrays(p['keep_rows'][0]['path']);x=source['states'][:8]
    for i in range(8):
        q=L.make(p)[0].eval().requires_grad_(False);q.sample=True;q.rng=np.random.default_rng([20261113523,999,i]);h=None
        with torch.no_grad():
            for state in x:_,_,h,_=q.step(torch.tensor(state[None]),h)
        tape=q.tape;end=np.zeros((len(x),3),np.uint8);end[-1,1 if i%2==0 else 2]=1;reward=np.zeros(len(x));reward[-1]=1. if i%2==0 else -.5
        arrays.append(dict(states=x.copy(),next_states=np.concatenate([x[1:],x[-1:]]),ownership=np.zeros(len(x),np.uint8),stochastic=np.ones(len(x),np.uint8),end_flags=end,rewards=reward,native_counts=np.full(len(x),100),raw_actions=np.asarray([t['raw_actions'] for t in tape]),atom=np.asarray([t['atom'] for t in tape]),logp=np.asarray([t['logp'] for t in tape]),locations=np.asarray([t['locations'] for t in tape])))
    aa=L.fit(obj,arrays,p,refs,lambda:None,lambda *_:None,steps=2);bb=L.fit(shadow,arrays,p,refs,lambda:None,lambda *_:None,steps=2)
    assert aa==bb and digest(P.state(*obj))==digest(P.state(*shadow))
    assert any(not torch.equal(before['actor'][k],P.state(*obj)['actor'][k]) for k in before['actor'] if k.startswith(('encoder.','core.')))
    model=L.A.export(obj,OUT/'discarded_preflight_clone.pt.gz',p['model_bytes']);assert digest(P.checkpoint(model['path'])['joint_state'])==model['joint_hash']
    for o in [obj,shadow]:P.load(o,before);assert digest(P.state(*o))==p['initial']['joint_hash']
    # Exact same registered Actor/runtime/worker implementation already executed
    # both tasks in T522; no new worker commands or native steps in preflight.
    for row in p['imported_rows']:
        a=G.E.N.S.arrays(row['path']);R.M.audit_full_history(p,row,a)
    actual_items,diag=L.peer_batch(L.make(p),[G.E.N.S.arrays(row['path']) for row in p['imported_rows']],p)
    assert len(actual_items)==8 and {x['group'] for x in diag}=={0,1}
    # Serialization scratch is exactly reconstructible from this registered
    # synthetic fixture; not a scientific candidate or unique navigation data.
    clone_receipt=dict(path=model['path'],bytes=model['bytes'],sha256=model['sha256'],joint_hash=model['joint_hash'],recipe='preflight fixture seeds20261113523/999/i,actual2maxAdam;exactlive-shadow')
    Path(model['path']).unlink()
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,commands=0,discarded_synthetic_Actor_Adam=aa['Actor_Adam']+bb['Actor_Adam']+4,original_failed_discarded_Adam=4,fresh_discarded_Adam=aa['Actor_Adam']+bb['Actor_Adam'],synthetic_fixture_NOT_navigation=True,full_actor_encoder_core_really_updates=True,independent_Value_weights_and_Adam_unchanged=True,live_shadow_ActorAdam_rollback_serialization_exact=True,worker_contract_reused_from_actual_T522_NOT_new_test=True,all_actual_imported_history_density_verified=True,discarded_nonunique_serialization_scratch=clone_receipt))
    print('T523_FULL_ACTOR_ONPOLICY_PREFLIGHT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');R.verify(p,True);pre=read(OUT/'preflight.json');assert pre['state']=='PASS' and not Path('/proc',str(pre['pid'])).exists();assert not (OUT/'launch.json').exists()
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert source['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time());assert seconds>2460
    args=['systemd-run','--user','--unit=stage-t523-joint-failure-signal','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T523','--why=Bounded research12 single old-center navigation repair','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')]
    subprocess.run(args,check=True);print('T523_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
