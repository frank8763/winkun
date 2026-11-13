"""Register current-policy multi-start credit pilot, no unlimited relaunches."""
from pathlib import Path
import copy,json,math,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t524_runtime as R
G,L,P,torch,np,read,save,sha,digest=R.G,R.L,R.P,R.torch,R.np,R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/day_20261113'

def keep_record(row):return {k:row[k] for k in ['job_id','path','sha256','success','collision']}
def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t523_joint_failure_signal'
    assert read(prior/'pilot/audit.json')['state']=='PASS' and not read(prior/'pilot/summary.json')['result']['accepted']
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T523.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert all(not Path('/proc',str(i)).exists() for i in read(prior/'SEALED.json')['scientific_pids']+[read(prior/'launch.json')['pid']])
    p=read(prior/'protocol.json');parent=p['initial'];rows=p['imported_rows'];assert len(rows)==8
    now=time.time();assert now+1800<p['absolute_all_stop'] and now+1200<p['absolute_native_stop']
    diag=read(prior/'pilot/summary.json')['result']['update']['initial_task_gradient'];norms=[diag['old_task_norm'],diag['new_task_norm']];assert min(norms)>0;scales=[sum(norms)/(2*n) for n in norms]
    p.update(id='T524_PAIRED_TASK_PG_INITIAL_NORM_BALANCE',task_pg_scales=scales,out=str(OUT),max_bytes=4*2**20,root_max_main_seconds=1200,
        scientific_scope='Explicit paired same-eight-batch control against T523,NOT fresh/independent data. Same source Actor, two tasks+weighting, reward/sigma/lr/16epochs/softreferences/native32gate. ONLY change fixed task PG scales from measured initial T523 norms:sum(norms)/(2*norm);all Actor and Value Adam states remain exact. Separate research clone; never overwrite source optimizer.',
        authorization='Nov13user check,analyzecause,andcontinue. Newmechanism tests observed task gradient magnitude imbalance,not additional rounds of equal-weight PG. Existing13:17 hardstop unchanged,0newtrainnative/68commands/4MiB own.')
    p['inputs']={f:v for f,v in p['inputs'].items() if Path(f).suffix=='.py'}
    for f in [*HERE.glob('*.py'),DOC/'T524_PLAN.md',prior/'pilot/summary.json',prior/'pilot/audit.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T523.json']:
        p['inputs'][str(f)]=sha(f)
    for row in p['keep_rows']+rows:p['inputs'][row['path']]=row['sha256']
    p['inputs'].update(p['protected_files']);obj=L.make(p);p['training_initial_joint_hash']=digest(P.state(*obj));assert p['training_initial_joint_hash']==parent['joint_hash'] and len(obj[2].state)==len(L.variables(obj[0]))
    reserve=p['model_bytes']+32*p['compact_bytes']+1280*1024;used=G.M.used();assert reserve<p['max_bytes'] and used+p['max_bytes']+65536<2**30
    OUT.mkdir();blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    save(DOC/'T524_BUDGET.json',dict(max_physics=p['pilot_max_physics'],max_commands=68,main_seconds=1200,learning_seconds=600,audit_seconds=180,backup_reserve=360,all_stop=p['absolute_all_stop'],own_bytes=p['max_bytes'],all_data_model_partial_reserve=reserve,global_reserved=used+p['max_bytes'],max_live_PG=16,max_shadow_PG=16,max_audit_historical_Adam=16,preflight_discarded_max=4,ValueAdam=0,new_training_navigation_samples=0,paired_T523_samebatch_NOT_independent=True,CPU=[14,15],CPUQuota='100%',GPU=False))
    print('T524_REGISTERED_NOT_LAUNCHED',flush=True)

def preflight():
    p=read(OUT/'protocol.json');R.verify(p,True);P.frozen=L.A.frozen;assert not (OUT/'preflight.json').exists();obj=L.make(p);shadow=L.make(p);before=P.state(*obj);refs=L.references(p);arrays=[]
    source=G.E.N.S.arrays(p['keep_rows'][0]['path']);x=source['states'][:8]
    for i in range(8):
        q=L.make(p)[0].eval().requires_grad_(False);q.sample=True;q.rng=np.random.default_rng([20261113524,999,i]);h=None
        with torch.no_grad():
            for state in x:_,_,h,_=q.step(torch.tensor(state[None]),h)
        tape=q.tape;end=np.zeros((len(x),3),np.uint8);end[-1,1 if i%2==0 else 2]=1;reward=np.zeros(len(x));reward[-1]=1. if i%2==0 else -.5
        arrays.append(dict(states=x.copy(),next_states=np.concatenate([x[1:],x[-1:]]),ownership=np.zeros(len(x),np.uint8),stochastic=np.ones(len(x),np.uint8),end_flags=end,rewards=reward,native_counts=np.full(len(x),100),raw_actions=np.asarray([t['raw_actions'] for t in tape]),atom=np.asarray([t['atom'] for t in tape]),logp=np.asarray([t['logp'] for t in tape]),locations=np.asarray([t['locations'] for t in tape])))
    aa=L.fit(obj,arrays,p,refs,lambda:None,lambda *_:None,steps=2);bb=L.fit(shadow,arrays,p,refs,lambda:None,lambda *_:None,steps=2)
    assert aa==bb and digest(P.state(*obj))==digest(P.state(*shadow))
    assert any(not torch.equal(before['actor'][k],P.state(*obj)['actor'][k]) for k in before['actor'] if k.startswith(('encoder.','core.')))
    model=L.A.export(obj,OUT/'discarded_preflight_clone.pt.gz',p['model_bytes']);assert digest(P.checkpoint(model['path'])['joint_state'])==model['joint_hash']
    for o in [obj,shadow]:P.load(o,before);assert digest(P.state(*o))==p['training_initial_joint_hash']
    # Exact same registered Actor/runtime/worker implementation already executed
    # both tasks in T522; no new worker commands or native steps in preflight.
    for row in p['imported_rows']:
        a=G.E.N.S.arrays(row['path']);R.M.audit_full_history(p,row,a)
    actual_items,diag=L.peer_batch(L.make(p),[G.E.N.S.arrays(row['path']) for row in p['imported_rows']],p)
    assert len(actual_items)==8 and {x['group'] for x in diag}=={0,1}
    # Serialization scratch is exactly reconstructible from this registered
    # synthetic fixture; not a scientific candidate or unique navigation data.
    clone_receipt=dict(path=model['path'],bytes=model['bytes'],sha256=model['sha256'],joint_hash=model['joint_hash'],recipe='preflight fixture seeds20261113524/999/i,actual2maxAdam;exactlive-shadow')
    Path(model['path']).unlink()
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,commands=0,discarded_synthetic_Actor_Adam=aa['Actor_Adam']+bb['Actor_Adam'],synthetic_fixture_NOT_navigation=True,full_actor_encoder_core_really_updates=True,independent_Value_weights_and_Adam_unchanged=True,live_shadow_ActorAdam_rollback_serialization_exact=True,worker_contract_reused_from_actual_T522_NOT_new_test=True,all_actual_imported_history_density_verified=True,discarded_nonunique_serialization_scratch=clone_receipt))
    print('T524_FULL_ACTOR_ONPOLICY_PREFLIGHT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');R.verify(p,True);pre=read(OUT/'preflight.json');assert pre['state']=='PASS' and not Path('/proc',str(pre['pid'])).exists();assert not (OUT/'launch.json').exists()
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert source['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time());assert seconds>1800
    args=['systemd-run','--user','--unit=stage-t524-task-gradient-balance','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T524','--why=Bounded research12 single old-center navigation repair','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')]
    subprocess.run(args,check=True);print('T524_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
