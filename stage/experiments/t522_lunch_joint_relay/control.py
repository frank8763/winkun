"""Register one finite two-mechanism lunch relay, no repeated old runs."""
from pathlib import Path
import datetime,json,math,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t522_runtime as R
G,L,P,torch,np,read,save,sha,digest=R.G,R.L,R.P,R.torch,R.np,R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/day_20261113'
def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t520_old_center_repair';pp=ROOT/'stage/runs/t519_fresh_full_actor_rl'
    assert read(prior/'pilot/audit.json')['state']=='PASS' and read(DOC/'T520_FINAL_REPORT.json')['remote']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert all(not Path('/proc',str(i)).exists() for i in read(prior/'SEALED.json')['scientific_pids']+[read(prior/'launch.json')['pid']])
    p=read(prior/'protocol.json');parent=p['initial'];old=next(j for j in p['old8_jobs'] if j['id']=='dev_task6_13');new=next(j for j in p['field_jobs'] if j['id']=='t502_field_sector0_offset2')
    def row(folder,id):return next(r for r in G.rows(folder/'pilot') if r['job_id']==id and r['role']=='AUTONOMOUS_FULL32')
    sources=[dict(job=old,model=read(prior/'pilot/summary.json')['result']['model'],expected=row(prior,old['id'])),dict(job=new,model=parent,expected=row(pp,new['id']))]
    assert all(s['expected']['success'] and not s['expected']['collision'] for s in sources)
    end=datetime.datetime(2026,11,13,13,17,tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp();now=time.time();assert end-now>4500
    evaluation=p['evaluation_jobs'];evphys=2*sum(j['case']['step_cap'] for j in evaluation)
    pgphys=8*(old['case']['step_cap']+new['case']['step_cap'])+evphys;bcphys=2*(old['case']['step_cap']+new['case']['step_cap'])+evphys
    p.update(id='T522_TWO_TASK_LUNCH_RELAY',out=str(OUT),initial=parent,parent_model=parent,initial_joint_locked=True,
        training_jobs=[old]*4+[new]*4,training_seeds=[[20261113522,g,i] for g in range(2) for i in range(4)],bc_sources=sources,
        registered_order=['joint_pg','success_bc'],phase_limits={'joint_pg':dict(main_seconds=1500,audit_seconds=300,learning_seconds=600,max_physics=pgphys,max_commands=84,mechanism='joint_pg',epochs=16),
        'success_bc':dict(main_seconds=1800,audit_seconds=600,learning_seconds=600,max_physics=bcphys,max_commands=72,mechanism='success_bc',epochs=64)},
        max_bytes=15*2**20,root_max_physics=pgphys+bcphys,root_max_commands=162,root_max_main_seconds=3300,
        absolute_all_stop=end,absolute_native_stop=end-660,absolute_learning_stop=end-1200,
        preflight_commands=6,model_bytes=2*2**20,
        scientific_scope='Two fixed mechanisms, both initialized from exact T519research12old7 including complete ActorAdam. A two actual tasks4+4 fresh, per-task peer RTG and equal episode/task PG; no pooling by length. B rebuild two actually verified deterministic source successes(T520old/T519field), all arrays/integration must match originals; two equal-task fullhistory location BC64. Same32real behavior+32exact acceptance. No teacher/map/router at deployment. No seed/rank/learning-rate grid. Not novel algorithm or proof.',
        acceptance='SameActor old8all8 and all initial12 field IDs retained, qualifiedfield>=12,contacts<=9 and noinvalid/censor. Does not claim fullfield24/old275/six54/eight54/fresh/reload78. Each rejection entireActorAdamrollbackT519. FrozenacceptedT50711 untouched. Stop relay on first accepted branch; otherwise second distinct mechanism only, no extra retries.',
        authorization='User Nov13 check and start two-hour experiment while lunch. Fixed all-stop13:17 Beijing, earlier branch deadlines stricter. No network/LLM needed for scientific relay; remote backups after all native/audits.')
    p['inputs']={f:h for f,h in p['inputs'].items() if Path(f).suffix=='.py'}
    p['protected_files']={parent['path']:parent['sha256'],p['locked_best']['path']:p['locked_best']['sha256'],sources[0]['model']['path']:sources[0]['model']['sha256']}
    for f in [*HERE.glob('*.py'),DOC/'T522_PLAN.md',DOC/'LUNCH_RELAY_PRIMARY_SOURCES.md',DOC/'T521_DISPOSABLE_PREFLIGHT_CLEANUP.json',prior/'pilot/summary.json',prior/'pilot/audit.json',pp/'pilot/summary.json']:
        p['inputs'][str(f)]=sha(f)
    for rr in p['keep_rows']:p['inputs'][rr['path']]=rr['sha256']
    p['inputs'].update(p['protected_files'])
    full=lambda j:1352*math.ceil(j['case']['step_cap']/100)+4480+65536
    reserve=4*(full(old)+full(new))+full(old)+full(new)+2*p['model_bytes']+64*p['compact_bytes']+2*2**20
    used=G.M.used();assert reserve<p['max_bytes'] and used+p['max_bytes']+64*1024<2**30,(reserve,used,p['max_bytes'])
    OUT.mkdir();blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    assert digest(P.state(*L.make(p)))==parent['joint_hash']
    save(DOC/'T522_BUDGET.json',dict(max_physics=p['root_max_physics'],max_commands=162,max_main_seconds=3300,max_audit_seconds=900,backup_seconds=600,own_bytes=p['max_bytes'],all_data_models_partial_reserve=reserve,global_reserved=used+p['max_bytes'],all_stop=end,CPU=[14,15],CPUQuota='100%',GPU=False,phases=p['phase_limits'],missing_is_null=True))
    print('T522_REGISTERED_NOT_LAUNCHED',flush=True)
def preflight():
    p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'preflight.json').exists();x=G.E.N.S.arrays(p['keep_rows'][0]['path'])['states'][:8];counts=[];last=None
    for mechanism in ['joint_pg','success_bc']:
        plan=dict(p,mechanism=mechanism);arrays=[]
        for i in range(8 if mechanism=='joint_pg' else 2):
            q=L.make(p)[0].eval().requires_grad_(False);q.sample=mechanism=='joint_pg';q.rng=np.random.default_rng([20261113522,999,i]);h=None
            with torch.no_grad():
                for state in x:_,_,h,_=q.step(torch.tensor(state[None]),h)
            tape=q.tape;end=np.zeros((len(x),3),np.uint8);end[-1,1 if mechanism=='success_bc' or i%2==0 else 2]=1;reward=np.zeros(len(x));reward[-1]=1. if i%2==0 else -.5
            arrays.append(dict(states=x.copy(),next_states=np.concatenate([x[1:],x[-1:]]),ownership=np.zeros(len(x),np.uint8),stochastic=np.full(len(x),mechanism=='joint_pg',np.uint8),end_flags=end,rewards=reward,native_counts=np.full(len(x),100),raw_actions=np.asarray([t['raw_actions'] for t in tape]),atom=np.asarray([t['atom'] for t in tape]),logp=np.asarray([t['logp'] for t in tape]),locations=np.asarray([t['locations'] for t in tape])+(0.001 if mechanism=='success_bc' else 0.)))
        obj,shadow=L.make(p),L.make(p);before=P.state(*obj);refs=L.references(p)
        aa=L.fit(obj,arrays,plan,refs,lambda:None,lambda *_:None,steps=2);bb=L.fit(shadow,arrays,plan,refs,lambda:None,lambda *_:None,steps=2)
        assert aa==bb and digest(P.state(*obj))==digest(P.state(*shadow));counts.append(aa['Actor_Adam']+bb['Actor_Adam']);last=L.A.export(obj,OUT/'discarded_preflight_clone.pt.gz',p['model_bytes'])
        assert digest(P.checkpoint(last['path'])['joint_state'])==last['joint_hash']
        for o in [obj,shadow]:P.load(o,before);assert digest(P.state(*o))==p['initial']['joint_hash']
        Path(last['path']).unlink()
    folder=OUT/'preflight_native';folder.mkdir();native=G.R.M.mp.get_context('spawn').RawArray('q',3);pool=R.F.Pool(dict(p,phase='student',out=str(folder),native_stop_epoch=time.time()+120,max_physics=1),native,lambda:None)
    try:assert all(v['state']=='CANCEL_PASS' for v in pool.request([dict(model=p['initial']['path'],job=j,seed=None,cancel_probe=True) for j in [p['training_jobs'][0],p['training_jobs'][4]]])) and sum(native)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,commands=6,discarded_synthetic_Actor_Adam=sum(counts),both_mechanisms_live_shadow_and_complete_Adam_rollback_exact=True,synthetic_NOT_navigation=True,no_unique_clone_deleted=True,all_worker_load_cancel_PASS=True))
    print('T522_BOTH_MECHANISMS_PREFLIGHT_PASS',flush=True)
def launch():
    p=read(OUT/'protocol.json');R.verify(p,True);pre=read(OUT/'preflight.json');assert pre['state']=='PASS' and not Path('/proc',str(pre['pid'])).exists() and not (OUT/'launch.json').exists()
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert source['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip() and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time());assert seconds>4500
    subprocess.run(['systemd-run','--user','--unit=stage-t522-lunch-joint-relay','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T522','--why=Two bounded lunch learning mechanisms','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],check=True)
    print('T522_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
