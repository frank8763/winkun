"""Register one finite teacher-validation -> learned-drive Actor experiment."""
from pathlib import Path
import copy,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import speed_runtime as R
G,S,read,save,sha,digest=R.G,R.S,R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/review_20261112'

def prepare():
    assert not OUT.exists()
    prior=ROOT/'stage/runs/t495_drive_head_cause'
    assert read(prior/'audit.json')['state']=='PASS'
    R.G.stopped(prior);assert not Path('/proc',str(read(prior/'launch.json')['pid'])).exists()
    assert (prior/'local_manifest.json').exists()
    old=read(prior/'protocol.json');rows=G.rows(prior);left=[r for r in rows if '_LEFT_' in r['job_id']]
    assert len(rows)==54 and len(left)==27 and all(r['success'] and not r['collision'] for r in left)
    # Constant-bias pilot harmed RIGHT. Its checkpoint is a teacher source,
    # never the student initialization or a promoted navigation policy.
    assert read(prior/'summary.json')['result']['sides']['RIGHT']<22
    p=dict(old);p.pop('current',None);now=time.time()
    by={j['id']:j for j in p['wide8_jobs']};right=[j for j in p['wide8_jobs'] if j['case']['side']=='RIGHT']
    sentinels=[by['wide117_8m_dev6_RIGHT_00'],by['wide117_8m_dev6_RIGHT_17']]
    teacher_jobs=sentinels+[j for j in right if j not in sentinels]
    baseline=read(ROOT/'stage/runs/t493_wide8_night/raw_ppo_02/summary.json')['result']['candidate']
    p.update(id='T496_VERIFIED_TEMPORAL_TEACHERS_NEURAL_DRIVE_READOUT',out=str(OUT),
             initial=baseline['model'],initial_success_ids=baseline['success_ids'],
             left_teacher_rows=left,teacher_jobs=teacher_jobs,pulse_decisions=50,
             teacher_first=0,adapter_seed=20261112496,bc_steps=512,
             max_bytes=90*2**20,full_bytes=800*1024,compact_bytes=8192,model_bytes=1792*1024,
             root_max_physics=8640108,root_max_commands=180,root_max_main_seconds=6600,
             preflight_commands=6,absolute_learning_stop=now+5400,
             absolute_native_stop=now+7800,absolute_all_stop=now+9000,
             scientific_scope='One fixed temporal teacher profile, not a duration/amplitude/seed grid: RIGHT first10s adds existing raw_sigma0(.2) to clipped base40 drive command, thereafter unchanged base40. All steering remains the same neural function. Two preregistered sentinels must GOAL before remaining25. All27 GOAL+27 exact+audit before any student optimization.',
             student_scope='54 real GOAL histories:27 already-audited LEFT T495 plus27 newly verified RIGHT pulse. Frozen plain40 encoder/core/head, zero-initialized5953-parameter learned184->32->1 bounded drive-logit readout.512 fixed Adam plus512 exact shadow; independent audit reconstructs512. EqualLEFT/RIGHTepisode weight; half fullhistory/half first20s drive-action loss. No new policy gradient or Critic steps; not algorithm originality or from-scratch RL.',
             deployment='One120-input64-carry causal neuralActor directly outputs both commands. No side/time threshold branch, map, planner, waypoints, runtime teacher or router in student.',
             acceptance='Full54 valid, all initial40 success IDs preserved, count>40 and0contact. All54safe triggers new worker/copy54 exact; otherwise original14 exact. Not fullunder8 stability: old275/six/fresh16 remainnull.',
             authorization='Nov12 user explicitly requests diagnose then continue modifying to solve14; new bounded different neural speed-profile learning mechanism after uniform calibration regression, not restart or extra calibration grid.')
    p['inputs']=dict(old['inputs'])
    for f in [*HERE.glob('*.py'),prior/'audit.json',prior/'summary.json',prior/'cases.jsonl',prior/'local_manifest.json',Path(read(prior/'preflight.json')['model']['path'])]:p['inputs'][str(f)]=sha(f)
    for row in left:p['inputs'][row['path']]=row['sha256']
    p['protected_files']=dict(old['protected_files']);p['protected_files'][p['initial']['path']]=p['initial']['sha256']
    assert G.M.used()+p['max_bytes']+2**20<2**30
    assert 81*p['full_bytes']+4*p['model_bytes']+12*2**20<p['max_bytes']
    OUT.mkdir();blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    save(DOC/'T496_BUDGET.json',dict(max_physics=p['root_max_physics'],max_commands=180,max_main_seconds=6600,own_bytes=p['max_bytes'],global_reserved=G.M.used()-G.M.used(OUT)+p['max_bytes'],learning_stop=p['absolute_learning_stop'],native_stop=p['absolute_native_stop'],all_stop=p['absolute_all_stop'],teacher_main=2400,teacher_audit=300,student_main=3600,student_audit=600,CPU=[14,15],CPUQuota='100%',GPU=False))
    print('T496_REGISTERED_NOT_LAUNCHED',flush=True)

def preflight():
    p=read(OUT/'protocol.json');assert not (OUT/'preflight.json').exists();R.verify(p,True)
    obj,shadow=S.make(p),S.make(p);before=G.P.state(*obj);plain=S.PLAIN_FROZEN(p,p['initial']['path']);row=p['left_teacher_rows'][0];a=G.E.N.S.arrays(row['path']);x=S.torch.tensor(a['states'])
    with S.torch.no_grad():
        old=plain.sequence(x);new=obj[0].sequence(x);assert S.torch.equal(old,new)
        n=plain.normalized(x);h,_=plain.core(plain.encoder(n)[:,None]);h=h[:,0]
        features=S.torch.cat([h,n],-1);logit=plain.head(h)[:,0]
    # Explicit disposable synthetic clone contract, not recovered RIGHT labels.
    data=[dict(id='SYNTHETIC_PREFLIGHT_'+str(i),side=i//27,features=features,logit=logit,target=S.torch.tensor(a['actions'][:,0],dtype=S.torch.float64),early=S.torch.tensor(G.np.r_[0.,G.np.cumsum(a['native_counts']*.002)[:-1]]<20),source_sha256=row['sha256']) for i in range(54)]
    aa=S.fit(obj,data,p,lambda:None,lambda *_:None,steps=2);bb=S.fit(shadow,data,p,lambda:None,lambda *_:None,steps=2)
    assert aa==bb and digest(G.P.state(*obj))==digest(G.P.state(*shadow))
    for o in [obj,shadow]:G.P.load(o,before);assert digest(G.P.state(*o))==digest(before)
    model=S.export(obj,OUT/'preflight_model.pt.gz',p['model_bytes']);frozen=S.frozen(p,model['path']);h=None;values=[]
    with S.torch.no_grad():
        for xx in x[:24]:_,loc,h,_=frozen.step(xx[None],h);values.append(loc[0])
        error=float((S.torch.stack(values)-frozen.sequence(x[:24])).abs().max());assert error<1e-10
    folder=OUT/'preflight_native';folder.mkdir();counts=G.R.M.mp.get_context('spawn').RawArray('q',3)
    pp=dict(p,phase='student',out=str(folder),native_stop_epoch=time.time()+120,max_physics=1)
    pool=R.Pool(pp,counts,lambda:None)
    try:
        values=pool.request([dict(model=model['path'],job=j,seed=None,cancel_probe=True) for j in p['teacher_jobs'][:2]])
        assert all(v['state']=='CANCEL_PASS' for v in values) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,commands=6,discarded_Actor_Adam=4,Value_Adam=0,new_readout_zero_preserves_actual_plain40_function=True,sequence_step_error=error,synthetic_clone_update_and_full_Adam_rollback_exact=True,teacher_controls_NOT_yet_validated=True,all_workers_stopped=True))
    print('T496_PREFLIGHT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');R.verify(p,True);pre=read(OUT/'preflight.json');assert pre['state']=='PASS' and not Path('/proc',str(pre['pid'])).exists()
    assert not (OUT/'launch.json').exists();source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert source['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time());assert seconds>8000
    args=['systemd-run','--user','--unit=stage-t496-neural-speed','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T496','--why=Verified temporal teacher and learned neural drive readout','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')]
    subprocess.run(args,check=True)
    print('T496_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
