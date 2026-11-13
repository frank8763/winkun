"""Register one daytime diagnostic; never restart a completed directory."""
from pathlib import Path
import json,math,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t512_runtime as R
G,P,torch,np,read,save,sha,digest=R.G,R.P,R.torch,R.np,R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/day_20261113'

def prepare():
    assert not OUT.exists()
    prior=ROOT/'stage/runs/t511_frozen_parent_residual'
    assert read(prior/'probe/audit.json')['state']=='PASS' if (prior/'probe/audit.json').exists() else read(prior/'pilot/audit.json')['state']=='PASS'
    assert all(not Path('/proc',str(i)).exists() for i in read(prior/'SEALED.json')['scientific_pids'])
    assert not Path('/proc',str(read(prior/'launch.json')['pid'])).exists()
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T511.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    old=read(prior/'protocol.json');accepted=ROOT/'stage/runs/t508_frozen_completion/pilot'
    assert read(accepted/'audit.json')['state']=='PASS'
    rows=read(accepted/'combined_field24_rows.json');dd=read(accepted/'diagnosis.json')
    choices=sorted([d for d in dd if d['timeout'] and not d['collision']],key=lambda d:(d['min_decision_distance_m'],d['id']))
    target=choices[0]['id'];assert target=='t502_field_sector1_offset1'
    row=next(r for r in rows if r['job_id']==target);assert sha(row['path'])==row['sha256']
    a=R.E.N.S.arrays(row['path']);assert all(digest(v)==row['array_hashes'][k] for k,v in a.items())
    t=np.r_[0.,np.cumsum(a['native_counts']*.002)]
    pos=np.vstack([a['poses'],a['next_poses'][-1]]);distance=np.linalg.norm(pos[:,:2]-a['task_goal_xy'],axis=1)
    closest=int(distance.argmin());start=max(1,closest-10)
    assert distance[closest]>.2 and not row['success'] and row['timeout']
    model=read(accepted/'summary.json')['result']['model'];assert sha(model['path'])==model['sha256']
    job=next(j for j in old['field_jobs'] if j['id']==target);cap=job['case']['step_cap']
    p={k:v for k,v in old.items() if k not in ['inputs','protected_files','repair_rows','keep_rows']}
    now=time.time();p.update(id='T512_FROZEN_ACTUAL_STATE_RECOVERY',out=str(OUT),initial=model,parent_model=model,
        baseline_row=row,probe_job=job,recovery_start_decision=start,prefix_physics_steps=int(a['native_counts'][:start].sum()),
        recovery_seeds=[[20261113512,i] for i in range(4)],max_bytes=8*2**20,main_seconds=900,learning_seconds=0,audit_seconds=300,
        max_physics=10*cap,max_commands=14,root_max_physics=10*cap,root_max_commands=20,preflight_commands=6,
        absolute_learning_stop=now,absolute_native_stop=now+1800,absolute_all_stop=now+3300,
        Actor_Adam=0,Value_Adam=0,new_policy_gradient_updates=0,
        scientific_scope='One frozen accepted field11 Actor. One deterministic full control exact original T508 arrays. Four fixed iid suffix innovations only after exact self-prefix closest-decision minus10. All10 episodes charged, no reset of hidden/history/clock/physics at branch. Four suffix samples conditional on ONE state, not independent full navigation validation. Zero training, no promotion.',
        authorization='Nov13 user requested daytime one-round-at-a-time experiment. Independent900s native300s audit bounded diagnostic, no extension of ended night T511.',
        hypotheses=['Stored original-start success labels may not cover the learner actual failure-state recovery distribution.',
                    'Near-goal steering/speed coordination may require local corrective exploration.',
                    'Successful local recovery may still not be learnable without interference in one global Actor. This round cannot resolve that third hypothesis.'])
    p['inputs']={f:h for f,h in old['inputs'].items() if Path(f).suffix=='.py'}
    for f in [*HERE.glob('*.py'),DOC/'T512_PLAN.md',DOC/'T512_PRIMARY_SOURCES.md',accepted/'summary.json',accepted/'audit.json',accepted/'diagnosis.json',accepted/'combined_field24_rows.json']:
        p['inputs'][str(f)]=sha(f)
    p['inputs'][row['path']]=row['sha256'];p['protected_files']={model['path']:model['sha256']};p['inputs'].update(p['protected_files'])
    shape=5*(1352*math.ceil(cap/100)+4480+65536)
    assert shape+3*2**20<p['max_bytes'] and G.M.used()+p['max_bytes']+512*1024<2**30
    OUT.mkdir();blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n'
    (HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    save(DOC/'T512_BUDGET.json',dict(native_max_seconds=900,audit_max_seconds=300,all_stop=p['absolute_all_stop'],max_physics=10*cap,all_commands=20,
         own_bytes=p['max_bytes'],full_array_shape_bound=shape,global_reserved=G.M.used()-G.M.used(OUT)+p['max_bytes'],CPU=[14,15],CPUQuota='100%',GPU=False,Adam=0))
    save(OUT/'feedback_loop.json',dict(state='REPRODUCIBLE_EXISTING_FROZEN_TIMEOUT',task=target,source_sha256=row['sha256'],min_decision_distance_m=float(distance[closest]),closest_seconds=float(t[closest]),prefix_seconds=float(t[start]),prefix_decision=start,
         command='taskset -c 14 ./run_stage.sh stage/experiments/t512_actual_state_recovery/t512_runtime.py main probe',pass_signal='Exact baseline control plus verified safe conditional recovery; model remains frozen.'))
    print('T512_REGISTERED_ZERO_TRAINING',target,'prefix_s',t[start],flush=True)

def preflight():
    p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'preflight.json').exists()
    a=R.E.N.S.arrays(p['baseline_row']['path']);q=R.frozen(p,p['initial']['path']);q.rng=np.random.default_rng([20261113512,999]);q.sample=True
    # Exercise boundary directly without thousands of synthetic GRU calls.
    original=R.F.L.frozen(p,p['initial']['path']);x=torch.tensor(a['states'][0:1]);k=p['recovery_start_decision']
    q.tape=[None]*(k-1)
    with torch.no_grad():
        q.step(x);assert int(q.tape[-1]['stochastic'])==0 and not q.tape[-1]['innovations'].any()
        q.step(x);assert int(q.tape[-1]['stochastic'])==1 and q.tape[-1]['innovations'].any()
        assert torch.equal(original.sequence(torch.tensor(a['states'][:12])),R.frozen(p,p['initial']['path']).sequence(torch.tensor(a['states'][:12])))
    folder=OUT/'preflight_native';folder.mkdir();counts=G.R.M.mp.get_context('spawn').RawArray('q',3)
    pool=R.Pool(dict(p,phase='student',out=str(folder),native_stop_epoch=time.time()+120,max_physics=1),counts,lambda:None)
    try:
        values=pool.request([dict(model=p['initial']['path'],job=p['probe_job'],seed=p['recovery_seeds'][i],cancel_probe=True) for i in range(2)])
        assert all(v['state']=='CANCEL_PASS' for v in values) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,Adam=0,commands=6,actual_workers_load_and_cancel=True,deterministic_prefix_and_iid_suffix_boundary_verified=True,all_workers_stopped=True))
    print('T512_PREFLIGHT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');R.verify(p,True)
    assert read(OUT/'preflight.json')['state']=='PASS' and not Path('/proc',str(read(OUT/'preflight.json')['pid'])).exists() and not (OUT/'launch.json').exists()
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert source['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time());assert seconds>2100
    args=['systemd-run','--user','--unit=stage-t512-actual-state-recovery','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T512','--why=Bounded daytime real-state recovery diagnostic','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')]
    subprocess.run(args,check=True);print('T512_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
