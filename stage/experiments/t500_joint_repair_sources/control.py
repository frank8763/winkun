"""One finite causal ablation and source-completion study, zero optimizer."""
from pathlib import Path
import gzip,json,math,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t500_runtime as R
G,S,P,torch,read,save,sha,digest=R.G,R.S,R.P,R.torch,R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/review_20261112'

def ablation(p):
    old=P.checkpoint(p['initial']['path'])['joint_state'];q=S.Actor(p);q.load_state_dict(old['actor']);q.requires_grad_(False)
    v=P.L.Value(q,p['value_seed']);v.load_state_dict(old['value']);v.requires_grad_(False)
    ao=torch.optim.Adam(q.drive_adapter.parameters(),lr=.001,foreach=False,fused=False);ao.load_state_dict(old['actor_opt'])
    vo=torch.optim.Adam(v.parameters(),lr=p['value_lr'],foreach=False,fused=False);vo.load_state_dict(old['value_opt'])
    assert digest(P.state(q,v,ao,vo))==p['initial']['joint_hash']
    with torch.no_grad():q.drive_adapter[-1].weight.zero_();q.drive_adapter[-1].bias.zero_()
    return q,v,ao,vo

def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t499_frozen_joint_validation';p=read(prior/'protocol.json');assert read(prior/'regression/audit.json')['state']=='PASS';G.stopped(prior/'regression')
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T499.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert not Path('/proc',str(read(prior/'launch.json')['pid'])).exists()
    review=read(DOC/'T500_CAUSE_AND_SOURCE_REVIEW.json');assert review['native']==review['Adam']==0
    failed=[r for r in G.rows(prior/'regression') if not r['success']];assert len(failed)==43
    old=ROOT/'stage/runs/t449_offline_joint_recovery';historical=[json.loads(x) for x in gzip.decompress((old/'cases.jsonl.t481.gz').read_bytes()).decode().splitlines()]
    allrefs={r['job_id']:r for r in historical if r['role']=='FINAL_recovery' and r['success'] and not r['collision']}
    refs={r['job_id']:allrefs[r['job_id']] for r in failed if r['job_id'] in allrefs};assert len(refs)==42
    recovery=review['verified_actual_recovery_source_for_missing'];assert recovery['job_id']=='dev_task6_00'
    by={j['id']:j for j in p['old275_jobs']+p['six_jobs']}
    source_jobs=[by[r['job_id']] for r in failed if r['job_id'] in refs]
    ablation_jobs=[by[k] for k in ['dev_task4_00','dev_task6_00','wide117_80_dev6_LEFT_18']]
    ap=2*sum(j['case']['step_cap'] for j in ablation_jobs);sp=2*sum(j['case']['step_cap'] for j in source_jobs);now=time.time()
    for key in ['old275_jobs','six_jobs','eight_summary','storage_shape_bound']:p.pop(key,None)
    p.update(id='T500_FROZEN_FAILURE_ABLATION_AND_VERIFIED43_SOURCES',out=str(OUT),
             ablation_jobs=ablation_jobs,source_jobs=source_jobs,source_references=refs,failed_references={r['job_id']:r for r in failed},
             success_source_model=review['T449_known_success_teacher_model'],existing_recovery_row=recovery,
             max_bytes=48*2**20,ablation_max_physics=ap,sources_max_physics=sp,root_max_physics=ap+sp,root_max_commands=104,root_max_main_seconds=3000,
             absolute_learning_stop=now,absolute_native_stop=now+3600,absolute_all_stop=now+5400,
             preflight_commands=6,Actor_Adam=0,Value_Adam=0,
             scientific_scope='Zero-optimizer initial-condition diagnosis only. Three fixed failed representatives: zero final drive-adapter weights/bias only, all other learned parameters and saved matureAdam unchanged. Baselines are actual T499 full histories. Do not extrapolate3 results to all43.',
             source_scope='Reconstruct42 exact historical T449 successful task histories plus42 exact native repeats. Together with one existing verified actual successful studentprefix5+safeT255 recovery fordev_task6_00 form43 training sources, not one autonomous successful policy. No runtime teacher/router at deployment.',
             next_scope='No student training or new383validation in this pilot. Next joint repair needs two-command representational/loss contract and preservation of actual340 own successful histories. Do not claim source success equals student gain.',
             authorization='Nov12 user focuses wide6 and old tasks; bounded cause/source stage enables repair without blindly extending old speed-only training.')
    p['inputs']=dict(p['inputs']);p['protected_files']=dict(p['protected_files']);p['protected_files'][p['success_source_model']['path']]=p['success_source_model']['sha256']
    for f in [*HERE.glob('*.py'),DOC/'T500_CAUSE_AND_SOURCE_REVIEW.json',prior/'regression/summary.json',prior/'regression/audit.json',prior/'regression/cases.jsonl',old/'cases.jsonl.t481.gz',old/'summary.json',Path(recovery['path']),ROOT/'stage/docs/long_distance_20261101/release_receipt_T499.json']:
        p['inputs'][str(f)]=sha(f)
    assert G.M.used()+p['max_bytes']+2**20<2**30
    storage=sum(1352*math.ceil(j['case']['step_cap']/100)+4480+65536 for j in source_jobs+ablation_jobs)+10*2**20
    assert storage<p['max_bytes']
    OUT.mkdir();p['ablation_model']=S.export(ablation(p),OUT/'zero_drive_adapter.pt.gz',p['model_bytes'])
    blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    save(DOC/'T500_BUDGET.json',dict(max_physics=ap+sp,ablation_max_physics=ap,sources_max_physics=sp,max_commands=104,max_main_seconds=3000,ablation_main=600,source_main=2400,audits=420,all_seconds=5400,backup_sync_reserve=1500,own_bytes=p['max_bytes'],storage_bound=storage,global_reserved=G.M.used()-G.M.used(OUT)+p['max_bytes'],native_stop=p['absolute_native_stop'],all_stop=p['absolute_all_stop'],CPU=[14,15],CPUQuota='100%',GPU=False,Adam=0))
    print('T500_REGISTERED_NOT_LAUNCHED',flush=True)

def preflight():
    p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'preflight.json').exists();before=P.checkpoint(p['initial']['path'])['joint_state'];after=P.checkpoint(p['ablation_model']['path'])['joint_state']
    for key in ['value','actor_opt','value_opt']:assert digest(before[key])==digest(after[key])
    for key in before['actor']:
        if key in ['drive_adapter.2.weight','drive_adapter.2.bias']:assert not torch.count_nonzero(after['actor'][key])
        else:assert torch.equal(before['actor'][key],after['actor'][key])
    row=p['existing_recovery_row'];a=G.E.N.S.arrays(row['path']);assert sha(row['path'])==row['sha256'] and all(digest(v)==row['array_hashes'][k] for k,v in a.items())
    P.frozen=S.frozen;G.M.audit_full_history(dict(p,teacher_first=5),row,a)
    with torch.no_grad():
        actual=S.frozen(p,p['ablation_model']['path']).sequence(torch.tensor(a['states']));plain=S.PLAIN_FROZEN(p,p['inputs_initial_plain40']).sequence(torch.tensor(a['states'])) if 'inputs_initial_plain40' in p else None
        if plain is not None:assert torch.equal(actual,plain)
    assert S.frozen(p,p['success_source_model']['path']).phash()==p['success_source_model']['policy_hash']
    folder=OUT/'preflight_native';folder.mkdir();counts=G.R.M.mp.get_context('spawn').RawArray('q',3);pool=R.R.Pool(dict(p,phase='student',out=str(folder),native_stop_epoch=time.time()+120,max_physics=1),counts,lambda:None)
    try:
        vals=pool.request([dict(model=p['ablation_model']['path'],job=p['ablation_jobs'][0],seed=None,cancel_probe=True),dict(model=p['success_source_model']['path'],job=p['source_jobs'][-1],seed=None,cancel_probe=True)])
        assert all(v['state']=='CANCEL_PASS' for v in vals) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,commands=6,Adam=0,only_two_final_drive_adapter_tensors_zeroed=True,all_other_parameters_and_Adam_unchanged=True,actual_dev_task6_00_success_raw_full_history_audit_PASS=True,two_models_native_cancel_before_physics=True,all_workers_stopped=True))
    print('T500_PREFLIGHT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');R.verify(p,True);pre=read(OUT/'preflight.json');assert pre['state']=='PASS' and not Path('/proc',str(pre['pid'])).exists();assert not (OUT/'launch.json').exists()
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert source['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time());assert seconds>4900
    args=['systemd-run','--user','--unit=stage-t500-joint-repair-sources','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T500','--why=Focused oldtask and wide6 repair sources','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')]
    subprocess.run(args,check=True);print('T500_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
