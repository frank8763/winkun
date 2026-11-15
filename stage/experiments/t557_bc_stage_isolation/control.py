"""Prepare the smallest frozen stage-attribution experiment; no optimizer."""
from pathlib import Path
import json,sys,time,shutil
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t557_runtime as R
ROOT,OUT,DOC,L,read,save,sha=R.ROOT,R.OUT,R.DOC,R.L,R.read,R.save,R.sha
IDS=['dev_task6_13','t502_field_sector0_offset2','t502_field_sector1_offset1','t502_field_sector1_offset2','t502_field_sector2_offset2','t502_field_sector4_offset2']

def build():
    old=ROOT/'stage/runs/t555_adam_direction_control';root=read(old/'protocol.json');assert read(old/'SEALED.json')['all_scientific_processes_stopped'];models={};post={}
    for arm in ['project','scalar']:
        f=old/('round_00_'+arm);assert read(f/'audit.json')['state']=='PASS';models[arm]=read(f/'summary.json')['result']['behavior'];by={r['job_id']:r for r in R.G.rows(f)[4:]};post[arm]={i:by[i] for i in IDS}
    by={j['id']:j for j in root['evaluation_jobs']};baseline={r['job_id']:r for r in R.G.rows(ROOT/'stage/runs/t552_joint_robust_night/final407')}
    p=dict(root);p.update(id='T557_FROZEN_BC_STAGE_ISOLATION_NO_LEARNING',models=models,post_RL_rows=post,baseline_rows={i:baseline[i] for i in IDS},job_ids=IDS,jobs=[by[i] for i in IDS],max_physics=1257916,max_commands=32,max_main_seconds=1800,main_seconds=1800,learning_seconds=0,audit_seconds=300,transport_seconds=300,source_seconds=180,max_bytes=17*2**20,child_bytes=17*2**20,scope='TwoalreadyfrozenBCcheckpoints6fixedconsumedtaskseach+12exact. CurrentT555baselineandpostRLoutcomesreusedsameinitialnativeSHA. NoActor/Value/Adam/teacher/noiseseed/thresholdchanges. OnlylocateBCvsRLstageloss;notnewlearnerorfullgate. Allhistorysaved; noautomatictrainingfollowon.')
    for k in ['evaluation_jobs','final_jobs','training_jobs','before_check','restore_rows','keep_rows','neighbor_rows','pair_rows','arms']:p.pop(k,None)
    return p

def preflight():
    assert not OUT.exists() and not (DOC/'PREFLIGHT.json').exists();DOC.mkdir(parents=True,exist_ok=True);p=build();R.torch.set_num_threads(1);R.torch.use_deterministic_algorithms(True)
    cfg=read(p['actor_configuration']);cfg.update(p);rows=[]
    source=read(ROOT/'stage/runs/t555_adam_direction_control/protocol.json')
    for name,m in p['models'].items():
        assert sha(m['path'])==m['sha256'];q=L.frozen(cfg,m['path']);assert q.phash()==m['policy_hash'];a=R.E.N.S.arrays(source['keep_rows'][0]['path']);x=R.torch.tensor(a['states'][:16]);f,base=L.feature_mean(q,x)
        with R.torch.no_grad():loc=base+q.p['readout_amplitude']*R.torch.tanh(q.extra(f));seq=q.sequence(x)
        assert R.torch.allclose(loc,seq,atol=1e-10,rtol=0);x2=x.clone();x2[8:]+=1;assert R.torch.equal(q.sequence(x2)[:8],seq[:8]);rows.append(dict(arm=name,policy_hash=q.phash(),feature_and_actual_sequence_maxerr=float((loc-seq).abs().max())))
    bound=4*sum(j['case']['step_cap'] for j in p['jobs']);assert bound==p['max_physics']
    prior=read(ROOT/'stage/docs/day_20261115/t555/FINAL_REPORT.json')['ledger'];assert prior['physics']+bound<=19_000_000 and prior['commands']+32<=600 and prior['main_seconds']+1800<=12300
    used=R.used(ROOT/'stage/runs');assert used+p['max_bytes']+131072<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    # 12 full files, one atomic full transient, two worker partials and 2MiB
    # metadata exhaustively reserved; each full schema bound reuses audited
    # T555 same max642 decisions and <=859664 raw upper, cap1MiB.
    storage=dict(full12=12*2**20,atomic_full=2**20,partial2=2*2**20,metadata=2*2**20);assert sum(storage.values())==p['max_bytes']
    save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',epoch=time.time(),new_native=0,new_optimizer=0,model_sequence=rows,source={str(f):sha(f) for f in HERE.glob('*.py')},storage=storage,global_worst=used+p['max_bytes']+131072,proposed_physics=bound,proposed_commands=32,proposed_day_combined_physics=prior['physics']+bound,proposed_day_combined_commands=prior['commands']+32,requires_authorized_scope_reallocation=True,no_registration_or_launch=True,not_repeating_T555_training=True));print('T557_ZERO_READY',bound,flush=True)

def prepare():
    assert not OUT.exists();pref=read(DOC/'PREFLIGHT.json');assert pref['state']=='PASS_ZERO_NATIVE_ZERO_OPTIMIZER';assert all(sha(f)==h for f,h in pref['source'].items())
    a=read(DOC/'AUTHORIZATION.json');assert a['scope_reallocation_approved'] and a['max_physics']==1257916 and a['max_commands']==32
    p=build();p.update(absolute_all_stop=a['all_stop_epoch'],absolute_native_stop=a['native_stop_epoch'],absolute_learning_stop=a['learning_stop_epoch'])
    for m in p['models'].values():p['protected_files'][m['path']]=m['sha256'];p['inputs'][m['path']]=m['sha256']
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
    for f in [*HERE.glob('*.py'),DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC/'AUTHORIZATION.json',ROOT/'stage/docs/day_20261115/T556_RAW_EVIDENCE.json']:p['inputs'][str(f)]=sha(f)
    assert R.used(ROOT/'stage/runs')+p['max_bytes']+131072<2**30 and time.time()+1800+300+300+180+60<p['absolute_all_stop']
    OUT.mkdir();blob=json.dumps(p,separators=(',',':'),allow_nan=False)+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True);print('T557_REGISTERED_NOT_LAUNCHED')
if __name__=='__main__':globals()[sys.argv[1]]()
