"""Real current-policy histories and conditional density checks; no steps."""
from pathlib import Path
import ast,copy,json,sys,time
import t655_runtime as T
ROOT,HERE,OUT,DOC=T.ROOT,T.HERE,T.OUT,T.DOC
M,R,torch,np=T.M,T.R,T.torch,T.np

def main():
    assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True)
    src=ROOT/'stage/runs/t653_afternoon_outcome_constraints';p=T.read(ROOT/'stage/runs/t654_verified_deviation_recovery/protocol.json');p.pop('registration_epoch',None);p.pop('_root_sha256',None)
    source=T.read(src/'protocol.json');old=T.Shared.load(source,lambda:None);assert len(old)==384
    rows=[json.loads(v) for v in T.gzip.decompress((ROOT/'stage/runs/t654_verified_deviation_recovery/shared/cases.jsonl.gz').read_bytes()).decode().splitlines()]
    field=p['existing_field_full_rows']+[r for r in rows if r['role']=='field_reconstruct'];assert len(field)==8
    original=[copy.deepcopy(next(j for j in p['evaluation_jobs'] if j['id']==key)) for key in p['protected_tasks']]
    p.update(id='T655_CURRENT_POLICY_MATCHED_OLD_RECOVERY_SUPPORT',out=str(OUT),authorization_approved=False,arms=['joint'],arm='shared',methods=['original','recovery'],evaluation_priority=['recovery','original'],
        old_source_protocol=str(src/'protocol.json'),original_jobs=original,field_train_jobs=[copy.deepcopy(next(j for j in p['field_jobs'] if j['id']==key)) for key in p['required_field_ids']],field_keep_rows=field,
        fit_steps=8,fit_lr=.00003,keep_weight=10.,new_keep_weight=10.,Actor_each_cap=16,total_max_physics=29500000,total_max_commands=1050,total_main_seconds=5100,
        shared_main_seconds=1100,shared_audit_seconds=240,main_seconds=4000,audit_seconds=600,learning_seconds=300,full_evaluation_reserve_seconds=1700,
        max_physics=24200000,max_commands=860,max_bytes=640*2**20,shared_bytes=272*2**20,child_bytes=256*2**20,
        new_Actor_Adam=16,shadow_Actor_Adam=16,independent_historical_Actor_Adam=16,new_RL=16,new_Critic=0,preflight_optimizer=0,new_teacher=0)
    prior=T.read(ROOT/'stage/runs/t654_verified_deviation_recovery/LEDGER.json')
    assert prior['Actor_live_each']==0
    p.update(prior_physics=prior['physics'],prior_commands=prior['commands'],prior_main_seconds=prior['main_seconds'],
        combined_physics_cap=30000000,combined_command_cap=1100,combined_main_cap=6300,combined_Actor_cap=64,combined_storage_cap=656*2**20)
    p['shared_physics']=32*sum(j['case']['step_cap'] for j in original)+8*sum(j['case']['step_cap'] for j in p['field_train_jobs'])
    worst=p['shared_physics']+2*sum(j['case']['step_cap'] for j in p['evaluation_jobs']+p['key_jobs']);assert worst<=p['total_max_physics']
    storage=80*p['fresh_bytes']+80*p['compact_bytes']+p['full_bytes']+12*2**20+2*(20*p['fresh_bytes']+p['model_bytes']+407*p['compact_bytes']+p['full_bytes']+20*2**20)+12*2**20
    assert storage<=p['max_bytes']
    assert p['prior_physics']+worst<=p['combined_physics_cap'] and p['prior_commands']+1026<=p['combined_command_cap']
    assert p['prior_main_seconds']+p['total_main_seconds']<=p['combined_main_cap']
    assert R.used(ROOT/'stage/runs/t654_verified_deviation_recovery')+p['max_bytes']<=p['combined_storage_cap']
    R.B.setup(p);q=R.L.frozen(p,p['initial']['path']);assert R.L.fixed_hash(q)==R.L.fixed_hash(R.L.frozen(source,source['initial']['path']))
    fields=[]
    for row in field:
        assert T.sha(row['path'])==row['sha256'];a=R.E.N.S.arrays(row['path']);T.Native.inspect(p,a,row)
        g=M.features(q,a,row);g['target']=torch.tensor(a['locations']);fields.append(g)
    # Existing noisy T501 episodes supply ONLY numerical fixtures. They are
    # relabelled with current mean densities in RAM, never training samples.
    rr=T.read(src/'shared/summary.json');assert rr['state']=='COMPLETE'
    actual=[json.loads(v) for v in T.gzip.decompress((src/'shared/cases.jsonl.gz').read_bytes()).decode().splitlines()]
    row=next(r for r in actual if r['role']=='shared_fresh');a=R.E.N.S.arrays(row['path']);fixed=M.digest(q.state_dict())
    with torch.no_grad():
        loc=q.sequence(torch.tensor(a['states'])).numpy();b=copy.deepcopy(a);b['locations']=loc
        b['logp']=M.P.L.density(torch.tensor(loc),torch.tensor(b['raw_actions']),torch.tensor(b['atom']),p['raw_sigma']).numpy()
        conditional=copy.deepcopy(b);k=7;conditional['recovery_first']=np.asarray(k,np.int64);conditional['ownership'][:k]=1;conditional['stochastic'][:k]=0;conditional['logp'][:k]=0
    original_fixture=[(b,dict(row,success=bool(i%2),collision=False,field_boundary=False)) for i in range(8)]
    suffix_fixture=[(conditional,dict(row,success=bool(i%2),collision=False,field_boundary=False)) for i in range(8)]
    groups=[('old_fixture',original_fixture),('suffix_fixture',suffix_fixture)]
    R.gpu_guard();T.os.environ['CUDA_VISIBLE_DEVICES']='0';T.os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';obj=M.make(p);q,opt=obj;before=M.state(obj);bh=M.digest(before)
    items,diag=M.build(q,groups,p);loss=sum(M.task_loss(q,g,p) for g in items);loss.backward()
    variables=[v for v in q.parameters() if v.requires_grad];assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in variables)
    # Paired fixture advantages sum tozero; replace one advantage numerically
    # to verify nonzero prefix-masked gradient. This is not scientific data.
    opt.zero_grad(set_to_none=True);items[1][0]['adv']=1.;probe=M.task_loss(q,items[1],p);probe.backward()
    grad=float(sum(v.grad.square().sum() for v in variables).sqrt());assert grad>0
    assert M.digest(M.state(obj))==bh and not opt.state
    cached=M.features(q,b,row)
    with torch.no_grad():
        q.steer_adapter[-1].bias.add_(.01);difference=float((q.sequence(torch.tensor(b['states'],device='cuda'))-M.location(q,cached)).abs().max());assert difference<1e-10
        for v in variables:opt.state[v]=dict(step=torch.tensor(5.),exp_avg=torch.full_like(v,.03125),exp_avg_sq=torch.full_like(v,.0625))
        M.load(obj,before);assert M.digest(M.state(obj))==bh and not opt.state and M.L.fixed_hash(q)==M.L.fixed_hash(R.L.frozen(p,p['initial']['path']))
    q.cpu();del obj,q,items;torch.cuda.empty_cache()
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    files=set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PRIMARY_REVIEW.md',ROOT/'stage/runs/t654_verified_deviation_recovery/FINAL_REPORT_FINAL.json'}|{Path(r['path']) for r in field}
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):files.add(Path(f).resolve())
    p['inputs'].update({str(f):T.sha(f) for f in files})
    for f,h in p['inputs'].items():assert T.sha(f)==h,f
    T.save(HERE/'protocol.json',p);global_worst=R.used(ROOT/'stage/runs')+p['max_bytes'];assert global_worst<p['global_bytes']
    T.save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',all384_plus8_current_full_history_checked=True,
        old_noisy_data_only_relabelled_numeric_fixture_not_training=True,variable_group_size_and_masked_suffix_density_checked=True,
        actual_GPU_conditional_gradient=grad,whole_sequence_changed_adapter_cache_equivalence=difference,mature_synthetic_full_rollback=True,
        new_native=0,new_optimizer=0,native_worst=worst,commands_worst=1026,storage_worst=storage,global_worst=global_worst))
    T.save(DOC/'BUDGET.json',{k:p[k] for k in ['total_max_physics','total_max_commands','total_main_seconds','Actor_each_cap','max_bytes','global_bytes','shared_physics','shared_main_seconds','shared_audit_seconds','main_seconds','audit_seconds','learning_seconds','backup_seconds','source_seconds','absolute_learning_stop','absolute_native_stop','absolute_all_stop','prior_physics','prior_commands','prior_main_seconds','combined_physics_cap','combined_command_cap','combined_main_cap','combined_Actor_cap','combined_storage_cap']})
    print('T655_ZERO_NATIVE_ZERO_OPTIMIZER_PASS',worst,storage,grad,flush=True)
if __name__=='__main__':main()
