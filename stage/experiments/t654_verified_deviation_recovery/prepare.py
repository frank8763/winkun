"""Full existing histories, native schema and cached gradient checks; zero steps."""
from pathlib import Path
import ast,copy,json,sys,time
import t654_runtime as T
ROOT,HERE,OUT,DOC=T.ROOT,T.HERE,T.OUT,T.DOC
M,R,torch,np=T.M,T.R,T.torch,T.np

def main():
    assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True)
    src=ROOT/'stage/runs/t653_afternoon_outcome_constraints';p=T.read(src/'protocol.json');source=copy.deepcopy(p)
    dep=T.read(DOC/'ACTUAL_DEPARTURES.json');old=T.Shared.load(source,lambda:None);assert len(old)==384
    rows=[json.loads(v) for v in T.gzip.decompress((src/'mean/cases.jsonl.gz').read_bytes()).decode().splitlines()]
    screen=T.read(src/'mean/summary.json')['screen'];field_ids=[r['job_id'] for r in screen['rows'] if r['job_id'].startswith('t502') and T.safe(r)];assert len(field_ids)==8
    full=[r for r in rows if r['full'] and r['job_id'] in field_ids];assert len(full)==3
    missing=[r for r in screen['rows'] if r['job_id'] in field_ids and r['job_id'] not in {v['job_id'] for v in full}];assert len(missing)==5
    for f in full:assert T.sha(f['path'])==f['sha256']
    p.update(id='T654_MATCHED_VERIFIED_ACTUAL_DEPARTURE_RECOVERY',out=str(OUT),authorization_approved=False,arms=['nominal','recovery'],arm='nominal',
        source_protocol_path=str(src/'protocol.json'),initial=dep['initial_research_model'],teacher_model=dep['teacher'],safe_retained=dep['teacher'],
        protected_tasks=[v['id'] for v in dep['fixed_3_first_departures']],recovery_jobs=[v['job'] for v in dep['fixed_3_first_departures']],
        required_field_ids=field_ids,existing_field_full_rows=full,field_reconstruct_jobs=[v['request']['job'] for v in missing],field_reconstruct_references=missing,
        fit_steps=32,fit_lr=.00003,keep_weight=10.,new_keep_weight=10.,Actor_each_cap=64,
        total_max_physics=26000000,total_max_commands=950,total_main_seconds=6300,
        shared_main_seconds=900,shared_audit_seconds=300,main_seconds=2700,audit_seconds=450,learning_seconds=300,full_evaluation_reserve_seconds=1700,
        max_physics=12100000,max_commands=433,max_bytes=384*2**20,shared_bytes=64*2**20,child_bytes=128*2**20,
        new_Actor_Adam=64,shadow_Actor_Adam=64,independent_historical_Actor_Adam=64,new_RL=0,new_Critic=0,preflight_optimizer=0,new_teacher=3)
    for k in ['registration_epoch','_root_sha256']:p.pop(k,None)
    p['shared_physics']=2*sum(j['case']['step_cap'] for j in p['recovery_jobs']+p['field_reconstruct_jobs'])
    worst=p['shared_physics']+2*sum(j['case']['step_cap'] for j in p['evaluation_jobs']+p['key_jobs']);assert worst<=p['total_max_physics']
    assert 22+2*433<=p['total_max_commands']
    storage=8*p['fresh_bytes']+8*p['compact_bytes']+p['full_bytes']+8*2**20+2*(20*p['fresh_bytes']+p['model_bytes']+407*p['compact_bytes']+p['full_bytes']+20*2**20)+8*2**20
    assert storage<=p['max_bytes'] and R.used(ROOT/'stage/runs')+p['max_bytes']<p['global_bytes']
    R.B.setup(p);q=R.L.frozen(p,p['initial']['path']);fixed=R.L.fixed_hash(q)
    ff=[]
    for row in full:
        a=R.E.N.S.arrays(row['path']);T.Native.inspect(p,a,row);ff.append(M.features(q,a,row,target=torch.tensor(a['locations'])))
    by={g['id']:g for g in old}
    nominal=[dict(by[key],first=0,target=by[key]['target'].clamp(-1,1).float().double()) for key in p['protected_tasks']]
    # Existing complete nominal histories are numeric fixtures, not qualified
    # recovery labels. No teacher continuation exists until approved launch.
    first=dep['fixed_3_first_departures'][0]['departure']['decision']
    f=full[0];a=R.E.N.S.arrays(f['path']);fixture=M.features(q,a,f,first=min(first,len(a['states'])-1))
    R.gpu_guard();T.os.environ['CUDA_VISIBLE_DEVICES']='0';T.os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    obj=M.make(p);q,opt=obj;before=M.state(obj);beforehash=M.digest(before)
    keep=M.pack(old,'cuda');fields=M.pack(ff,'cuda');repair=[{k:v.cuda() if torch.is_tensor(v) else v for k,v in g.items()} for g in nominal]
    loss=p['keep_weight']*M.keep_loss(q,keep)+p['new_keep_weight']*M.keep_loss(q,fields)+M.repair_loss(q,repair);loss.backward()
    variables=[v for v in q.parameters() if v.requires_grad];gradnorm=float(sum(v.grad.square().sum() for v in variables).sqrt());assert gradnorm>0 and all(torch.isfinite(v.grad).all() for v in variables)
    assert M.digest(M.state(obj))==beforehash and not opt.state
    # Nonzero synthetic parameter fixture proves cache equivalence for an
    # entire real causal sequence after changing adapters, without Adam.step.
    with torch.no_grad():
        q.steer_adapter[-1].bias.add_(.01)
        x=torch.tensor(a['states'],device='cuda');cached={k:v.cuda() if torch.is_tensor(v) else v for k,v in fixture.items()}
        difference=float((q.sequence(x)-M.location(q,cached)).abs().max());assert difference<1e-10
        for v in variables:opt.state[v]=dict(step=torch.tensor(5.),exp_avg=torch.full_like(v,.03125),exp_avg_sq=torch.full_like(v,.0625))
        mature=M.state(obj);M.load(obj,before);assert M.digest(M.state(obj))==beforehash and not opt.state and R.L.fixed_hash(q)==fixed
    q.cpu();del obj,q,keep,fields,repair;torch.cuda.empty_cache()
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    files=set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PRIMARY_REVIEW.md',DOC/'ACTUAL_DEPARTURES.json',src/'FINAL_REPORT_FINAL.json',src/'mean/audit.json',src/'paired/audit.json'}
    files|={Path(v['path']) for v in full}
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):files.add(Path(f).resolve())
    p['inputs'].update({str(f):T.sha(f) for f in files})
    for f,h in p['inputs'].items():assert T.sha(f)==h,f
    T.save(HERE/'protocol.json',p)
    T.save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',existing384_full_histories_checked=True,three_existing_field_full_histories_checked=True,
        fixed3_real_departure_anchors=True,recovery_teacher_labels_not_yet_collected=True,actual_GPU_gradient_norm=gradnorm,
        whole_real_sequence_cache_equivalence_after_synthetic_nonzero_adapter_change=difference,mature_synthetic_Adam_full_rollback_exact=True,
        full_frozen_body_unchanged=True,native_worst=worst,commands_worst=888,storage_worst=storage,global_worst=R.used(ROOT/'stage/runs')+p['max_bytes'],new_native=0,new_optimizer=0))
    T.save(DOC/'BUDGET.json',{k:p[k] for k in ['total_max_physics','total_max_commands','total_main_seconds','Actor_each_cap','max_bytes','global_bytes','shared_physics','shared_main_seconds','shared_audit_seconds','main_seconds','audit_seconds','learning_seconds','backup_seconds','source_seconds','absolute_learning_stop','absolute_native_stop','absolute_all_stop']})
    print('T654_ZERO_NATIVE_ZERO_OPTIMIZER_PASS',worst,storage,gradnorm,flush=True)
if __name__=='__main__':main()
