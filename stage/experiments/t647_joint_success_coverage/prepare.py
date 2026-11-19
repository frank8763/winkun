"""Lock original-start successful trajectory coverage, no native/optimizer steps."""
from pathlib import Path
import ast,copy,gzip,io,os,sys,time
import t647_runtime as T
R,M,torch,np=T.R,T.M,T.torch,T.np
ROOT,HERE,DOC,OUT=T.ROOT,T.HERE,T.DOC,T.OUT

def main():
    assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True)
    p=T.read(ROOT/'stage/experiments/t646_offline_night/protocol.json')
    for k in ['registration_epoch','absolute_learning_stop','absolute_native_stop','absolute_all_stop','user_authorization','_root_sha256']:p.pop(k,None)
    source=ROOT/'stage/runs/t646_offline_night/coverage/summary.json';s=T.read(source)
    assert s['state']=='COMPLETE' and T.read(source.parent/'audit.json')['state']=='PASS'
    rows=s['new_success_rows'];assert len(rows)==198
    tasks=sorted({r['job_id'] for r in rows});assert len(tasks)==8
    # Preserved collection order, not fastest/most favourable outcome selection.
    dense=[dict(id=r['job_id'],path=r['path'],sha256=r['sha256'],seed=r['seed']) for r in rows]
    sparse=[next(r for r in dense if r['id']==key) for key in tasks]
    jobs=p['evaluation_jobs'];field=[j for j in jobs if j['id'].startswith('t502_')];assert len(field)==24
    screen=p['sentinel_jobs']+field;assert len(screen)==30 and len({j['id'] for j in screen})==30
    keys=p['sentinel_jobs']+[next(j for j in field if j['id']==key) for key in tasks]
    old_field=[j['id'] for j in field if j['id'] in p['before_success_ids']];assert len(old_field)==5
    worst=sum(j['case']['step_cap'] for j in jobs+keys)
    p.update(id='T647_SUCCESS_COVERAGE_ABLATION',authorization_approved=False,out=str(OUT),arms=['sparse','dense'],arm='dense',
        sparse_rows=sparse,dense_rows=dense,coverage_summary=str(source),new_tasks=tasks,screen_jobs=screen,field_jobs=field,key_jobs=keys,
        old_screen_ids=[j['id'] for j in p['sentinel_jobs']]+old_field,fit_steps=192,old_batch=8,fit_lr=.0001,keep_weight=10.,
        max_physics=12000000,max_commands=427,max_bytes=200*2**20,child_bytes=96*2**20,global_bytes=10*2**30,
        total_max_physics=24000000,total_max_commands=860,total_main_seconds=6000,Actor_each_cap=384,
        new_Critic=0,new_RL=0,preflight_optimizer=0,learning_seconds=900,main_seconds=3000,audit_seconds=900,
        full_evaluation_reserve_seconds=1800,backup_seconds=1200,source_seconds=300,old_backup_seconds=900,
        all_seconds=10800,model_bytes=16*2**20,fresh_bytes=3*2**20,full_bytes=4*2**20,compact_bytes=32768)
    assert worst<=p['max_physics'] and (407+14+6)==427 and 2*worst<=p['total_max_physics']
    storage=2*(p['model_bytes']+14*p['fresh_bytes']+407*p['compact_bytes']+p['full_bytes']+10*2**20)+4*2**20
    assert storage<=p['max_bytes']
    for r in dense:assert T.sha(r['path'])==r['sha256']
    # Validate every source's sampler provenance without any native stepping.
    T.Shared.O.input_check(p)
    R.B.setup(p);q=R.L.frozen(p,p['initial']['path']);original_hash=q.phash()
    manifest=T.read(T.Shared.OUT/'OFFLINE_INPUTS.json');by={r['job_id']:r for r in R.D.selected()};old=[]
    for row in manifest['members'][:8]:
        a=R.E.N.S.arrays(row['offline_path']);g=R.D.features(q,a,by[row['id']]);g['x']=torch.tensor(a['states']);old.append(g)
    audited=T.load_new(dict(p,arm='dense'),lambda:None);assert len(audited)==198
    del audited
    first=[];second=[]
    for key in tasks:
        rr=[r for r in dense if r['id']==key]
        for selected,target in [(rr[0],first),(rr[1],second)]:
            a=R.E.N.S.arrays(selected['path']);full=next(r for r in rows if r['sha256']==selected['sha256']);T.Coverage.inspect(p,a,full)
            target.append(dict(id=key,x=torch.tensor(a['states']),target=torch.tensor(a['actions']).double(),first=0,source_sha256=selected['sha256']))
    R.gpu_guard();os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    obj=M.make(dict(p,arm='full_command'));model,opt=obj;before=M.digest(M.state(obj));started=time.time()
    loss,repair,keep=M.objective(model,old,first,dict(p,arm='full_command'));loss.backward();torch.cuda.synchronize()
    grads={k:float(v.grad.norm()) for k,v in model.named_parameters() if v.grad is not None}
    assert all(np.isfinite(v) for v in grads.values()) and any(k.startswith('encoder.') and v>0 for k,v in grads.items())
    assert any(k.startswith('core.') and v>0 for k,v in grads.items()) and abs(float(keep.detach()))<1e-16
    assert model.phash()==original_hash and not opt.state and M.digest(M.state(obj))==before
    with torch.no_grad():
        permuted=M.objective(model,old,list(reversed(first)),dict(p,arm='full_command'))[0]
        assert abs(float(permuted-loss.detach()))<1e-12
    dummy=[dict(id=f'old{i}') for i in range(384)]
    # The actual schedule must expose every one of198 success sources, each
    # task exactly once per step, old trajectories exactly four complete passes.
    refs=[dict(id=r['id'],source_sha256=r['sha256']) for r in dense];seen=set();count={r['id']:0 for r in dense}
    for k in range(p['fit_steps']):
        a,b=M.schedule(dummy,refs,k,p);assert len(a)==8 and sorted(g['id'] for g in b)==tasks
        for g in b:seen.add(g['source_sha256']);count[g['id']]+=1
    assert seen=={r['sha256'] for r in dense} and set(count.values())=={192}
    repeat=[dict(id=r['id'],source_sha256=r['sha256']) for r in sparse]
    assert all(M.schedule(dummy,repeat,k,p)[1]==repeat for k in range(192))
    assert p['fit_steps']*8//384==4
    model.cpu();del obj,model;torch.cuda.empty_cache()
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    files=set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PRIMARY_METHOD_REVIEW.md',DOC/'COVERAGE_ANALYSIS.json',source,source.parent/'audit.json',source.parent/'cases.jsonl.gz'}
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):files.add(Path(f).resolve())
    p['inputs'].update({str(f):T.sha(f) for f in files});p['inputs'].update({r['path']:r['sha256'] for r in dense})
    global_worst=R.used(ROOT/'stage/runs')+p['max_bytes'];assert global_worst<10*2**30
    T.save(HERE/'protocol.json',p)
    T.save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',actual_success_tasks=8,source_trajectories=198,
        sparse_task_order=tasks,all198_actual_history_audits=True,all198_source_SHAs=True,all198_exposed_by_schedule=True,
        task_equal_weight_and_permutation=True,old384_four_passes=True,initial_Actor_Adam_unchanged=True,
        actual_GPU_forward_backward_s=time.time()-started,gradient_norms=grads,
        physics_worst_each=worst,commands_worst_each=427,storage_worst=storage,global_worst=global_worst,no_nav_claim=True))
    T.save(DOC/'BUDGET.json',{k:p[k] for k in ['total_max_physics','total_max_commands','total_main_seconds','Actor_each_cap','max_bytes','global_bytes','main_seconds','audit_seconds','old_backup_seconds','backup_seconds','source_seconds','all_seconds']})
    print('T647_ZERO_NATIVE_ZERO_OPTIMIZER_PASS',worst,storage)

if __name__=='__main__':main()
