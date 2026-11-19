"""True-history finite response and full Actor/Adam rollback preflight; no steps."""
from pathlib import Path
import ast,copy,io,os,sys,time
import t648_runtime as T
R,M,torch,np=T.R,T.M,T.torch,T.np
ROOT,HERE,DOC,OUT=T.ROOT,T.HERE,T.DOC,T.OUT

def main():
    assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True)
    p=T.read(ROOT/'stage/experiments/t647_joint_success_coverage/protocol.json')
    for k in ['registration_epoch','absolute_learning_stop','absolute_native_stop','absolute_all_stop','user_authorization','_root_sha256']:p.pop(k,None)
    dense=p['dense_rows'];tasks=p['new_tasks'];train=[];validation=[]
    for key in tasks:
        group=[r for r in dense if r['id']==key]
        for index,row in enumerate(group):(validation if index%4==3 else train).append(row)
    assert len(train)+len(validation)==198 and set(r['sha256'] for r in train).isdisjoint(r['sha256'] for r in validation)
    p.update(id='T648_FINITE_RESPONSE_CONSTRAINED_REPLAY',authorization_approved=False,out=str(OUT),arms=['replay','constrained'],arm='replay',
        training_rows=train,validation_rows=validation,training_trajectory_count=len(train),validation_trajectory_count=len(validation),
        response_peak=.01,response_rms=.002,response_batch=16,commit_block=8,candidate_scales=[1.,.25,.0625,.015625],
        partial_Adam_state='full proposed block moments; actor is finite scaled displacement; reject restores full before moments',
        max_bytes=240*2**20,child_bytes=112*2**20,learning_seconds=1200,main_seconds=3300,audit_seconds=1200,
        total_main_seconds=6600,all_seconds=12600,backup_seconds=1200,source_seconds=300,old_backup_seconds=600,
        Actor_each_cap=384,max_response_checks_each_arm=96,extra_response_native=0)
    p.pop('sparse_rows',None);p.pop('dense_rows',None)
    superseded=str(ROOT/'stage/docs/day_20261119/t647/PLAN.md')
    p['superseded_unlaunched_T647_plan_lock']=p['inputs'].pop(superseded,None)
    calibration=T.read(DOC/'RESPONSE_CALIBRATION.json');assert calibration['state']=='PASS_ZERO_NATIVE_ZERO_OPTIMIZER'
    assert T.read(DOC/'RESPONSE_LIMITS.json')['response_peak']==p['response_peak']
    for r in train+validation:assert T.sha(r['path'])==r['sha256']
    T.Shared.O.input_check(p);R.B.setup(p)
    q=R.L.frozen(p,p['initial']['path']);hash0=q.phash()
    manifest=T.read(T.Shared.OUT/'OFFLINE_INPUTS.json');by={r['job_id']:r for r in R.D.selected()};old=[]
    for row in manifest['members'][:8]:
        a=R.E.N.S.arrays(row['offline_path']);g=R.D.features(q,a,by[row['id']]);g['x']=torch.tensor(a['states']);old.append(g)
    new=T.load_new(p,lambda:None);heldout=T.load_heldout(p,lambda:None)
    assert len(new)==len(train) and len(heldout)==len(validation)
    a,b=M.schedule(old,new,0,p);assert len(b)==8
    R.gpu_guard();os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    obj=M.make(dict(p,arm='full_command'));model,opt=obj;initial=M.digest(M.state(obj));started=time.time()
    loss,repair,keep=M.Base.objective(model,a,b,dict(p,arm='full_command'));loss.backward();torch.cuda.synchronize()
    grads={k:float(v.grad.norm()) for k,v in model.named_parameters() if v.grad is not None}
    assert all(np.isfinite(v) for v in grads.values()) and any(k.startswith('core.') and v>0 for k,v in grads.items())
    assert model.phash()==hash0 and not opt.state and initial==M.digest(M.state(obj))
    with torch.no_grad():
        before=M.state(obj)
        # Synthetic optimizer moment fixture: this is a serialization/rollback
        # test, NOT an optimizer call, fitted state or scientific candidate.
        parameters=list(model.parameters())
        for param in parameters:
            opt.state[param]=dict(step=torch.tensor(7.),exp_avg=torch.full_like(param,.03125),exp_avg_sq=torch.full_like(param,.0625))
        mature_before=M.state(obj);mature_hash=M.digest(mature_before)
        model.head.bias.add_(.3)
        for st in opt.state.values():st['step'].add_(1);st['exp_avg'].add_(.125)
        proposed=M.state(obj)
        fixture_reports=[]
        for factor in p['candidate_scales']:
            M.commit_scaled(obj,mature_before,proposed,factor)
            expected={k:mature_before['actor'][k]+factor*(v-mature_before['actor'][k]) if v.is_floating_point() else v for k,v in proposed['actor'].items()}
            assert all(torch.equal(v,model.state_dict()[k]) for k,v in expected.items())
            assert M.digest(opt.state_dict())==M.digest(proposed['actor_opt'])
            response=M.responses(model,old,p,lambda:None)
            fixture_reports.append(dict(scale=factor,metrics=M.metrics_summary(response,p)))
        assert not fixture_reports[0]['metrics']['passed']
        M.commit_scaled(obj,mature_before,proposed,0.)
        assert M.digest(M.state(obj))==mature_hash
        M.load(obj,before);assert M.digest(M.state(obj))==initial and not opt.state
        clean=M.responses(model,old,p,lambda:None);assert M.response_gate(clean,p)
        # Boundary: one bad task must fail even if global average could hide it.
        positive=[dict(id='good',peak_command_error=.001,RMS_command_error=.0001)]*383
        bad=positive+[dict(id='bad',peak_command_error=.011,RMS_command_error=.001)]
        assert not M.response_gate(bad,p) and M.response_gate(positive+[dict(id='at_limit',peak_command_error=.01,RMS_command_error=.002)],p)
        batch=M.Base.evaluate_sequences(model,b,'cuda')
        assert all(torch.allclose(v,model.sequence(g['x'].cuda()),atol=1e-12,rtol=1e-12) for v,g in zip(batch,b))
        # Heldout validation is read-only and task equal, not an acceptance gate.
        val=M.validation(model,heldout,p,lambda:None);assert val['heldout_trajectories']==len(validation)
    assert initial==M.digest(M.state(obj)) and model.phash()==hash0
    dummy=[dict(id=str(i)) for i in range(384)];seen=set();oldcounts={str(i):0 for i in range(384)}
    for k in range(192):
        aa,bb=M.schedule(dummy,new,k,p)
        assert sorted(g['id'] for g in bb)==tasks
        for g in aa:oldcounts[g['id']]+=1
        seen.update(g['source_sha256'] for g in bb)
    assert seen=={r['sha256'] for r in train} and set(oldcounts.values())=={4}
    model.cpu();del obj,model;torch.cuda.empty_cache()
    worst=sum(j['case']['step_cap'] for j in p['evaluation_jobs']+p['key_jobs']);assert worst<=p['max_physics']
    storage=2*(p['model_bytes']+14*p['fresh_bytes']+407*p['compact_bytes']+p['full_bytes']+20*2**20)+4*2**20
    assert storage<=p['max_bytes'] and storage/2<p['child_bytes']
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    files=set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'RESPONSE_LIMITS.json',DOC/'RESPONSE_CALIBRATION.json',DOC/'LITERATURE.md'}
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):files.add(Path(f).resolve())
    p['inputs'].update({str(f):T.sha(f) for f in files})
    global_worst=R.used(ROOT/'stage/runs')+p['max_bytes'];assert global_worst<10*2**30
    T.save(HERE/'protocol.json',p)
    T.save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',all198_actual_histories_checked=True,training_trajectories=len(train),validation_trajectories=len(validation),heldout_sources_disjoint=True,
       actual_GPU_gradient_norms=grads,all_scales_actual_parameter_formula_and_proposed_moments_checked=True,mature_Adam_fixture_reject_restores_all=True,
       isolated_synthetic_moments_not_training=True,fixture_reports=fixture_reports,one_bad_of384_task_rejected=True,causal_batch_individual_equal=True,
       original_Actor_Adam_unchanged=True,all_training_sources_scheduled=True,old384_four_complete_passes=True,
       physics_worst_each=worst,commands_worst_each=427,storage_worst=storage,global_worst=global_worst,
       max_response_checks_each_fit=96,full384_forward_measured_seconds=calibration['models'][0]['seconds'],preflight_seconds=time.time()-started))
    T.save(DOC/'BUDGET.json',{k:p[k] for k in ['total_max_physics','total_max_commands','total_main_seconds','Actor_each_cap','max_bytes','global_bytes','learning_seconds','main_seconds','audit_seconds','all_seconds','old_backup_seconds','backup_seconds','source_seconds','max_response_checks_each_arm']})
    print('T648_ZERO_NATIVE_ZERO_OPTIMIZER_PASS',len(train),len(validation),worst,storage,flush=True)

if __name__=='__main__':main()
