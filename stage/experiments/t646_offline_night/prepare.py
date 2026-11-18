"""Zero native/optimizer preflight on actual original histories and recoveries."""
from pathlib import Path
import ast,copy,io,json,os,sys,time
import t646_runtime as T
R,M,O=T.R,T.M,T.O
ROOT,HERE,OUT,DOC=T.ROOT,T.HERE,T.OUT,T.DOC

def main():
    DOC.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    p=T.read(ROOT/'stage/experiments/t645_iterative_recovery_relay/protocol.json')
    for k in ['registration_epoch','absolute_all_stop','absolute_native_stop','absolute_learning_stop','user_authorization']:p.pop(k,None)
    p.update(id='T646_OFFLINE_MATCHED_NIGHT',out=str(OUT),authorization_approved=False,
       arms=['heads_command','full_command','full_raw'],arm='full_command',max_rounds=4,fit_steps=192,old_batch=8,fit_lr=.0001,
       night_max_physics=128000000,night_max_commands=3000,night_Actor_each_cap=2304,night_main_seconds=36000,
       new_Actor_Adam=768,shadow_Actor_Adam=768,independent_historical_Actor_Adam=768,preflight_optimizer=0,new_RL=0,new_Critic=0,
       max_physics=14000000,max_commands=465,max_bytes=3*2**30,child_bytes=128*2**20,model_bytes=16*2**20,
       main_seconds=7200,audit_seconds=2400,learning_seconds=1200,input_audit_seconds=60,full_evaluation_reserve_seconds=2700,
       profile_seconds=1800,profile_audit_seconds=900,backup_seconds=1200,source_seconds=300,finalization_reserve_seconds=1620,
       exploration_seed=20261118646,global_bytes=10*2**30,free_bytes=40*2**30)
    by=p['reference_by_id'];pre=p['precheck_jobs']
    worst=4*sum(j['case']['step_cap'] for j in pre)+3*8*53334+sum(j['case']['step_cap'] for j in p['evaluation_jobs'] if j['id'] not in by)+sum(j['case']['step_cap'] for j in pre)
    assert worst<=p['max_physics'] and 4*7+3*8+400+7+6==465
    deltas=[(.01,0,0),(-.01,0,0),(0,.01,0),(0,-.01,0),(0,0,.5),(0,0,-.5),(.01,.01,.5),(-.01,-.01,-.5)]
    jobs=[]
    for base in p['sentinel_jobs']:
        for i,(dx,dy,da) in enumerate(deltas):
            job=copy.deepcopy(base);job['id']=f"profile_{base['id']}_{i}"
            for key,d in [('delta_x_m',dx),('delta_y_m',dy),('delta_yaw_deg',da)]:job[key]=job.get(key,0)+d
            job.pop('expected_initial120_sha256',None);jobs.append(job)
    assert len(jobs)==48
    p['coverage_jobs']=[j for j in p['evaluation_jobs'] if j['id'].startswith('t502_') and j['id'] not in p['before_success_ids']]
    assert len(p['coverage_jobs'])==19
    p['coverage_max_physics']=64*sum(j['case']['step_cap'] for j in p['coverage_jobs'])
    p['coverage_seconds']=28800;p['coverage_audit_seconds']=2400
    p['profile_jobs']=jobs;profile_worst=2*sum(j['case']['step_cap'] for j in jobs)
    assert 3*worst+profile_worst+p['coverage_max_physics']<=p['night_max_physics'] and 3*465+102+1222<=p['night_max_commands']
    storage=3*(4*p['model_bytes']+34*p['fresh_bytes']+4*p['full_bytes']+435*p['compact_bytes']+8*2**20)+128*2**20+96*2**20+8*2**20
    storage+=608*(3*2**20+32768)+8*2**20
    assert storage<=p['max_bytes'],storage
    original=T.read(OUT/'OFFLINE_INPUTS.json');assert len(original['members'])==384
    assert all(T.sha(r['offline_path'])==r['sha256'] for r in original['members'])
    R.B.setup(p);q=R.L.frozen(p,p['initial']['path']);hash0=q.phash()
    selected={r['job_id']:r for r in R.D.selected()};data=[]
    for row in original['members'][:8]:
        a=R.E.N.S.arrays(row['offline_path']);g=R.D.features(q,a,selected[row['id']]);g['x']=T.torch.tensor(a['states']);data.append(g)
    new=[O.correction(p,item) for item in p['initial_corrections']]
    R.gpu_guard();os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    reports=[]
    for arm in p['arms']:
        pc=dict(p,arm=arm);obj=M.make(pc);model,opt=obj;before=M.digest(M.state(obj));started=time.time()
        loss,repair,keep=M.objective(model,data,new,pc);loss.backward();T.torch.cuda.synchronize()
        elapsed=time.time()-started
        grads={key:float(v.grad.norm()) for key,v in model.named_parameters() if v.grad is not None}
        assert grads and all(T.np.isfinite(v) for v in grads.values()) and abs(float(keep.detach()))<1e-16
        assert model.phash()==hash0 and not opt.state and M.digest(M.state(obj))==before
        if arm=='heads_command':assert all(k.startswith(('drive_adapter.','steer_adapter.')) for k in grads)
        else:assert any(k.startswith('encoder.') and v>0 for k,v in grads.items()) and any(k.startswith('core.') and v>0 for k,v in grads.items())
        with T.torch.no_grad():
            batch=M.evaluate_sequences(model,data[:2],'cuda')
            assert all(T.torch.allclose(v,model.sequence(g['x'].cuda()),atol=1e-12,rtol=1e-12) for v,g in zip(batch,data[:2]))
            duplicate=M.objective(model,data,new+[new[0]],pc)[0]
            assert abs(float(duplicate-loss.detach()))<1e-12
        # Actual state packing and initial exact model reload; no fake Adam updates.
        model.cpu();buf=io.BytesIO();T.torch.save(M.state(obj),buf);assert len(buf.getvalue())*4<pc['model_bytes']
        reports.append(dict(arm=arm,parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),gradient_norms=grads,
           forward_backward_seconds=elapsed,new_Adam=0,full_history_batch_matches_individual=True))
        del obj,model;T.torch.cuda.empty_cache()
    x=T.torch.tensor([[1.1,-1.1],[.8,-.8]],requires_grad=True);target=T.torch.tensor([[1.,-1.],[.7,-.7]])
    value=M.P.action_loss(x,target);assert T.torch.equal(value[0],T.torch.zeros(2)) and bool((value[1]>0).all())
    count=[0]*384
    for k in range(192):
        for i in range(8):count[(8*k+i)%384]+=1
    assert set(count)=={4}
    files=set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'LITERATURE.md',OUT/'OFFLINE_INPUTS.json'}
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):files.add(Path(f).resolve())
    # Keep original evidence input locks, then add all current scientific modules.
    p['inputs'].update({str(f):T.sha(f) for f in files})
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    global_worst=R.used(ROOT/'stage/runs')-R.used(OUT)+p['max_bytes'];assert global_worst<10*2**30
    T.save(HERE/'protocol.json',p)
    T.save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',arms=reports,all384_original_SHA=True,old_passes=4,
        causal_batch_verified=True,task_duplication_weight_invariant=True,saturation_feasible_set_checked=True,
        native_arm_worst=worst,profile_worst=profile_worst,total_worst=3*worst+profile_worst+p['coverage_max_physics'],commands_worst=2719,
        storage_worst=storage,global_worst=global_worst,no_navigation_claim=True))
    T.save(DOC/'BUDGET.json',{k:p[k] for k in ['night_max_physics','night_max_commands','night_Actor_each_cap','night_main_seconds','max_bytes','global_bytes','main_seconds','audit_seconds','profile_seconds','profile_audit_seconds','backup_seconds']})
    print('T646_PREFLIGHT_PASS',3*worst+profile_worst+p['coverage_max_physics'],storage,flush=True)

if __name__=='__main__':main()
