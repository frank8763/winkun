"""Risk-gradient checks on actual trajectories; no optimizer or simulation."""
from pathlib import Path
import ast,json,sys,time,subprocess
import t642_runtime as T
R=T.R;ROOT,HERE,DOC,OUT=T.ROOT,T.HERE,T.DOC,T.OUT


def main():
    assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True)
    p=R.read(ROOT/'stage/experiments/t641_verified_recovery_learning/protocol.json')
    for key in ['registration_epoch','absolute_native_stop','absolute_learning_stop','absolute_all_stop','user_authorization']:p.pop(key,None)
    previous=R.read(ROOT/'stage/runs/t641_verified_recovery_learning/summary.json')
    assert previous['state']=='COMPLETE' and not previous['result']['accepted'] and R.read(ROOT/'stage/runs/t641_verified_recovery_learning/audit.json')['state']=='PASS'
    p.update(id='T642_TAIL_RISK_FULL384_RETENTION',out=str(OUT),authorization_approved=False,
        critical_trace_id='wide117_80_dev6_LEFT_18',retention_aggregation='half384taskmean_plus_half_worst10pct_tasks_of_halfmean_halfworst10pct_time',
        prior_commands=p['prior_commands']+previous['commands'],prior_physics=p['prior_physics']+previous['physics'],
        prior_main_seconds=p['prior_main_seconds']+previous['elapsed_s'])
    p['combined_command_cap']=p['prior_commands']+p['max_commands']
    p['combined_physics_cap']=p['prior_physics']+p['max_physics']
    p['combined_Actor_each_cap']=256
    # Main wall uses remaining prior3600 allowance; no need increase it.
    p['main_seconds']=int(3600-p['prior_main_seconds'])
    assert p['main_seconds']>3000 and p['fit_steps']==128
    q=R.L.frozen(p,p['initial']['path']);before=R.digest(q.state_dict())
    repro=ROOT/'stage/runs/t636_joint_best_reproduction'
    rr=[json.loads(z) for z in (repro/'cases.jsonl').read_text().splitlines() if json.loads(z)['role']=='fresh']
    original={r['job_id']:r for r in R.D.selected()};rr+=[r for r in original.values() if Path(r['path']).exists()]
    data=[R.D.features(q,R.E.N.S.arrays(r['path']),original[r['job_id']]) for r in rr]
    assert len(data)==11
    correction=R.M.correction(p,R.E.N.S.arrays(p['recovery_row']['path']),5)
    R.gpu_guard();T.Old.os.environ['CUDA_VISIBLE_DEVICES']='0';T.Old.os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    q.cuda();q.drive_adapter.requires_grad_(True);q.steer_adapter.requires_grad_(True)
    keep=R.M.pack(data,'cuda');new={k:v.cuda() for k,v in correction.items()}
    loss,repair,protection=R.M.objective(q,keep,new,p);loss.backward()
    assert all(v.grad is not None and R.torch.isfinite(v.grad).all() for v in R.L.variables(q))
    assert float(protection.detach())<1e-18 and R.digest(q.state_dict())==before
    gradient=float(sum(v.grad.square().sum() for v in R.L.variables(q)).sqrt())
    # The same failure-bearing historical candidate supplies a real nonzero
    # old-policy deviation. Verify the new risk doesn't collapse to taskmean.
    candidate=R.L.frozen(p,previous['result']['proposal']['path']).cuda()
    with R.torch.no_grad():
        err=((R.M.location(candidate,keep)-keep['target'])/.2).square().mean(-1)
        mean=float((err*keep['weights']).sum());tail=float(R.M.retention(err,keep['lengths']))
    assert tail>mean>0
    e=err.detach().clone().requires_grad_(True);v=R.M.retention(e,keep['lengths']);v.backward()
    assert R.torch.isfinite(e.grad).all() and float(e.grad.sum())>0
    q.cpu();candidate.cpu()
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    files=set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'T641_SAME_STATE_ERRORS.json',
        ROOT/'stage/runs/t641_verified_recovery_learning/summary.json',ROOT/'stage/runs/t641_verified_recovery_learning/audit.json',
        Path(previous['result']['proposal']['path'])}
    for module in list(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):files.add(Path(f).resolve())
    p['inputs'].update({str(f):R.sha(f) for f in files})
    storage=R.used(ROOT/'stage/runs')+p['max_bytes']+131072;assert storage<2**30
    prior_bytes=sum(R.used(ROOT/'stage/runs'/name) for name in ['t638_memory_ready_recovery','t639_neural_recovery','t640_first_divergence_recovery','t641_verified_recovery_learning'])
    assert prior_bytes+p['max_bytes']<=52*2**20
    R.save(HERE/'protocol.json',p)
    R.save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',actual_histories=11,
         real_successful_recovery_source=p['recovery_row']['sha256'],GPU_gradient_norm=gradient,
         initial_keep_loss=float(protection.detach()),T641_actual_candidate_meanrisk=mean,T641_actual_candidate_tailrisk=tail,
         tail_gradient_finite_nonzero=True,original_T501_unchanged=True,optimizer_steps=0,
         old_critical_full_trace_enabled=True,full384_RAM_admission_required=True,
         combined_physics_cap=p['combined_physics_cap'],combined_command_cap=p['combined_command_cap'],
         combined_main_seconds_cap=p['prior_main_seconds']+p['main_seconds'],combined_records_reserved=prior_bytes+p['max_bytes'],global_worst=storage))
    R.save(DOC/'BUDGET.json',{k:p[k] for k in ['max_physics','max_commands','max_bytes','main_seconds','all_seconds','combined_physics_cap','combined_command_cap','combined_Actor_each_cap','new_Actor_Adam','shadow_Actor_Adam','independent_historical_Actor_Adam']})
    result=subprocess.check_output([str(ROOT/'run_stage.sh'),str(HERE/'control.py'),'resources'],text=True);R.save(DOC/'RESOURCE_PREFLIGHT.json',json.loads(result))
    print('T642_ZERO_NATIVE_ZERO_OPTIMIZER_PREFLIGHT_PASS',mean,tail,p['combined_physics_cap'],p['combined_command_cap'])


if __name__=='__main__':main()
