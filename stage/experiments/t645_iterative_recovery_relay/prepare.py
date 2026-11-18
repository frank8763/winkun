"""Real two-task recovery gradients and finite three-round budget preflight."""
from pathlib import Path
import ast,copy,json,sys,subprocess
import t645_runtime as T
R,M=T.R,T.M;ROOT,HERE,DOC,OUT=T.ROOT,T.HERE,T.DOC,T.OUT


def main():
    assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True)
    p=R.read(ROOT/'stage/experiments/t641_verified_recovery_learning/protocol.json')
    for key in ['registration_epoch','absolute_native_stop','absolute_learning_stop','absolute_all_stop','user_authorization']:p.pop(key,None)
    newgoal=p['recovery_row'];oldroot=ROOT/'stage/runs/t644_sampled_recovery';old=R.read(oldroot/'summary.json')['result']['first_qualified']
    assert old and R.read(oldroot/'audit.json')['state']=='PASS'
    initial=[dict(row=newgoal,first=5,origin='validated_T640_new_target'),dict(row=old,first=7,origin='validated_T644_old_task')]
    reference=[json.loads(z) for z in (ROOT/'stage/runs/t636_joint_best_reproduction/cases.jsonl').read_text().splitlines() if json.loads(z)['role']=='fresh']
    reference.append(newgoal);by={r['job_id']:r for r in reference};assert len(by)==7
    pre=[p['target_job']]+p['sentinel_jobs']
    priority=['wide117_80_dev6_LEFT_18',p['target_job']['id'],'wide117_8m_dev6_LEFT_13']+[j['id'] for j in pre if j['id'] not in ['wide117_80_dev6_LEFT_18',p['target_job']['id'],'wide117_8m_dev6_LEFT_13']]
    p.update(id='T645_ITERATIVE_VERIFIED_RECOVERY_RELAY',out=str(OUT),authorization_approved=False,
        max_rounds=3,initial_corrections=initial,reference_by_id=by,precheck_jobs=pre,failure_priority_ids=priority,
        second_input_audit=str(oldroot/'audit.json'),exploration_seed=20261118645,
        fit_steps=128,new_Actor_Adam=384,shadow_Actor_Adam=384,independent_historical_Actor_Adam=384,
        preflight_optimizer=0,new_RL=0,new_Critic=0,max_physics=13000000,max_commands=450,max_bytes=48*2**20,
        fresh_bytes=1280*1024,full_bytes=2**21,model_bytes=2**21,compact_bytes=12288,
        preload_seconds=900,input_audit_seconds=60,main_seconds=6000,audit_seconds=1200,
        learning_seconds=600,full_evaluation_reserve_seconds=2700,transport_seconds=300,source_seconds=180,all_seconds=10800,
        new_command_scope='independent finite relay allowance, not prior-window extension')
    # 3 seven-case screens +2*(4fresh+4exact)+400 remaining+7final exact +6 startup/close.
    assert 3*7+2*8+400+7+6==p['max_commands']
    worst=3*sum(j['case']['step_cap'] for j in pre)+2*8*53334+sum(j['case']['step_cap'] for j in p['evaluation_jobs'] if j['id'] not in by)+sum(j['case']['step_cap'] for j in pre)
    assert worst==12964814<=p['max_physics']
    assert p['preload_seconds']+p['input_audit_seconds']+p['main_seconds']+p['audit_seconds']+p['transport_seconds']+p['source_seconds']+120<p['all_seconds']
    assert len(str(OUT/'raw.sock').encode())<108
    # Full-successful correction labels are checked against actual executed
    # actions. The old stochastic mean was not a successful deterministic law.
    R.B.setup(p);q=R.L.frozen(p,p['initial']['path']);body=R.digest(q.state_dict())
    cpu=[T.correction(p,item) for item in initial]
    rows=[json.loads(z) for z in (ROOT/'stage/runs/t636_joint_best_reproduction/cases.jsonl').read_text().splitlines() if json.loads(z)['role']=='fresh']
    originals={r['job_id']:r for r in R.D.selected()};rows+=[r for r in originals.values() if Path(r['path']).exists()]
    data=[R.D.features(q,R.E.N.S.arrays(r['path']),originals[r['job_id']]) for r in rows];assert len(data)==11
    R.gpu_guard();T.os.environ['CUDA_VISIBLE_DEVICES']='0';T.os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    q.cuda();q.drive_adapter.requires_grad_(True);q.steer_adapter.requires_grad_(True)
    keep=M.pack(data,'cuda');new=[{k:v.cuda() if T.torch.is_tensor(v) else v for k,v in g.items()} for g in cpu]
    loss,repair,protect=M.objective(q,keep,new,p);loss.backward()
    gradients=[v.grad.detach().clone() for v in R.L.variables(q)]
    assert all(T.torch.isfinite(g).all() for g in gradients) and float(protect.detach())<1e-18 and R.digest(q.state_dict())==body
    # Duplicating one task's trajectory must not double its supervision weight.
    with T.torch.no_grad():
        duplicate=M.objective(q,keep,new+[new[0]],p)[0];permuted=M.objective(q,keep,list(reversed(new)),p)[0]
        assert abs(float(duplicate-loss.detach()))<1e-12 and abs(float(permuted-loss.detach()))<1e-12
    norm=float(sum(g.square().sum() for g in gradients).sqrt());assert norm>0
    q.cpu();q.zero_grad(set_to_none=True)
    # Original schema measured from actual full histories: longest fresh/pre/key
    # case has <=534 decisions, unlike generic possible partials of field24.
    sample=R.E.N.S.arrays(rows[0]['path']);n=len(sample['states'])
    uncompressed=sum(v.dtype.itemsize*(534*int(T.np.prod(v.shape[1:])) if v.ndim and v.shape[0]==n else v.size) for v in sample.values())+534*2*4+131072
    assert uncompressed<=p['fresh_bytes']
    upper=18*p['fresh_bytes']+4*p['full_bytes']+3*p['model_bytes']+429*p['compact_bytes']+6*2**20
    assert upper<=p['max_bytes'],upper
    global_worst=R.used(ROOT/'stage/runs')+p['max_bytes']+131072;assert global_worst<2**30
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    files=set(HERE.glob('*.py'))|{DOC/'PLAN.md',oldroot/'summary.json',oldroot/'audit.json',oldroot/'SEALED.json',Path(old['path']),Path(newgoal['path'])}
    files|={Path(r['path']) for r in reference}
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):files.add(Path(f).resolve())
    p['inputs'].update({str(f):R.sha(f) for f in files})
    R.save(HERE/'protocol.json',p)
    R.save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',actual_old_histories=11,
      actual_recovery_tasks=2,actual_success_source_SHAs=[v['row']['sha256'] for v in initial],masked_prefixes=[5,7],
      actual_GPU_joint_gradient_norm=norm,initial_repair_loss=float(repair.detach()),initial_keep_loss=float(protect.detach()),
      task_weight_duplicate_and_permutation_invariant=True,original_T501_unchanged=True,new_optimizer=0,
      full384_RAM_gate_required=True,per_fresh_uncompressed_upper=uncompressed,own_storage_upper=upper,global_worst=global_worst,
      native_worst=worst,commands_worst=450,maximum_live_shadow_history_each=384,no_identical_data_repeat=True))
    R.save(DOC/'BUDGET.json',{k:p[k] for k in ['max_rounds','max_physics','max_commands','max_bytes','preload_seconds','main_seconds','audit_seconds','all_seconds','new_Actor_Adam','shadow_Actor_Adam','independent_historical_Actor_Adam']})
    result=subprocess.check_output([str(ROOT/'run_stage.sh'),str(HERE/'control.py'),'resources'],text=True);R.save(DOC/'RESOURCE_PREFLIGHT.json',json.loads(result))
    print('T645_PREFLIGHT_PASS_ZERO_NATIVE_ZERO_OPTIMIZER',norm,upper,global_worst,worst,flush=True)


if __name__=='__main__':main()
