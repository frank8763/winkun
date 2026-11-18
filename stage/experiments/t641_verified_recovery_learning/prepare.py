"""Actual successful suffix + original histories, zero-optimizer GPU gradients."""
from pathlib import Path
import ast,copy,gzip,json,sys,time,subprocess
import t641_runtime as T
R=T.R;ROOT,HERE,DOC,OUT=T.ROOT,T.HERE,T.DOC,T.OUT


def main():
    assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True)
    old=ROOT/'stage/runs/t640_first_divergence_recovery';s=R.read(old/'summary.json');assert s['result']['qualified'] and R.read(old/'audit.json')['state']=='PASS'
    row=s['result']['recovery'];assert R.sha(row['path'])==row['sha256']
    p=R.read(ROOT/'stage/experiments/t638_memory_ready_recovery/protocol.json')
    for k in ['authorization_approved','user_authorization','registration_epoch','absolute_native_stop','absolute_learning_stop','absolute_all_stop']:p.pop(k,None)
    prior=[R.read(ROOT/'stage/runs'/name/'summary.json') for name in ['t638_memory_ready_recovery','t639_neural_recovery','t640_first_divergence_recovery']]
    p.update(id='T641_VALIDATED_EARLY_RECOVERY_AND_FULL384',out=str(OUT),recovery_row=row,teacher_first=5,
        teacher_protocol=str(ROOT/'stage/experiments/t640_first_divergence_recovery/protocol.json'),
        teacher_configuration=str(ROOT/'stage/experiments/t639_neural_recovery/teacher_configuration.json'),teacher_audit=str(old/'audit.json'),
        teacher_kind='VALIDATED_T608_ACTUAL_SOURCE',max_physics=11575404,max_commands=420,max_bytes=40*2**20,
        preload_seconds=900,input_audit_seconds=30,main_seconds=3300,audit_seconds=900,learning_seconds=600,
        full_evaluation_reserve_seconds=2700,transport_seconds=180,source_seconds=120,all_seconds=7200,
        combined_command_cap=448,prior_commands=sum(v['commands'] for v in prior),prior_physics=sum(v['physics'] for v in prior),
        prior_main_seconds=sum(v['elapsed_s'] for v in prior),authorization_approved=False)
    assert p['prior_commands']+p['max_commands']==448
    assert p['prior_physics']+p['max_physics']<=11982225
    assert p['prior_main_seconds']+p['main_seconds']<=3600
    exact=sum(j['case']['step_cap'] for j in p['evaluation_jobs'] if j['id'] in p['key_replay_ids'])
    assert sum(j['case']['step_cap'] for j in p['evaluation_jobs'])+exact==p['max_physics']
    assert 3+7+400+7+3==p['max_commands']
    assert p['fit_steps']==128 and p['keep_weight']==10 and p['fit_lr']==.001
    # Real teacher full trajectory is already independently audited. Match its
    # raw arrays now; no unsuccessful teacher or forced prefix becomes a label.
    a=R.E.N.S.arrays(row['path']);assert all(R.digest(v)==row['array_hashes'][k] for k,v in a.items())
    assert a['end_flags'][-1].tolist()==[0,1,0] and int(a['student_takeover_first'])==5
    assert a['ownership'][:5].all() and not a['ownership'][5:].any()
    q=R.L.frozen(p,p['initial']['path']);before=R.digest(q.state_dict())
    selected=R.D.selected();by={r['job_id']:r for r in selected};data=[]
    repro=ROOT/'stage/runs/t636_joint_best_reproduction'
    rr=[json.loads(v) for v in (repro/'cases.jsonl').read_text().splitlines() if json.loads(v)['role']=='fresh']
    local=[r for r in selected if Path(r['path']).exists()]
    for r in rr+local:
        arr=R.E.N.S.arrays(r['path']);original=by[r['job_id']]
        # Different serialized bytes can encode identical original full arrays.
        assert all(R.digest(v)==original['array_hashes'][k] for k,v in arr.items())
        data.append(R.D.features(q,arr,original))
    assert len(data)==11 and len({g['id'] for g in data})==11
    correction=R.M.correction(p,a,5)
    R.gpu_guard();T.os.environ['CUDA_VISIBLE_DEVICES']='0';T.os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    q.cuda();q.drive_adapter.requires_grad_(True);q.steer_adapter.requires_grad_(True)
    packed=R.M.pack(data,'cuda');new={k:v.cuda() for k,v in correction.items()}
    loss,repair,keep=R.M.objective(q,packed,new,p);loss.backward()
    gradients=[v.grad.detach().clone() for v in R.L.variables(q)]
    assert all(R.torch.isfinite(g).all() for g in gradients)
    norm=float(sum(g.square().sum() for g in gradients).sqrt());assert norm>0
    assert float(keep.detach())<1e-18 and R.digest(q.state_dict())==before
    assert float(repair.detach())>0
    q.cpu();q.zero_grad(set_to_none=True)
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    files=set(HERE.glob('*.py'))|{DOC/'PLAN.md',Path(row['path']),old/'summary.json',old/'audit.json',old/'SEALED.json',
        Path(p['teacher_protocol']),Path(p['teacher_configuration']),repro/'cases.jsonl'}|{Path(r['path']) for r in rr+local}
    teacherp=R.read(p['teacher_protocol']);p['inputs'].update(teacherp['inputs'])
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):files.add(Path(f).resolve())
    p['inputs'].update({str(f):R.sha(f) for f in files})
    own_upper=12*p['full_bytes']+p['model_bytes']+400*p['compact_bytes']+8*2**20
    assert own_upper<=p['max_bytes']
    prior_bytes=sum(R.used(ROOT/'stage/runs'/name) for name in ['t638_memory_ready_recovery','t639_neural_recovery','t640_first_divergence_recovery'])
    assert prior_bytes+p['max_bytes']<=52*2**20
    worst=R.used(ROOT/'stage/runs')+p['max_bytes']+131072;assert worst<2**30
    R.save(HERE/'protocol.json',p)
    R.save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',actual_successful_T640_source_SHA=row['sha256'],
      genuine_old_histories=11,actual_full_causal_student_features=True,masked_prefix_decisions=5,
      real_GPU_gradient_norm=norm,initial_recovery_loss=float(repair.detach()),initial_keep_loss=float(keep.detach()),model_unchanged=True,
      optimizer_created_or_stepped=0,full384_ready_gate_still_required=True,prechecks_same_candidate_reused_once=7,
      unique407_plus7exact=414,max_commands=420,combined_commands=448,combined_physics=p['prior_physics']+p['max_physics'],
      combined_main_seconds_max=p['prior_main_seconds']+p['main_seconds'],own_storage_upper=own_upper,combined_reserved_bytes=prior_bytes+p['max_bytes'],global_worst=worst))
    R.save(DOC/'BUDGET.json',{k:p[k] for k in ['max_physics','max_commands','max_bytes','preload_seconds','input_audit_seconds','main_seconds','audit_seconds','learning_seconds','all_seconds','prior_commands','prior_physics','combined_command_cap','new_Actor_Adam','shadow_Actor_Adam','independent_historical_Actor_Adam']})
    print('T641_PREFLIGHT_ZERO_NATIVE_ZERO_OPTIMIZER',norm,worst,p['prior_commands']+p['max_commands'],flush=True)


if __name__=='__main__':main()
