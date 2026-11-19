"""Reuse only canonical T501 data; bounded continuation, zero new optim/native."""
from pathlib import Path
import ast,copy,gzip,hashlib,json,os,time
import t651_runtime as T
ROOT,HERE,OUT,DOC=T.ROOT,T.HERE,T.OUT,T.DOC
R,M,np,torch=T.R,T.M,T.np,T.torch

def main():
    assert not OUT.exists()
    prior=ROOT/'stage/runs/t649_fresh_outcome_pg';repair=ROOT/'stage/runs/t650_reset_contract_repair'
    assert T.read(prior/'SEALED.json')['all_scientific_processes_stopped'] and T.read(repair/'audit.json')['state']=='PASS'
    base=T.read(prior/'protocol.json');ledger=T.read(prior/'LEDGER.json');fix=T.read(repair/'summary.json')
    assert fix['successes']==6 and fix['new_optimizer']==0
    p=copy.deepcopy(base);oldrows=[json.loads(v) for v in gzip.decompress((prior/'shared/cases.jsonl.gz').read_bytes()).decode().splitlines() if json.loads(v)['role']=='shared_fresh']
    refs=T.read(ROOT/'stage/runs/t650_reset_contract_repair/protocol.json')['references'];contracts={};wide=None;narrow=None;radial={}
    for row in refs:
        a=R.E.N.S.arrays(row['path']);scene=bytes(a['scene_sha256']).hex();j=row['request']['job']
        contracts[row['job_id']]=dict(scene_sha256=scene,initial120_sha256=hashlib.sha256(a['states'][0].tobytes()).hexdigest(),integration_sha256=hashlib.sha256(a['initial_native_integration'].tobytes()).hexdigest(),goal_xy=a['task_goal_xy'].tolist())
        if j.get('wide_course'):wide=scene
        else:narrow=scene
        if j.get('radial_distance_m')==8:radial[j['case']['side']]=a['task_goal_xy'].tolist()
    assert len(radial)==2 and wide!=narrow
    replacement_ids=[j['id'] for j in p['sentinel_jobs'] if j.get('wide_course')];assert len(replacement_ids)==4
    reuse={}
    for job in p['training_jobs']:
        if job['id'] in replacement_ids:continue
        rr=[r for r in oldrows if r['job_id']==job['id']];assert len(rr)==4
        for r in rr:
            assert T.sha(r['path'])==r['sha256'];a=R.E.N.S.arrays(r['path']);assert bytes(a['scene_sha256']).hex()==narrow
        reuse[job['id']]=rr
        if job['id'] not in contracts:
            a=R.E.N.S.arrays(rr[0]['path']);contracts[job['id']]=dict(scene_sha256=narrow,initial120_sha256=hashlib.sha256(a['states'][0].tobytes()).hexdigest(),integration_sha256=hashlib.sha256(a['initial_native_integration'].tobytes()).hexdigest(),goal_xy=a['task_goal_xy'].tolist())
    assert sum(len(v) for v in reuse.values())==60
    for key in ['training_jobs','screen_jobs','field_jobs','key_jobs','evaluation_jobs','sentinel_jobs']:
        for j in p[key]:
            if j['id'] in contracts:j['initial_contract']=contracts[j['id']]
    replacement=[j for j in p['training_jobs'] if j['id'] in replacement_ids]
    p.update(id='T651_CANONICAL_T501_OUTCOME_PG',out=str(OUT),arms=['pg'],arm='pg',authorization_approved=True,continuation_under_existing_T649=True,
        reuse_rows=reuse,replacement_jobs=replacement,narrow_scene=narrow,wide_scene=wide,radial8_goals=radial,
        fit_lr=.00003,fit_steps=8,shared_physics=8*sum(j['case']['step_cap'] for j in replacement),shared_main_seconds=600,shared_audit_seconds=240,
        total_max_physics=14000000,total_max_commands=465,total_main_seconds=3000,Actor_each_cap=8,
        max_bytes=192*2**20,shared_bytes=64*2**20,child_bytes=112*2**20,old_backup_seconds=0,backup_seconds=300,source_seconds=180,
        main_seconds=2400,audit_seconds=600,learning_seconds=600,
        inherited_T649_absolute_bounds_unchanged=True,prior_T649_physics=ledger['physics'],prior_T649_commands=ledger['commands'],
        prior_T650_physics=fix['physics'],prior_T650_commands=fix['commands'])
    worst=p['shared_physics']+sum(j['case']['step_cap'] for j in p['evaluation_jobs']+p['key_jobs']);assert worst<=p['total_max_physics']
    assert ledger['physics']+fix['physics']+p['total_max_physics']<=base['total_max_physics']
    assert ledger['commands']+fix['commands']+p['total_max_commands']<=base['total_max_commands']
    assert ledger['main_seconds']+fix['elapsed_s']+p['total_main_seconds']<=base['total_main_seconds']
    used_updates={}
    for who in ['live','shadow','independent_history']:
        used_updates[who]=sum(sum(json.loads(v)[0]==who and json.loads(v)[1]=='AFTER' for v in (prior/arm/'optimizer_calls.jsonl').read_text().splitlines()) for arm in ['bc','pg'])
        assert used_updates[who]+8<=base['Actor_each_cap']
    assert time.time()+600+240+2400+600+300+180+60<p['absolute_all_stop']
    assert time.time()+600+240+600<p['absolute_learning_stop']
    old_bytes=sum(f.stat().st_size for d in [prior,repair] for f in d.rglob('*') if f.is_file());assert old_bytes+p['max_bytes']<=base['max_bytes']
    # Real task-scoped boundary rule: a canonical8m goal outside +/-7 is
    # legitimate in the wide course; field24 remains restricted to its square.
    row=dict(complete=True,success=True,collision=False,field_boundary=True,request=dict(job=dict(wide_course=True)))
    assert T.safe(row) and M.outcome(row)==1
    row['request']['job']['fullfield_start_pose']=[0,0,0];assert not T.safe(row) and M.outcome(row)==0
    # Fixture existing actual canonical field trajectories; no optimizer step.
    key=next(key for key in p['new_tasks'] if len({M.outcome(r) for r in reuse[key]})>1);rr=reuse[key];arrays=[(R.E.N.S.arrays(r['path']),r) for r in rr]
    R.B.setup(p);R.gpu_guard();os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    obj=M.make(p);initial=M.digest(M.state(obj));items,diag=M.Prior.build(obj[0],[(key,arrays)],p)
    loss=M.Prior.pg_objective(obj[0],items,p);loss.backward()
    gradient=torch.cat([v.grad.reshape(-1) for v in obj[0].parameters() if v.grad is not None]);assert torch.isfinite(gradient).all() and float(gradient.norm())>0
    assert initial==M.digest(M.state(obj)) and not obj[1].state
    obj[0].cpu();del obj;torch.cuda.empty_cache()
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    # Drop superseded producer preparation-locks, retain runtime/data provenance
    # and re-add actually imported scientific modules below.
    for key in list(p['inputs']):
        if '/stage/experiments/t649_fresh_outcome_pg/' in key or '/stage/docs/day_20261119/t649/' in key:p['inputs'].pop(key)
    files=set(HERE.glob('*.py'))|{DOC/'PLAN.md',repair/'summary.json',repair/'audit.json',prior/'LEDGER.json',prior/'protocol.json'}
    for mod in list(T.sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):files.add(Path(f).resolve())
    p['inputs'].update({str(f):T.sha(f) for f in files})
    for rr in reuse.values():
        for row in rr:p['inputs'][row['path']]=row['sha256']
    for f,h in p['inputs'].items():assert T.sha(f)==h,f
    T.save(HERE/'protocol.json',p)
    T.save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',canonical_six_full_exact_repair_PASS=True,correct_scene_reuse60=True,replacement16_required=True,
      actual_old_updates=used_updates,maximum_additional_updates_each=8,task_scoped_boundary_positive_negative_checked=True,frozen_T501_density_and_full_history_checked=True,
      initial_Actor_Adam_unchanged=True,fit_lr_single_fixed_correction=.00003,no_learningrate_grid=True,
      combined_physics_upper=ledger['physics']+fix['physics']+p['total_max_physics'],combined_commands_upper=ledger['commands']+fix['commands']+p['total_max_commands'],
      shared_physics=p['shared_physics'],native_worst=worst,absolute_all_stop=p['absolute_all_stop'],not_new_independent60=True))
    T.save(DOC/'BUDGET.json',{k:p[k] for k in ['total_max_physics','total_max_commands','total_main_seconds','Actor_each_cap','max_bytes','shared_physics','shared_main_seconds','main_seconds','audit_seconds','absolute_learning_stop','absolute_native_stop','absolute_all_stop']})
    print('T651_ZERO_NATIVE_ZERO_OPTIMIZER_PASS_EXISTING_REMAINING',worst,used_updates,flush=True)

if __name__=='__main__':main()
