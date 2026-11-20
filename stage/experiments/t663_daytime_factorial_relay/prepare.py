"""Zero-update entry, distribution, objective, isolation and budget checks."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import ast,copy,gzip,importlib.util,json,os,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t663_daytime_factorial_relay';DOC=ROOT/'stage/docs/day_20261120/t663'
import t663_learning as M
import t663_wait as W
import t663_admission as admission
torch,np=M.torch,M.np
def module(name,file):
    spec=importlib.util.spec_from_file_location(name,file);value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value
def main():
    assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True)
    old=M.read(ROOT/'stage/runs/t660s_remaining_night_relay/protocol.json');p=copy.deepcopy(old)
    for key in ['registration_epoch','_root_sha256','original_arm_indices']:p.pop(key,None)
    zone=ZoneInfo('Asia/Shanghai')
    epoch=lambda hour,minute:datetime(2026,11,20,hour,minute,tzinfo=zone).timestamp()
    p.update(id='T663_DAYTIME_FACTORIAL_RELAY',out=str(OUT),authorization_approved=False,
        night_max_physics=240000000,night_max_commands=6500,night_main_seconds=28000,audit_total_seconds=3600,
        total_max_physics=240000000,total_max_commands=6500,total_main_seconds=28000,
        Actor_each_cap=48,Value_each_cap=192,new_optimizer_preflight=0,
        max_bytes=7*2**30,global_bytes=10*2**30,final_storage_reserve=32*2**20,
        absolute_learning_stop=epoch(19,0),absolute_native_stop=epoch(19,45),absolute_all_stop=epoch(20,30),
        first_start_epoch=epoch(11,0),backup_seconds=900,source_seconds=240,finalization_seconds=1800,
        night_seed=202611206630,baseline_seed=20261120663,
        arm_configurations=[dict(arm='ppo_binary_s020',arm_index=0,objective_kind='binary',raw_sigma=[.2,.2]),
            dict(arm='ppo_binary_s005',arm_index=1,objective_kind='binary',raw_sigma=[.05,.05]),
            dict(arm='ppo_collision_s020',arm_index=0,objective_kind='collision_cost',raw_sigma=[.2,.2]),
            dict(arm='ppo_collision_s005',arm_index=1,objective_kind='collision_cost',raw_sigma=[.05,.05])],
        objective_kind='binary',learner_device='cpu',admission_physics_per_second=2500,
        coverage_batches=32,coverage_replicates_per_batch=2,
        user_authorization='User confirmed today20:30 final stop and daytime scheduled experiments; new explicit aggregate computation/storage allowance pending approval')
    p['arms']=[c['arm'] for c in p['arm_configurations']]
    field=[j for j in p['field_jobs'] if j['id'] not in set(p['before_success_ids'])]
    critical=[j for j in p['screen_jobs'] if j['id'] in ['dev_task6_00','dev_task6_26','wide117_8m_dev6_RIGHT_18']]
    p['coverage_jobs']=field+critical;assert len(p['coverage_jobs'])==22
    screen=sum(j['case']['step_cap'] for j in p['screen_jobs']);keys=sum(j['case']['step_cap'] for j in p['key_jobs'])
    rest=sum(j['case']['step_cap'] for j in p['evaluation_jobs'] if j['id'] not in {j['id'] for j in p['screen_jobs']})
    train=4*sum(j['case']['step_cap'] for j in p['training_jobs'])
    p['rl_template']=dict(kind='learn',rounds=3,actor_cap=12,value_cap=48,main_seconds=7200,audit_seconds=900,
        learning_seconds=600,block_science_reserve_seconds=1500,full_evaluation_reserve_seconds=3900,
        max_physics=3*(train+screen+keys)+rest,max_commands=3*184+371+6+6*5,
        child_bytes=1152*2**20)
    coverage=4*sum(j['case']['step_cap'] for j in p['coverage_jobs'])
    p['coverage_template']=dict(kind='coverage',arm='coverage',actor_cap=0,value_cap=0,main_seconds=1800,audit_seconds=120,
        max_physics=coverage,max_commands=94,child_bytes=192*2**20)
    p.pop('bc_template',None);p.pop('profile_template',None)
    for template in [p['rl_template'],p['coverage_template']]:
        template.update(max_bytes=template['child_bytes'],fresh_bytes=4*2**20,compact_bytes=32768,full_bytes=4*2**20)
    assert p['rl_template']['max_commands']==959 and p['rl_template']['max_physics']==37361312
    p['model_bytes']=p['state_bytes']=4*2**20
    resource=W.Raw.resources();os.sched_setaffinity(0,{14});torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    R=module('_t663_runtime_preflight',HERE/'t663_runtime.py')
    driver=module('_t663_driver_preflight',HERE/'driver.py')
    assert R.R.used(ROOT/'stage/runs')+p['max_bytes']<p['global_bytes']
    assert R.R.D.B.shutil.disk_usage(ROOT).free>=40*2**30
    source=ROOT/'stage/runs/t660s_remaining_night_relay/ppo_mean'
    record=M.read(source/'ROUNDS.json')[0];row=record['training_rows'][0]
    assert M.sha(row['path'])==row['sha256'];a=M.E.N.S.arrays(row['path'])
    q0=M.L.frozen(p,p['initial']['path']);fixed=M.L.fixed_hash(q0)
    checks={}
    for config in p['arm_configurations']:
        child=dict(p);child.update(p['rl_template']);child.update(config)
        blob,policy_hash,weights=M.variant_payload(child)
        fd=os.memfd_create('stage_t663_noise_variant')
        try:
            os.write(fd,blob);path=f'/proc/self/fd/{fd}'
            q=M.L.frozen(child,path)
            assert q.phash()==policy_hash and M.digest(q.state_dict())==M.digest(q0.state_dict())
            assert M.L.fixed_hash(q)==fixed
            child['initial']=dict(p['initial'],path=path,sha256=M.sha(path),policy_hash=policy_hash)
            obj=M.make(child,'cpu')
            g=M.features(obj[0],a,row['job_id'],row['sha256'])
            with torch.no_grad():
                location=M.location(obj[0],g);assert torch.equal(location,q0.sequence(torch.tensor(a['states'])))
                # Explicit mathematical density fixture, not a new rollout.
                innovation=torch.linspace(-3,3,location.numel()).reshape_as(location)
                raw=location+location.new_tensor(child['raw_sigma'])*innovation
                atom=torch.where(raw<-1,-1,torch.where(raw>1,1,0))
                logp=M.P.L.density(location,raw,atom,child['raw_sigma'])
            fixture=dict(g,raw=raw,atom=atom,old=logp.detach(),adv=logp.new_ones(logp.shape))
            M.pg_loss(obj[0],[fixture],child).backward()
            gradient=float(sum(x.grad.square().sum() for x in obj[0].parameters() if x.requires_grad).sqrt())
            assert gradient>0 and torch.isfinite(logp).all() and all(x.grad is None for x in obj[1].parameters())
            checks[config['arm']]=dict(policy_hash=policy_hash,weights_sha256=weights,synthetic_density_fixture_gradient=gradient,optimizer_steps=0)
            del obj,q,g,fixture
        finally:os.close(fd)
    assert len({v['weights_sha256'] for v in checks.values()})==1
    assert len({v['policy_hash'] for v in checks.values()})==2
    # Real full current-policy batch validates both declared terminal targets.
    rows=record['training_rows'];groups=[]
    for job in p['training_jobs']:
        values=[]
        for row in rows:
            if row['job_id']==job['id']:
                assert M.sha(row['path'])==row['sha256'];values.append((M.E.N.S.arrays(row['path']),row))
        assert len(values)==2;groups.append((job['id'],values))
    label_report={};actual_gradients={}
    references=M.references(p,q0,lambda:None);assert len(references)==387
    packed_references=M.pack(references)
    for kind in ['binary','collision_cost']:
        child=dict(p,arm='ppo_preflight',objective_kind=kind);obj=M.make(child,'cpu')
        items=M.fresh_items(obj,groups,child)
        targets=[float(g['target_value'][0]) for task in items for g in task]
        assert set(targets)<={0.,1.} if kind=='binary' else set(targets)<={-1.,0.,1.}
        if kind=='collision_cost':assert -1. in targets
        label_report[kind]={str(v):targets.count(v) for v in sorted(set(targets))}
        loss=sum(M.pg_loss(obj[0],task,child) for task in items)/len(items)
        keep,_=M.retention(obj[0],packed_references,False)
        (loss+p['replay_weight']*keep).backward()
        gradient=torch.cat([v.grad.flatten() for v in obj[0].parameters() if v.requires_grad]).detach()
        assert torch.isfinite(gradient).all() and float(gradient.norm())>0 and not obj[2].state and not obj[3].state
        assert all(v.grad is None for v in obj[1].parameters())
        actual_gradients[kind]=gradient.clone()
    actual_objective_gradient_difference=float((actual_gradients['binary']-actual_gradients['collision_cost']).abs().max())
    assert actual_objective_gradient_difference>1e-12
    # Mature saved Adam must not share CPU step or moments across copies.
    saved=M.read_state(record['trained']);first=M.make(p,'cpu');second=M.make(p,'cpu')
    M.load(first,saved);M.load(second,saved);h=M.digest(saved)
    x=next(iter(first[3].state.values()));y=next(iter(second[3].state.values()))
    assert x['step'].data_ptr()!=y['step'].data_ptr() and x['exp_avg'].data_ptr()!=y['exp_avg'].data_ptr()
    before=float(y['step']);x['step'].add_(1);assert float(y['step'])==before and M.digest(saved)==h
    # The actual inherited runtime heartbeat must preserve its signal target.
    R.setup(dict(p,out=str(OUT/'fixture'),arm='fixture'))
    captured=[];old_verify,old_save=R.verify,R.save
    try:
        R.verify=lambda p:None;R.save=lambda file,value:captured.append(value)
        rt=R.Runtime(dict(p,main_seconds=100,max_physics=1,max_commands=6));rt.guard()
        assert captured[-1]['child_pid']==os.getpid()
    finally:R.verify,R.save=old_verify,old_save
    assert R.R.N.Pool.__init__.__globals__['worker'] is R.N.worker
    # CPU fallback selection and aggregate resource admission use real code.
    idle=driver.gpu_idle
    try:
        driver.gpu_idle=lambda:False;assert driver.select_device()=='cpu'
        driver.gpu_idle=lambda:True;assert driver.select_device()=='cuda'
    finally:driver.gpu_idle=idle
    ledger=dict(physics=0,commands=0,main_seconds=0,Actor_live=0,Value_live=0)
    c=dict(p);c.update(p['rl_template']);okay,why=admission.allowed(p,c,ledger,time.time(),0);assert okay,why
    test=dict(ledger,physics=p['night_max_physics']);assert not admission.allowed(p,c,test,time.time(),0)[0]
    helper=module('_t663_affinity_preflight',HERE/'affinity_guard.py');assert helper.OUT==OUT and helper.CG.name=='stage-t663.service'
    transport=module('_t663_transport_preflight',HERE/'transport.py');assert transport.OUT==OUT and callable(transport.Prior.client().req)
    dependencies=set(HERE.glob('*.py'))|{DOC/'PLAN.md',Path(p['reference_contract']),source/'ROUNDS.json'}
    dependencies.update(Path(r['path']) for r in rows)
    dependencies.update(Path(r['source_path']) for r in M.read(p['reference_contract'])['reference_rows'])
    for file,h in old['inputs'].items():assert M.sha(file)==h,file;dependencies.add(Path(file))
    for mod in list(sys.modules.values()):
        file=getattr(mod,'__file__',None)
        if file and str(file).startswith(str(ROOT)) and str(file).endswith('.py'):dependencies.add(Path(file).resolve())
    for file in HERE.glob('*.py'):ast.parse(file.read_text())
    p['inputs']={str(file):M.sha(file) for file in sorted(dependencies)};M.save(HERE/'protocol.json',p)
    report=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',device_preflight='cpu',noise_variants_actual_loader_and_unchanged_weights=checks,
        real64_density_and_terminal_labels=label_report,mature_Adam_independent_storage=True,
        actual387_reference_gradient=True,actual_binary_collision_gradient_difference=actual_objective_gradient_difference,
        actual_runtime_child_pid_preserved=True,actual_canonical_worker_binding=True,CPU_fallback_and_fixed_branch_device=True,
        real_full_budget_admission=True,actual_guard_and_transport_install=True,
        new_native=0,new_optimizer=0,new_GPU=0,resource=resource,worst_global_bytes=R.R.used(ROOT/'stage/runs')+p['max_bytes'])
    M.save(DOC/'PREFLIGHT.json',report);M.save(DOC/'ENTRY_PREFLIGHT.json',dict(state=report['state'],new_native=0,new_optimizer=0))
    M.save(DOC/'BUDGET.json',{k:p[k] for k in ['night_max_physics','night_max_commands','night_main_seconds','audit_total_seconds','Actor_each_cap','Value_each_cap','max_bytes','global_bytes','first_start_epoch','absolute_learning_stop','absolute_native_stop','absolute_all_stop','rl_template','coverage_template','coverage_batches']})
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
