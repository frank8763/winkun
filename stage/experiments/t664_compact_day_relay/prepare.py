"""Reconcile bounded old work, prepare fewer future blocks, zero optimizer."""
from pathlib import Path
import ast,copy,gzip,importlib.util,json,os,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t664_compact_day_relay';DOC=ROOT/'stage/docs/day_20261120/t664'
OLD=ROOT/'stage/runs/t663_daytime_factorial_relay'
import t664_learning as M
import t664_wait as W
import t664_admission as admission
torch,np=M.torch,M.np

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value

def main():
    assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True)
    boundary=M.read(DOC/'BOUNDARY_STOP.json');old=M.read(OLD/'protocol.json')
    assert boundary['state']=='USER_REQUESTED_COMPACT_SCHEDULE_BOUNDARY_STOP'
    ids={boundary['driver'],boundary['child'],boundary['helper']}
    for file in OLD.glob('*/worker_generations.json'):
        for group in M.read(file):ids.update([group['parent'],*group['workers']])
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    baseline=M.read(OLD/'LEDGER.json');assert len(baseline['children'])==1 and baseline['children'][0]['audit']['state']=='PASS'
    low=OLD/'ppo_binary_s005';rounds=M.read(low/'ROUNDS.json');assert len(rounds)==1 and rounds[0]['round']==0
    native=[]
    for file in low.glob('native_*.json.gz'):
        value=json.loads(gzip.decompress(file.read_bytes()));native.append(value['native_result'])
    # At most three native requests are in flight per synchronous batch. Raw
    # completed returns plus three full task caps conservatively cover SIGKILL.
    low_physics_upper=sum(row['physics'] for row in native)+3*max(j['case']['step_cap'] for j in old['evaluation_jobs'])
    low_commands_upper=max(boundary['last_status']['commands'],len(native)+6*len(M.read(low/'worker_generations.json'))+3)
    prior_physics=baseline['physics']+low_physics_upper
    prior_commands=baseline['commands']+low_commands_upper
    prior_main=baseline['main_seconds']+boundary['epoch']-M.read(low/'science_started.json')['epoch']+10
    prior_audit=baseline['audit_seconds']
    optim={who:{kind:0 for kind in ['actor','value']} for who in ['live','shadow','independent_history']}
    for folder in OLD.iterdir():
        f=folder/'optimizer_calls.jsonl'
        if not folder.is_dir() or not f.exists():continue
        for row in [json.loads(x) for x in f.read_text().splitlines()]:
            if row[3]=='BEFORE':optim[row[0]][row[2]]+=1
    old_bytes=sum(f.stat().st_size for f in OLD.rglob('*') if f.is_file())
    reconcile=dict(state='BOUNDED_ACCOUNTING_COMPLETE',all_old_scientific_pids_gone=True,pids=sorted(ids),
        prior_physics_upper=prior_physics,prior_commands_upper=prior_commands,prior_main_seconds_upper=prior_main,
        prior_audit_seconds=prior_audit,prior_optimizer=optim,prior_bytes=old_bytes,
        low_actual_complete_native_returns=len(native),low_inflight_full_task_reserve=3,
        low_prefix_complete_rounds=1,completed_baseline_rounds=3,new_native=0,new_optimizer=0)
    M.save(DOC/'BOUNDARY_RECONCILIATION.json',reconcile)
    p=copy.deepcopy(old)
    p.pop('registration_epoch',None);p.pop('_root_sha256',None)
    record=rounds[0];initial=M.read(low/'INITIAL_MODEL.json')
    resume_state=record['trained'] if record['research_state_kept'] else record['rejected_Actor_kept_Value']
    resume_model=record['proposal'] if record['research_state_kept'] else initial
    resume_ids=set(p['original_screen_success_ids'])
    if record['research_state_kept']:resume_ids.update(r['job_id'] for r in record['screen'] if r['success'] and not r['collision'])
    p.update(id='T664_USER_SHORTER_DAY_SEGMENTS',out=str(OUT),authorization_approved=False,
        night_max_physics=old['night_max_physics']-prior_physics,night_max_commands=old['night_max_commands']-prior_commands,
        night_main_seconds=int(old['night_main_seconds']-prior_main),audit_total_seconds=int(old['audit_total_seconds']-prior_audit),
        Actor_each_cap=min(old['Actor_each_cap']-v['actor'] for v in optim.values()),
        Value_each_cap=min(old['Value_each_cap']-v['value'] for v in optim.values()),
        max_bytes=old['max_bytes']-old_bytes,learner_device='cpu',first_start_epoch=time.time(),
        prior_accounting=reconcile,original_authorization=str(ROOT/'stage/docs/day_20261120/t663/AUTHORIZATION.json'),
        prefix_rounds=str(low/'ROUNDS.json'),prefix_cases=str(low/'cases.jsonl.gz'),prefix_initial_model=str(low/'INITIAL_MODEL.json'),
        arm_configurations=[dict(arm='ppo_binary_s005_remaining',arm_index=1,objective_kind='binary',raw_sigma=[.05,.05],resume_round=1,
            resume_state=resume_state,resume_model=resume_model,resume_screen_ids=sorted(resume_ids)),
            dict(arm='ppo_collision_s020',arm_index=0,objective_kind='collision_cost',raw_sigma=[.2,.2],
                reuse_training_round=str(OLD/'ppo_binary_s020/ROUNDS.json'),reuse_training_cases=str(OLD/'ppo_binary_s020/cases.jsonl.gz')),
            dict(arm='ppo_collision_s005',arm_index=1,objective_kind='collision_cost',raw_sigma=[.05,.05],
                reuse_training_round=str(low/'ROUNDS.json'),reuse_training_cases=str(low/'cases.jsonl.gz'))],
        user_authorization='User requested shorter runtime segments and20:00 return review; only reduced future fixed work, unchanged original20:30 hard stop and aggregate budgets')
    p['arms']=[a['arm'] for a in p['arm_configurations']]
    for config in p['arm_configurations']:
        if config.get('reuse_training_round'):
            source=M.read(config['reuse_training_round'])[0]
            if source['training_rows'][0]['policy_hash']!= (initial['policy_hash'] if config['raw_sigma']==[.05,.05] else p['initial']['policy_hash']):raise AssertionError('Different behavior policy')
    base=dict(old['rl_template']);base.update(rounds=2,actor_cap=8,value_cap=32,main_seconds=5400,audit_seconds=600,
        max_physics=28068463,max_commands=763,child_bytes=768*2**20,max_bytes=768*2**20)
    p['rl_template']=base
    p['resume_template']=dict(base,rounds=1,actor_cap=4,value_cap=16,main_seconds=4500,audit_seconds=300,max_physics=18775614,max_commands=567,child_bytes=512*2**20,max_bytes=512*2**20)
    p['prefix_template']=dict(kind='audit_adoption',actor_cap=4,value_cap=16,audit_seconds=300,main_seconds=0,child_bytes=16*2**20)
    assert 24<=p['Actor_each_cap'] and 96<=p['Value_each_cap']
    for name in ['absolute_learning_stop','absolute_native_stop','absolute_all_stop','global_bytes']:
        assert p[name]==old[name]
    resources=W.Raw.resources();os.sched_setaffinity(0,{14});torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    R=module('_t664_preflight_runtime',HERE/'t664_runtime.py')
    assert R.R.used(ROOT/'stage/runs')+p['max_bytes']<p['global_bytes']
    assert R.R.D.B.shutil.disk_usage(ROOT).free>=40*2**30
    def guard():W.Raw.resources()
    data_reuse=[]
    for config in p['arm_configurations'][1:]:
        child=dict(p);child.update(base);child.update(config)
        child['initial']=initial if child['raw_sigma']==[.05,.05] else p['initial']
        R.setup(child)
        rows=R.verified_reuse(child,child['initial'])
        assert len(rows)==64
        q=M.L.frozen(child,child['initial']['path']);obj=M.make(child,'cpu')
        groups=[(j['id'],[(M.E.N.S.arrays(r['path']),r) for r in rows if r['job_id']==j['id']]) for j in child['training_jobs']]
        items=M.fresh_items(obj,groups,child);assert len(items)==32
        (sum(M.pg_loss(obj[0],g,child) for g in items)/len(items)).backward()
        gradient=torch.cat([x.grad.flatten() for x in obj[0].parameters() if x.requires_grad]);assert torch.isfinite(gradient).all() and gradient.norm()>0
        assert not obj[2].state and not obj[3].state
        data_reuse.append(dict(arm=child['arm'],full64=True,exact64=True,policy_hash=q.phash(),no_new_native=True,gradient_norm=float(gradient.norm())))
    # Saved low-prefix state remains bitwise equal after fresh deep-copy load.
    child=dict(p,raw_sigma=[.05,.05],initial=initial);obj=M.make(child,'cpu');shadow=M.make(child,'cpu');s=M.read_state(resume_state)
    M.load(obj,s);M.load(shadow,s);assert M.digest(M.cpu_state(obj))==M.digest(M.cpu_state(shadow))==resume_state['state_hash']
    x=next(iter(obj[3].state.values()))['step'];y=next(iter(shadow[3].state.values()))['step'];assert x.data_ptr()!=y.data_ptr()
    # Actual plan shape with output overrides, heartbeat and canonical worker.
    R.setup(dict(p,out=str(OUT/'fixture'),arm='fixture'))
    previous_verify,previous_save=R.verify,R.save;captured=[]
    try:
        R.verify=lambda p:None;R.save=lambda f,v:captured.append(v)
        rt=R.Runtime(dict(p,main_seconds=100,max_commands=6,max_physics=1));rt.guard();assert captured[-1]['child_pid']==os.getpid()
    finally:R.verify,R.save=previous_verify,previous_save
    assert R.R.N.Pool.__init__.__globals__['worker'] is R.N.worker
    transport=module('_t664_transport_fixture',HERE/'transport.py');assert transport.OUT==OUT and callable(transport.Prior.client().req)
    helper=module('_t664_guard_fixture',HERE/'affinity_guard.py');assert helper.OUT==OUT and helper.CG.name=='stage-t664.service'
    driver=module('_t664_driver_fixture',HERE/'driver.py');prefix=driver.prefix_plan(p)
    assert prefix['arm'].startswith('ppo') and prefix['ppo_epochs']==4 and prefix['value_steps']==16
    assert prefix['raw_sigma']==[.05,.05] and prefix['learner_device']=='cpu'
    dependencies=set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'BOUNDARY_STOP.json',DOC/'BOUNDARY_RECONCILIATION.json',Path(p['original_authorization']),low/'ROUNDS.json',low/'cases.jsonl.gz',low/'INITIAL_MODEL.json',Path(resume_state['path']),Path(resume_model['path'])}
    for file,h in old['inputs'].items():assert M.sha(file)==h,file;dependencies.add(Path(file))
    for folder in [OLD/'ppo_binary_s020',low]:
        for file in folder.rglob('*'):
            if file.is_file() and file.name not in ['main.log','audit.log']:dependencies.add(file)
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):dependencies.add(Path(f).resolve())
    for file in HERE.glob('*.py'):ast.parse(file.read_text())
    p['inputs']={str(file):M.sha(file) for file in sorted(dependencies)};M.save(HERE/'protocol.json',p)
    result=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',actual_shared_behavior_full_history_density_and_exact_reuse=data_reuse,
        resume_Actor_Value_and_separate_Adam_bitwise=True,actual_runtime_heartbeat_and_worker=True,actual_guard_transport_loader=True,
        all_old_processes_gone=True,aggregate_budget_subtracted=True,absolute_deadlines_unchanged=True,new_native=0,new_optimizer=0,new_GPU=0,resources=resources)
    M.save(DOC/'PREFLIGHT.json',result);M.save(DOC/'ENTRY_PREFLIGHT.json',dict(state=result['state'],new_native=0,new_optimizer=0))
    M.save(DOC/'BUDGET.json',{key:p[key] for key in ['night_max_physics','night_max_commands','night_main_seconds','audit_total_seconds','Actor_each_cap','Value_each_cap','max_bytes','global_bytes','absolute_learning_stop','absolute_native_stop','absolute_all_stop','rl_template','resume_template','prefix_template']})
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
