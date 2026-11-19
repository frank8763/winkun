"""Real-source, causality, gradients and entry checks; zero optimizer steps."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import copy,ast,gzip,json,hashlib,sys,importlib.util,os
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
DOC=ROOT/'stage/docs/overnight_20261119/resume';OUT=ROOT/'stage/runs/t660s_remaining_night_relay'
sys.path.insert(0,str(ROOT/'stage/experiments/t645_iterative_recovery_relay'));import control as Resource

def main():
    Resource.resources()
    import t660_runtime as T
    M,R,torch,np=T.M,T.R,T.torch,T.np
    assert not OUT.exists();DOC.mkdir(exist_ok=True)
    p=T.read(ROOT/'stage/runs/t656_qualified_recovery_pg/protocol.json')
    oldp=copy.deepcopy(p);initial=p['safe_retained'];contract=ROOT/'stage/docs/day_20261119/innovation_reset/UNIQUE_387_REFERENCE_CONTRACT.json'
    refs=T.read(contract);assert len(refs['reference_rows'])==387
    fixed_ids=['dev_task6_00','dev_task6_02','dev_task6_03','dev_task6_26','dev_task3_00','dev_task7_13','wide117_80_dev6_LEFT_18','wide117_8m_dev6_RIGHT_18']
    jobs={j['id']:j for j in p['evaluation_jobs']};training=p['field_jobs']+[jobs[key] for key in fixed_ids];assert len(training)==32
    allstop=datetime(2026,11,20,9,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
    learning=datetime(2026,11,20,7,30,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
    native_stop=datetime(2026,11,20,8,15,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
    p.update(id='T660_OFFLINE_JOINT_LEARNING_NIGHT',out=str(OUT),authorization_approved=False,initial=initial,
        reference_contract=str(contract),required387_ids=refs['required_success_ids'],training_jobs=training,
        original_screen_success_ids=sorted(set(p['before_success_ids'])&{j['id'] for j in p['screen_jobs']}),
        key_ids=[j['id'] for j in p['key_jobs']],arms=['bc_mean','bc_tail','ppo_mean','ppo_tail'],
        night_max_physics=240000000,night_max_commands=6500,night_main_seconds=40000,
        Actor_each_cap=1072,Value_each_cap=192,max_bytes=7*2**30,global_bytes=10*2**30,
        final_storage_reserve=32*2**20,absolute_all_stop=allstop,absolute_native_stop=native_stop,absolute_learning_stop=learning,
        backup_seconds=1800,source_seconds=240,finalization_seconds=2400,
        fit_lr=3e-5,baseline_lr=3e-4,baseline_seed=20261119660,night_seed=202611196600,
        episodes_per_task=2,bc_steps=128,ppo_epochs=4,value_steps=16,replay_weight=10.,score_scale=512.,ratio_clip=.2,
        max_task_latent_KL=.01,model_bytes=4*2**20,state_bytes=4*2**20,coverage_batches=32,coverage_replicates_per_batch=2,
        user_authorization='Deadline09:00confirmed; independent night compute and optimizer budget pending explicit approval')
    for key in ['registration_epoch','_root_sha256','combined_physics_cap','combined_command_cap','combined_main_cap',
                'combined_storage_cap','prior_physics','prior_commands','prior_main_seconds','recovery_jobs','original_jobs',
                'field_train_jobs','field_keep_rows','existing_field_full_rows','field_reconstruct_jobs','field_reconstruct_references',
                'reuse_rows','training_rows','reference_by_id','memory_sources','reconstruction_references','profile_jobs',
                'initial_corrections','target_reference','recovery_row','new_teacher','new_RL','Actor_each_cap_old']:
        p.pop(key,None)
    oldset=set(p['before_success_ids']);assert len(oldset)==384
    coverage=[j for j in p['field_jobs'] if j['id'] not in oldset];assert len(coverage)==19
    p['coverage_jobs']=coverage
    screen_cap=sum(j['case']['step_cap'] for j in p['screen_jobs']);key_cap=sum(j['case']['step_cap'] for j in p['key_jobs'])
    rest_cap=sum(j['case']['step_cap'] for j in p['evaluation_jobs'] if j['id'] not in {q['id'] for q in p['screen_jobs']})
    train_cap=4*sum(j['case']['step_cap'] for j in training)
    p['bc_template']=dict(kind='learn',arm='bc_mean',rounds=4,actor_cap=512,value_cap=0,main_seconds=3600,audit_seconds=1200,
        learning_seconds=600,block_science_reserve_seconds=700,full_evaluation_reserve_seconds=1800,
        max_physics=4*(screen_cap+key_cap)+rest_cap,max_commands=4*56+371+6+6*3,child_bytes=384*2**20)
    p['rl_template']=dict(kind='learn',arm='ppo_mean',rounds=6,actor_cap=24,value_cap=96,main_seconds=14400,audit_seconds=1800,
        learning_seconds=300,block_science_reserve_seconds=2400,full_evaluation_reserve_seconds=1800,
        max_physics=6*(train_cap+screen_cap+key_cap)+rest_cap,max_commands=6*(128+56)+371+6+6*11,child_bytes=2*2**30)
    profile_cap=2*sum(j['case']['step_cap'] for j in p['evaluation_jobs'])
    p['profile_template']=dict(kind='profile',arm='profile',actor_cap=0,value_cap=0,main_seconds=3600,audit_seconds=600,
        max_physics=profile_cap,max_commands=820,child_bytes=64*2**20)
    coverage_cap=4*sum(j['case']['step_cap'] for j in coverage)
    p['coverage_template']=dict(kind='coverage',arm='coverage',actor_cap=0,value_cap=0,main_seconds=2400,audit_seconds=600,
        max_physics=coverage_cap,max_commands=82,child_bytes=160*2**20)
    for template in [p['bc_template'],p['rl_template'],p['profile_template'],p['coverage_template']]:
        template.update(max_bytes=template['child_bytes'],fresh_bytes=3*2**20,compact_bytes=32768,full_bytes=4*2**20)
    assert p['bc_template']['actor_cap']*2+p['rl_template']['actor_cap']*2==p['Actor_each_cap']
    assert 2*p['rl_template']['value_cap']==p['Value_each_cap']
    assert R.used(ROOT/'stage/runs')+p['max_bytes']<p['global_bytes']
    # Zero-update real GPU gradient and exact cached recurrent equivalence.
    R.B.setup(p);R.gpu_guard();os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    obj=M.make(p,'cuda');q,v,ao,vo=obj;before=M.cpu_state(obj);qhash=q.phash()
    data=M.references(p,q,lambda:None);assert len(data)==387;packed=M.pack(data)
    reports={}
    for tail in [False,True]:
        ao.zero_grad(set_to_none=True);loss,losses=M.retention(q,packed,tail);loss.backward()
        grad=float(sum(x.grad.square().sum() for x in q.parameters() if x.requires_grad).sqrt())
        assert grad>0 and torch.isfinite(loss) and all(torch.isfinite(x.grad).all() for x in q.parameters() if x.requires_grad)
        reports['tail' if tail else 'mean']=dict(loss=float(loss.detach()),gradient_norm=grad)
    # Previous-block Value is initially exactly0; fitting is not performed.
    assert v(packed['f']).abs().max()==0 and not ao.state and not vo.state and digest_before(M,before,obj)
    with torch.no_grad():
        q.steer_adapter[-1].bias.add_(.01)
        for param in [x for x in q.parameters() if x.requires_grad]:ao.state[param]=dict(step=torch.tensor(7.),exp_avg=torch.full_like(param,.03),exp_avg_sq=torch.full_like(param,.06))
        for param in v.parameters():vo.state[param]=dict(step=torch.tensor(9.),exp_avg=torch.full_like(param,.02),exp_avg_sq=torch.full_like(param,.04))
        M.load(obj,before);assert digest_before(M,before,obj) and q.phash()==qhash and not ao.state and not vo.state
    q.cpu();v.cpu();del data,packed,obj,q,v,ao,vo;torch.cuda.empty_cache();os.environ['CUDA_VISIBLE_DEVICES']=''
    # Actual original-start stochastic histories exercise the NEW PPO path,
    # not a fabricated successful rollout. This is a gradient fixture only.
    source=ROOT/'stage/runs/t655_joint_current_recovery'
    sourcep=T.read(source/'protocol.json')
    rows=[json.loads(x) for x in gzip.decompress((source/'shared/cases.jsonl.gz').read_bytes()).decode().splitlines()]
    field=[r for r in rows if r['role']=='field_fresh']
    key=next(r['job_id'] for r in field if r['success'] and not r['collision'])
    chosen=[r for r in field if r['job_id']==key][:2]
    assert len(chosen)==2 and any(M.outcome(r) for r in chosen)
    for r in chosen:assert T.sha(r['path'])==r['sha256']
    groups=[(key,[(R.E.N.S.arrays(r['path']),r) for r in chosen])]
    fixturep=dict(p,initial=sourcep['initial'],arm='ppo_mean')
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    obj=M.make(fixturep,'cuda');q,v,ao,vo=obj
    items=M.fresh_items(obj,groups,fixturep);old_value=M.digest(v.state_dict());before=M.cpu_state(obj)
    loss=M.pg_loss(q,items[0],fixturep);loss.backward()
    parameters=[z for z in q.parameters() if z.requires_grad]
    g1=torch.cat([z.grad.flatten() for z in parameters]).detach().clone();ao.zero_grad(set_to_none=True)
    direct=sum(-(M.P.L.density(M.location(q,g),g['raw'],g['atom'],fixturep['raw_sigma'])*g['adv']).sum()/(len(items[0])*fixturep['score_scale']) for g in items[0])
    direct.backward();g2=torch.cat([z.grad.flatten() for z in parameters]).detach().clone()
    reinforce_error=float((g1-g2).abs().max());assert reinforce_error<1e-10
    assert v.net[-1].weight.grad is None and old_value==M.digest(v.state_dict())
    assert all(g['baseline'].abs().max()==0 for g in items[0]) and not ao.state and not vo.state
    cpu=M.make(fixturep,'cpu');M.load(cpu,before)
    cpuitems=M.fresh_items(cpu,groups,fixturep);M.pg_loss(cpu[0],cpuitems[0],fixturep).backward()
    g3=torch.cat([z.grad.flatten() for z in cpu[0].parameters() if z.requires_grad])
    cpu_gpu_error=float((g3-g1.cpu()).abs().max());assert cpu_gpu_error<1e-8
    # Change the Value numerically only after advantages are cached: existing
    # detached advantages stay fixed; a next batch sees the changed baseline.
    fixed_adv=[g['adv'].clone() for g in items[0]]
    with torch.no_grad():v.net[-1].bias.add_(.25)
    assert all(torch.equal(g['adv'],a) for g,a in zip(items[0],fixed_adv))
    nextitems=M.fresh_items(obj,groups,fixturep)
    assert all(torch.equal(g['baseline'],torch.full_like(g['baseline'],.25)) for g in nextitems[0])
    M.load(obj,before);assert digest_before(M,before,obj)
    q.cpu();v.cpu();del obj,q,v,cpu,items,cpuitems,nextitems;torch.cuda.empty_cache();os.environ['CUDA_VISIBLE_DEVICES']=''
    # Import and install actual transport without network; validate real guard
    # loader using source protocol, not a fake successful JSON receipt.
    def module(name,path):
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
    T.save(HERE/'protocol.json',p)
    trans=module('_t660_transport_preflight',HERE/'transport.py');client=trans.Prior.client()
    assert Path(trans.Prior.A.OLD.T.__file__).resolve()==ROOT/'stage/experiments/t377_full_human_observation_transfer/transport.py'
    affinity=module('_t660_affinity_preflight',HERE/'affinity_guard.py');config,df,sf=affinity.configuration(HERE)
    assert config['out']==str(OUT) and df==HERE/'DRIVER_STARTED.json'
    T.setup(dict(p,out=str(OUT/'fixture'),arm='fixture'))
    assert R.Runtime.guard.__globals__['verify'] is T.verify and O_worker(T) is T.N.worker
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    files=set(HERE.glob('*.py'))|{contract,DOC/'PLAN.md',DOC/'PRIMARY_REVIEW.md',ROOT/'stage/runs/t656_qualified_recovery_pg/protocol.json'}
    files.update(Path(row['source_path']) for row in refs['reference_rows'])
    files.update(Path(j['exact_snapshot']) for j in p['evaluation_jobs'])
    files.add(Path(initial['path']))
    files.add(Path(sourcep['initial']['path']));files.add(source/'shared/cases.jsonl.gz')
    files.update(Path(r['path']) for r in chosen)
    for m in list(sys.modules.values()):
        f=getattr(m,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):files.add(Path(f).resolve())
    p['inputs']={str(f):T.sha(f) for f in files};T.save(HERE/'protocol.json',p)
    T.save(DOC/'ENTRY_PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',actual_transport_client_install=True,
         actual_guard_configuration_loader=True,actual_pool_worker_bound=True,new_native=0,new_optimizer=0,network_calls=0))
    T.save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',references=387,unique_task_weighting=True,
         actual_GPU_mean_and_tail_gradients=reports,full_mature_Actor_Value_Adam_rollback=True,initial_value_exact_zero=True,
         new_native=0,new_optimizer=0,new_Actor=0,new_Critic=0,all_source_bytes_prepared_offline=True,
         actual_original_start_PPO_density_and_gradient=True,REINFORCE_initial_gradient_error=reinforce_error,
         Actor_CPU_GPU_gradient_error=cpu_gpu_error,previous_baseline_detached_and_next_block_only=True,
         worst_global_bytes=R.used(ROOT/'stage/runs')+p['max_bytes']))
    T.save(DOC/'BUDGET.json',{k:p[k] for k in ['night_max_physics','night_max_commands','night_main_seconds','Actor_each_cap','Value_each_cap',
        'max_bytes','global_bytes','absolute_all_stop','absolute_native_stop','absolute_learning_stop','backup_seconds','source_seconds',
        'bc_template','rl_template','profile_template','coverage_template','coverage_batches']})
    print(json.dumps(dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',gradients=reports,night_physics=p['night_max_physics'],Actor_each=p['Actor_each_cap'],Value_each=p['Value_each_cap'])))

def digest_before(M,before,obj):return M.digest(before)==M.digest(M.cpu_state(obj))
def O_worker(T):return T.R.N.Pool.__init__.__globals__['worker']
if __name__=='__main__':main()
