"""Real qualified56 data, all historical replay and GPU gradients; zero steps."""
from pathlib import Path
import ast,copy,json,sys,time
import t656_runtime as T
ROOT,HERE,OUT,DOC=T.ROOT,T.HERE,T.OUT,T.DOC
M,R,torch,np=T.M,T.R,T.torch,T.np

def main():
    assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True)
    source=ROOT/'stage/runs/t655_joint_current_recovery';p=T.read(source/'protocol.json');p.pop('registration_epoch',None);p.pop('_root_sha256',None)
    assert T.read(source/'FINAL_REPORT_FINAL.json')['remote']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    sealed=T.read(source/'SEALED.json');assert all(not Path('/proc',str(v)).exists() for v in sealed['scientific_pids'])
    prior654=T.read(ROOT/'stage/runs/t654_verified_deviation_recovery/LEDGER.json');prior655=T.read(source/'LEDGER.json')
    p.update(id='T656_QUALIFIED_RECOVERY_DATA_SINGLE_ARM_PG',out=str(OUT),authorization_approved=False,arms=['pg'],arm='pg',recovery_source=str(source),
        total_max_physics=12100000,total_max_commands=433,total_main_seconds=2700,Actor_each_cap=8,
        main_seconds=2700,audit_seconds=450,learning_seconds=300,full_evaluation_reserve_seconds=1700,max_physics=12100000,max_commands=433,
        max_bytes=128*2**20,child_bytes=124*2**20,new_Actor_Adam=8,shadow_Actor_Adam=8,independent_historical_Actor_Adam=8,new_RL=8,new_Critic=0,preflight_optimizer=0,
        prior_physics=prior654['physics']+prior655['physics'],prior_commands=prior654['commands']+prior655['commands'],prior_main_seconds=prior654['main_seconds']+prior655['main_seconds'])
    worst=sum(j['case']['step_cap'] for j in p['evaluation_jobs']+p['key_jobs']);assert worst<=p['total_max_physics']
    storage=20*p['fresh_bytes']+p['model_bytes']+407*p['compact_bytes']+p['full_bytes']+24*2**20;assert storage<=p['max_bytes']
    assert p['prior_physics']+p['total_max_physics']<=p['combined_physics_cap'] and p['prior_commands']+p['total_max_commands']<=p['combined_command_cap']
    assert p['prior_main_seconds']+p['total_main_seconds']<=p['combined_main_cap']
    assert sum(R.used(ROOT/'stage/runs'/name) for name in ['t654_verified_deviation_recovery','t655_joint_current_recovery'])+p['max_bytes']<=p['combined_storage_cap']
    R.B.setup(p);old,new=T.data(p,lambda:None);assert len(old)==384 and len(new['groups'])==11
    R.gpu_guard();T.os.environ['CUDA_VISIBLE_DEVICES']='0';T.os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';obj=M.make(p);q,opt=obj;before=M.state(obj);bh=M.digest(before)
    items,diag=M.build(q,new['groups'],p);assert all(0<v['successes']<8 for v in diag[:3])
    for g in items:(M.task_loss(q,g,p)/len(items)).backward()
    variables=[v for v in q.parameters() if v.requires_grad];norm=float(sum(v.grad.square().sum() for v in variables).sqrt());assert norm>0 and all(torch.isfinite(v.grad).all() for v in variables)
    keep=M.pack(old,'cuda');fields=M.pack(new['fields'],'cuda');a=M.keep_loss(q,keep);b=M.keep_loss(q,fields)
    assert float(b.detach())<1e-18 and M.digest(M.state(obj))==bh and not opt.state
    with torch.no_grad():
        for v in variables:opt.state[v]=dict(step=torch.tensor(5.),exp_avg=torch.full_like(v,.03125),exp_avg_sq=torch.full_like(v,.0625))
        q.steer_adapter[-1].bias.add_(.01);M.load(obj,before);assert M.digest(M.state(obj))==bh and not opt.state
    q.cpu();del obj,q,items,keep,fields;torch.cuda.empty_cache()
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    files=set(HERE.glob('*.py'))|{DOC/'PLAN.md',ROOT/'stage/docs/day_20261119/t655/PRIMARY_REVIEW.md',source/'FINAL_REPORT_FINAL.json',source/'shared/summary.json',source/'shared/audit.json'}
    files|={Path(row['path']) for key,group in new['groups'] for a,row in group}
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):files.add(Path(f).resolve())
    p['inputs'].update({str(f):T.sha(f) for f in files})
    for f,h in p['inputs'].items():assert T.sha(f)==h,f
    T.save(HERE/'protocol.json',p)
    T.save(DOC/'DATA_MANIFEST.json',T.data_manifest(old,new))
    T.save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',actual_reused56_full_history_and_density=True,all384_and8_current_history=True,
        recovery_successes=[v['successes'] for v in diag[:3]],actual_GPU_PG_gradient_norm=norm,initial_field_keep_zero=float(b.detach()),
        full_Actor_Adam_synthetic_mature_rollback=True,new_native=0,new_optimizer=0,native_worst=worst,commands_worst=433,storage_worst=storage))
    T.save(DOC/'BUDGET.json',{k:p[k] for k in ['total_max_physics','total_max_commands','total_main_seconds','Actor_each_cap','max_bytes','global_bytes','main_seconds','audit_seconds','learning_seconds','backup_seconds','source_seconds','absolute_learning_stop','absolute_native_stop','absolute_all_stop','prior_physics','prior_commands','prior_main_seconds','combined_physics_cap','combined_command_cap','combined_main_cap','combined_Actor_cap','combined_storage_cap']})
    print('T656_ZERO_NATIVE_ZERO_OPTIMIZER_PASS',worst,storage,norm,flush=True)
if __name__=='__main__':main()
