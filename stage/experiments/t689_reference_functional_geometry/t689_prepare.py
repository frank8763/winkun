"""Real one-step learning path preflight, no optimizer/native calls."""
from pathlib import Path
import ast,copy,json,os,resource,sys,time
import t689_runtime as S
ROOT,HERE,OUT,DOC=S.ROOT,S.HERE,S.ROOTOUT,S.ROOTDOC
L,T,torch,np=S.L,S.T,S.torch,S.np
read,sha,digest=S.read,S.sha,S.digest
from status_io import save

def main():
    start=time.time();os.sched_setaffinity(0,{14});assert not OUT.exists();DOC.mkdir(exist_ok=True)
    last=ROOT/'stage/runs/t687_full_actor_gradient_contract';report=read(last/'FINAL_REPORT.json');seal=read(last/'SEALED.json')
    assert report['state']=='COMPLETE' and report['audit']['state']=='PASS' and not report['audit']['full_scope_pilot_numerically_admitted']
    ids=seal['scientific_pids']+[report['helper_pid'],read(last/'DRIVER_STARTED.json')['pid']]
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    source=ROOT/'stage/runs/t678_reference_target_pilot';original=read(source/'protocol.json');p=copy.deepcopy(original)
    p.pop('registration_epoch',None);p.pop('inputs',None)
    a=read(DOC.parent/'AUTHORIZATION.json');now=time.time()
    control=read(source/'executed/summary.json')['proposal'];assert sha(control['path'])==control['sha256']
    p.update(id='T689_REFERENCE_FUNCTIONAL_NATURAL_GEOMETRY_ONE_STEP',authorization_approved=False,out=str(OUT),arms=['executed'],target_arm='executed',arm='root',
        control_model=control,control_summary=str(source/'executed/summary.json'),source_science_protocol=str(source/'protocol.json'),
        metric_damping=.1,metric_chunk=4096,metric_numeric_epsilon=1e-6,cg_iterations=10,cg_max_residual_ratio=.1,cg_early_residual_ratio=1e-8,
        root_iterations=32,root_relative_tolerance=.001,root_branch_knots=8,
        Actor_steps=1,new_Value=0,new_optimizer=1,preflight_optimizer=0,learning_seconds=1200,child_main_seconds=5700,child_audit_seconds=600,
        child_bytes=384*2**20,max_bytes=416*2**20,global_bytes=10*2**30,final_storage_reserve=16*2**20,
        night_main_seconds=5700,audit_total_seconds=600,full_evaluation_reserve_seconds=3500,
        absolute_all_stop=min(now+7200,a['absolute_all_stop']),campaign_learning_stop=a['absolute_learning_stop'],campaign_native_stop=a['absolute_native_stop'],
        absolute_learning_stop=min(now+7200,a['absolute_learning_stop']),absolute_native_stop=min(now+6900,a['absolute_native_stop']),
        finalization_seconds=60,weekend_authorization=str(DOC.parent/'AUTHORIZATION.json'),learner_device='cpu')
    p['night_max_physics']=p['child_max_physics'];p['night_max_commands']=p['child_max_commands']
    assert len(p['screen_jobs'])==55 and len(p['required_screen_ids'])==32 and len(p['gradient_panel_ids'])==68
    assert p['night_max_physics']==13625321 and p['night_max_commands']==468
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);T.setup(dict(p,out=str(OUT),arm='executed'))
    q,opt=L.make(p);before=L.optimizer_state(q,opt);theta=L.flat(q);original_hash=digest(before)
    def forbidden(*args,**kwargs):raise AssertionError('No optimizer/native calls in preflight')
    L.NaturalStep.step=forbidden
    if 'mujoco' in sys.modules:sys.modules['mujoco'].mj_step=forbidden
    last_guard=[0.]
    def guard():
        assert time.time()<start+600
        if time.time()-last_guard[0]>5:S.Wait.Raw.resources();last_guard[0]=time.time()
    rows,unused,keep=L.data(p,q,guard);delta,detail,metric=L.proposal(p,q,rows,keep,guard)
    assert L.torch.equal(L.flat(q),theta) and digest(L.optimizer_state(q,opt))==original_hash and not opt.state
    shadow,shadowopt=L.make(p);L.load(shadow,shadowopt,before);assert digest(L.optimizer_state(shadow,shadowopt))==original_hash
    # Same actual canonical pool-worker and loader binding used after fit.
    assert T.N.P.frozen==T.N.L.frozen and T.R.N.worker==T.N.worker
    assert L.M.L.fixed_hash(q)==L.M.L.fixed_hash(shadow) and q.phash()==p['initial']['policy_hash']
    resources=S.Wait.Raw.resources();peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    assert peak<a['memory_bytes_cap'] and T.R.used(ROOT/'stage/runs')+p['max_bytes']<p['global_bytes']
    save(DOC/'PROPOSED_STEP_PREFLIGHT.json',detail)
    pre=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',real_all407_and_387_fixed_gradient=True,gradient_error=detail['gradient_error'],
        original384_only_Gaussian_reference_metric=True,metric_numeric=detail['metric_numeric'],cg_residual_ratio=detail['cg_residual_ratio'],
        control_mean_KL=detail['control_reference384_mean_latent_KL'],proposed_mean_KL=detail['matched_reference384_mean_latent_KL'],
        max_panel_task_KL=detail['max_actual_sigma_task_mean_KL'],numerical_admission=detail['max_actual_sigma_task_mean_KL']<=p['max_task_latent_KL'],
        canonical_worker_and_real_loader=True,whole_Actor_and_optimizer_restore_and_deepcopy=True,all_prior_processes_gone=True,
        peak_rss_bytes=peak,new_native=0,new_optimizer=0,resources=resources,elapsed_seconds=time.time()-start)
    # No failed numerical gate is repaired here. A registered real step is
    # warranted only when its fixed original gate already admits the proposal.
    assert pre['numerical_admission'],'Fixed numerical gate rejected; no new pilot and no matching-target rescue'
    save(DOC/'PREFLIGHT.json',pre)
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    deps={Path(f) for f in original['inputs']}|set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC/'PROPOSED_STEP_PREFLIGHT.json',DOC.parent/'T689_PRIMARY_REVIEW.md',DOC.parent/'T688_PRIMARY_REVIEW.md'}
    deps|={last/f for f in ['protocol.json','RESULTS.json','FINAL_REPORT.json','audit.json','SEALED.json','DRIVER_STARTED.json']}
    deps|={Path(p['control_model']['path']),source/'protocol.json',source/'executed/summary.json',source/'executed/LEARNING.json',source/'executed/SCREEN.json'}
    for module in list(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):deps.add(Path(f).resolve())
    p['inputs']={str(f):sha(f) for f in sorted(deps)};p['preflight_seconds']=time.time()-start
    save(HERE/'protocol.json',p);assert (HERE/'protocol.json').stat().st_size<2**21
    save(DOC/'BUDGET.json',{k:p[k] for k in ['arms','Actor_steps','new_Value','night_max_physics','night_max_commands','night_main_seconds','audit_total_seconds','child_main_seconds','child_audit_seconds','child_bytes','max_bytes','absolute_all_stop','preflight_seconds']})
    print(json.dumps(pre),flush=True)

if __name__=='__main__':main()
