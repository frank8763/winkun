"""Real one-step learning path preflight, no optimizer/native calls."""
from pathlib import Path
import ast,copy,json,os,resource,sys,time
import t690_runtime as S
ROOT,HERE,OUT,DOC=S.ROOT,S.HERE,S.ROOTOUT,S.ROOTDOC
L,T,torch,np=S.L,S.T,S.torch,S.np
read,sha,digest=S.read,S.sha,S.digest
from status_io import save

def main():
    start=time.time();os.sched_setaffinity(0,{14});assert not OUT.exists();DOC.mkdir(exist_ok=True)
    assert (DOC.parent/'T690_PRIMARY_REVIEW.md').exists(),'Complete primary review before the real preflight'
    last=ROOT/'stage/runs/t687_full_actor_gradient_contract';report=read(last/'FINAL_REPORT.json');seal=read(last/'SEALED.json')
    assert report['state']=='COMPLETE' and report['audit']['state']=='PASS' and not report['audit']['full_scope_pilot_numerically_admitted']
    ids=seal['scientific_pids']+[report['helper_pid'],read(last/'DRIVER_STARTED.json')['pid']]
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    source=ROOT/'stage/runs/t678_reference_target_pilot';original=read(source/'protocol.json');p=copy.deepcopy(original)
    p.pop('registration_epoch',None);p.pop('inputs',None)
    a=read(DOC.parent/'AUTHORIZATION.json');now=time.time()
    p.update(id='T690_SCORE_NORM_WEIGHTED_PEER_LOO_ONE_STEP',authorization_approved=False,out=str(OUT),arms=['executed'],target_arm='executed',arm='root',
        source_science_protocol=str(source/'protocol.json'),baseline_estimator='SCORE_NORM_WEIGHTED_PEER_LOO',
        control_policy_hash=read(source/'executed/summary.json')['proposal']['policy_hash'],
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
    q,opt=L.make(p);before=L.optimizer_state(q,opt);theta=torch.cat([v.detach().flatten() for v in q.parameters() if v.requires_grad]).clone();original_hash=digest(before)
    def forbidden(*args,**kwargs):raise AssertionError('No optimizer/native calls in preflight')
    torch.optim.Adam.step=forbidden
    if 'mujoco' in sys.modules:sys.modules['mujoco'].mj_step=forbidden
    last_guard=[0.]
    def guard():
        assert time.time()<start+600
        if time.time()-last_guard[0]>5:S.Wait.Raw.resources();last_guard[0]=time.time()
    rows,unused,keep=L.data(p,q,guard);panel,detail=L.gradients(p,q,rows,keep,guard)
    detail['innovation_streams']=unused
    assert torch.equal(torch.cat([v.detach().flatten() for v in q.parameters() if v.requires_grad]),theta) and digest(L.optimizer_state(q,opt))==original_hash and not opt.state
    variables=[v for v in q.parameters() if v.requires_grad];torch.nn.utils.clip_grad_norm_(variables,1.,error_if_nonfinite=True)
    clipped=torch.cat([v.grad.detach().flatten() for v in variables]);formula=-p['fit_lr']*clipped/(clipped.abs()+p['Adam_eps'])
    k=0
    try:
        with torch.no_grad():
            for v in variables:v.copy_((theta+formula)[k:k+v.numel()].reshape_as(v));k+=v.numel()
        metric={key:L.B.metrics(q,g,p) for key,g in panel.items()};maxkl=max(v['task_episode_mean_KL'] for v in metric.values())
        represented=torch.cat([v.detach().flatten() for v in variables])-theta;formula_error=float((represented-formula).abs().max());assert formula_error<1e-12
        proposed_policy_hash=q.phash();assert proposed_policy_hash!=p['control_policy_hash'],'Exact old candidate is not a fresh proposal'
    finally:
        k=0
        with torch.no_grad():
            for v in variables:v.copy_(theta[k:k+v.numel()].reshape_as(v));k+=v.numel()
    assert digest(L.optimizer_state(q,opt))==original_hash and not opt.state
    shadow,shadowopt=L.make(p);L.load(shadow,shadowopt,before);assert digest(L.optimizer_state(shadow,shadowopt))==original_hash
    # Same actual canonical pool-worker and loader binding used after fit.
    assert T.N.P.frozen==T.N.L.frozen and T.R.N.worker==T.N.worker
    assert L.M.L.fixed_hash(q)==L.M.L.fixed_hash(shadow) and q.phash()==p['initial']['policy_hash']
    resources=S.Wait.Raw.resources();peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    assert peak<a['memory_bytes_cap'] and T.R.used(ROOT/'stage/runs')+p['max_bytes']<p['global_bytes']
    save(DOC/'BASELINE_PREFLIGHT.json',detail)
    pre=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',real_all407_and_387_fixed_gradient=True,gradient_error=detail['gradient_error'],
        fixtures=detail['fixtures'],numeric=detail['numeric'],weighted_vs_ordinary_cosine=detail['weighted_vs_ordinary_cosine'],
        max_panel_task_KL=maxkl,numerical_admission=maxkl<=p['max_task_latent_KL'],analytic_represented_formula_error=formula_error,
        proposed_formula_policy_hash=proposed_policy_hash,original_control_policy_hash=p['control_policy_hash'],
        innovation_streams=unused,
        canonical_worker_and_real_loader=True,whole_Actor_and_optimizer_restore_and_deepcopy=True,all_prior_processes_gone=True,
        peak_rss_bytes=peak,new_native=0,new_optimizer=0,resources=resources,elapsed_seconds=time.time()-start)
    if not pre['numerical_admission']:pre['state']='REJECTED_FIXED_KL_ZERO_NATIVE_ZERO_OPTIMIZER'
    save(DOC/'PREFLIGHT.json',pre)
    assert pre['numerical_admission'],'Fixed numerical gate rejected; no learning-rate rescue'
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    deps={Path(f) for f in original['inputs']}|set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC/'BASELINE_PREFLIGHT.json',DOC.parent/'T690_PRIMARY_REVIEW.md',DOC.parent/'T688_PRIMARY_REVIEW.md'}
    deps|={last/f for f in ['protocol.json','RESULTS.json','FINAL_REPORT.json','audit.json','SEALED.json','DRIVER_STARTED.json']}
    deps|={source/'protocol.json',source/'executed/summary.json',source/'executed/LEARNING.json',source/'executed/SCREEN.json',DOC.parent/'t689/PRE_REGISTRATION_REJECTION.json',DOC.parent/'t689/FINDINGS.md'}
    deps.add(DOC/'PRE_REGISTRATION_NOTES.json')
    for module in list(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):deps.add(Path(f).resolve())
    p['inputs']={str(f):sha(f) for f in sorted(deps)};p['preflight_seconds']=time.time()-start+read(DOC/'PRE_REGISTRATION_NOTES.json')['extra_prepare_seconds_upper']
    save(HERE/'protocol.json',p);assert (HERE/'protocol.json').stat().st_size<2**21
    save(DOC/'BUDGET.json',{k:p[k] for k in ['arms','Actor_steps','new_Value','night_max_physics','night_max_commands','night_main_seconds','audit_total_seconds','child_main_seconds','child_audit_seconds','child_bytes','max_bytes','absolute_all_stop','preflight_seconds']})
    print(json.dumps(pre),flush=True)

if __name__=='__main__':main()
