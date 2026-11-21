"""Same targets-only real one-step path gradient/initial state preflight, no steps."""
from pathlib import Path
import ast,copy,json,sys,time
import t678_runtime as S
ROOT,HERE,OUT,DOC=S.ROOT,S.HERE,S.ROOTOUT,S.ROOTDOC
L,T,torch,np=S.L,S.T,S.torch,S.np
read,sha,digest=S.read,S.sha,S.digest
from status_io import save

def main():
    start=time.time();assert not OUT.exists();DOC.mkdir(exist_ok=True)
    source=ROOT/'stage/runs/t677_reference_target_contract';prior=read(source/'protocol.json');result=read(source/'RESULTS.json')
    assert read(source/'FINAL_REPORT.json')['state']=='COMPLETE' and read(source/'audit.json')['state']=='PASS' and result['state']=='REFERENCE_CONTRACT_PASS'
    for folder in [str(source),*prior['sealed_sources']]:
        seal=read(Path(folder)/'SEALED.json');assert seal['all_scientific_processes_stopped'] and all(not Path('/proc',str(pid)).exists() for pid in seal['scientific_pids'])
        assert not Path('/proc',str(read(Path(folder)/'DRIVER_STARTED.json')['pid'])).exists()
    p=copy.deepcopy(read(prior['science_protocol']));p.pop('registration_epoch',None);p.pop('batch_configurations',None)
    a=read(DOC.parent/'AUTHORIZATION.json');assert a['approved'];now=time.time();by={j['id']:j for j in p['evaluation_jobs']};assert len(by)==407
    old=set(p['before_success_ids']);outside=set(by)-old;assert len(old)==384 and len(outside)==23
    known={'dev_task6_00','dev_task6_03','dev_task6_26','dev_task6_20','dev_task6_21','dev_task6_23','hold_task6_06','dev_task6_13','dev_task6_25','dev_task7_17'}
    known|=set(result['direction_summaries']['executed']['old_negative_ids'])|set(result['direction_summaries']['self']['old_negative_ids'])
    screenids=sorted(set(j['id'] for j in p['screen_jobs'])|outside|known);assert len(screenids)<=80
    protected_panel=sorted(set(screenids)&old);panel=sorted(set(k for k in result['nonlinear']['executed']))
    screen=[by[k] for k in screenids];n=len(screen)
    p.update(id='T678_MATCHED_REFERENCE_TARGET_ONE_STEP',authorization_approved=False,out=str(OUT),arms=['executed','self'],target_arm=None,arm='root',
             source_target_protocol=str(source/'protocol.json'),summaries=prior['summaries'],prior_target_gradients=str(source/'gradients.npz'),outcome_gradients=prior['outcome_gradients'],
             initial=prior['initial'],raw_sigma=[.05,.05],screen_jobs=screen,key_jobs=screen,required_screen_ids=protected_panel,gradient_panel_ids=panel,
             Actor_steps=1,new_Value=0,new_optimizer=2,preflight_optimizer=0,Adam_eps=1e-8,learning_seconds=600,child_main_seconds=5700,child_audit_seconds=300,
             child_bytes=384*2**20,max_bytes=800*2**20,global_bytes=10*2**30,final_storage_reserve=16*2**20,
             child_max_physics=sum(j['case']['step_cap'] for j in list(by.values())+screen),child_max_commands=407+n+6,
             night_main_seconds=11400,audit_total_seconds=600,full_evaluation_reserve_seconds=4500,fresh_bytes=4*2**20,compact_bytes=32768,model_bytes=2*2**20,
             absolute_all_stop=min(now+13200,a['absolute_all_stop']),campaign_learning_stop=a['absolute_learning_stop'],campaign_native_stop=a['absolute_native_stop'],
             absolute_learning_stop=min(now+13200,a['absolute_learning_stop']),absolute_native_stop=min(now+12900,a['absolute_native_stop']),
             finalization_seconds=60,weekend_authorization=str(DOC.parent/'AUTHORIZATION.json'),learner_device='cpu')
    p['night_max_physics']=2*p['child_max_physics'];p['night_max_commands']=2*p['child_max_commands']
    # Real PG and retention code path, both arms at exactly the same initial
    # state. No opt.step() or fit() is called here.
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);S.Wait.Raw.resources()
    pre_errors={};initials={};delta_errors={}
    for arm in p['arms']:
        science=dict(p,target_arm=arm,out=str(OUT),arm=arm);T.setup(science);q,opt=L.make(science)
        original=digest(L.optimizer_state(q,opt));initials[arm]=original;variables=[v for v in q.parameters() if v.requires_grad]
        rows,unused,keep=L.data(science,q,lambda:None);opt.zero_grad(set_to_none=True)
        for key,rr in rows.items():
            g,y=L.B.A.group(q,rr,science);(L.Prior.pg(q,g,science)/407).backward()
        kloss,_=L.M.retention(q,keep,False);(p['replay_weight']*kloss).backward();actual=torch.cat([v.grad.detach().flatten() for v in variables]).numpy().copy()
        with np.load(p['outcome_gradients'],allow_pickle=False) as f:reward=f['task_gradients'].mean(0)
        with np.load(p['prior_target_gradients'],allow_pickle=False) as f:expected=-reward+p['replay_weight']*f[arm+'_retention_gradient'];prior_delta=f[arm+'_formula_displacement'].copy()
        err=float(np.max(np.abs(actual-expected)));assert err<1e-10;pre_errors[arm]=err
        torch.nn.utils.clip_grad_norm_(variables,1.,error_if_nonfinite=True)
        clipped=torch.cat([v.grad.detach().flatten() for v in variables]).numpy().copy()
        formula=-p['fit_lr']*clipped/(np.abs(clipped)+p['Adam_eps']);derr=float(np.max(np.abs(formula-prior_delta)));assert derr<1e-12;delta_errors[arm]=derr
        assert not opt.state and digest(L.optimizer_state(q,opt))==original and q.phash()==p['initial']['policy_hash']
        # Pack and roundtrip the entire empty optimizer state independently.
        shadow,shadowopt=L.make(science);L.load(shadow,shadowopt,L.optimizer_state(q,opt));assert digest(L.optimizer_state(shadow,shadowopt))==original
        del rows,unused,keep,q,opt,shadow,shadowopt
    assert initials['executed']==initials['self']
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    assert T.R.used(ROOT/'stage/runs')+p['max_bytes']<p['global_bytes']
    # Actual helper config and child plan shape before registration.
    import importlib.util
    spec=importlib.util.spec_from_file_location('_t678_helper_preflight',HERE/'affinity_guard.py');Helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(Helper)
    assert Helper.OUT==OUT and Helper.CG.name=='stage-t678.service' and Helper.SAFE=={8,10,12,14}
    fake=dict(p,out=str(OUT/'executed'),target_arm='executed',arm='executed',_root_sha256='fixture');assert len(fake['screen_jobs'])==len(set(j['id'] for j in fake['screen_jobs']))
    pre=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',same_initial_Actor_and_empty_Adam=True,actual_all407_PG_and_387_target_gradient_errors=pre_errors,analytic_first_delta_errors=delta_errors,
             entire_state_deepcopy_roundtrip=True,screen_full_known_failures_and_all23_missing=True,screen_count=n,screen_required_old_count=len(protected_panel),
             all_screen_exact_budgeted=True,actual_resource_helper_configuration=True,new_native=0,new_optimizer=0,resources=S.Wait.Raw.resources())
    save(DOC/'PREFLIGHT.json',pre)
    deps={Path(f) for f in prior['inputs']}|set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC.parent/'T678_PRIMARY_REVIEW.md',source/'protocol.json',source/'RESULTS.json',source/'gradients.npz',source/'audit.json',source/'FINAL_REPORT.json',source/'SEALED.json',source/'DRIVER_STARTED.json'}
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):deps.add(Path(f).resolve())
    p['inputs']={str(f):sha(f) for f in sorted(deps)};p['preflight_seconds']=time.time()-start
    save(HERE/'protocol.json',p);assert (HERE/'protocol.json').stat().st_size<2**21
    save(DOC/'BUDGET.json',{k:p[k] for k in ['arms','Actor_steps','new_Value','night_max_physics','night_max_commands','night_main_seconds','audit_total_seconds','child_main_seconds','child_audit_seconds','child_bytes','max_bytes','absolute_all_stop','preflight_seconds']})
    print(json.dumps(pre))

if __name__=='__main__':main()
