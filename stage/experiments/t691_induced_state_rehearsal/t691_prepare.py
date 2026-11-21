"""Real390-ID/32 induced-history supervised entry, no optimizer/native calls."""
from pathlib import Path
import ast,copy,json,os,resource,sys,time
import t691_runtime as S
ROOT,HERE,OUT,DOC=S.ROOT,S.HERE,S.ROOTOUT,S.ROOTDOC
L,T,torch,np=S.L,S.T,S.torch,S.np
read,sha,digest=S.read,S.sha,S.digest
from status_io import save

def main():
    start=time.time();os.sched_setaffinity(0,{14});assert not OUT.exists() and (DOC.parent/'T691_OCCUPANCY_PRIMARY_REVIEW.md').exists()
    previous=ROOT/'stage/runs/t690_score_weighted_baseline';report=read(previous/'FINAL_REPORT.json');seal=read(previous/'SEALED.json')
    assert report['state']=='COMPLETE' and report['children'][0]['audit']['state']=='PASS'
    ids=seal['scientific_pids']+[report['helper_pid'],read(previous/'DRIVER_STARTED.json')['pid']]
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    original=read(previous/'protocol.json');p=copy.deepcopy(original);p.pop('inputs',None);p.pop('registration_epoch',None)
    a=read(DOC.parent/'AUTHORIZATION.json');now=time.time();old=set(p['before_success_ids'])
    induced=[r for r in read(previous/'executed/summary.json')['screen'] if r['job_id'] in old]
    assert len(induced)==32 and len({r['job_id'] for r in induced})==32
    transfer=ROOT/'stage/runs/t681_frozen_variant_transfer';source=read(transfer/'summary.json')['models']['direction00_minus']
    novel=[r for r in source['field_rows'] if r['safe'] and not r['T501']['safe']]
    expected={'t502_field_sector0_offset1','t502_field_sector0_offset2','t502_field_sector1_offset1','t502_field_sector2_offset1','t502_field_sector5_offset1','t502_field_sector7_offset1'}
    assert {r['id'] for r in novel}==expected and len(novel)==6
    p.update(id='T691_FIXED_INDUCED_HISTORY_REHEARSAL_PAIR',authorization_approved=False,out=str(OUT),arms=['nominal','induced'],target_arm='executed',arm='root',
        novel_rows=novel,novel_source_model=source['model'],induced_rows=induced,old_rehearsal_weight=10.,
        fit_lr=1e-4,Adam_eps=1e-8,Actor_steps=64,new_Value=0,new_optimizer=128,preflight_optimizer=0,
        learning_seconds=1200,child_main_seconds=5700,child_audit_seconds=900,child_bytes=384*2**20,max_bytes=800*2**20,
        global_bytes=10*2**30,final_storage_reserve=16*2**20,night_main_seconds=11400,audit_total_seconds=1800,full_evaluation_reserve_seconds=3500,
        absolute_all_stop=min(now+14400,a['absolute_all_stop']),campaign_learning_stop=a['absolute_learning_stop'],campaign_native_stop=a['absolute_native_stop'],
        absolute_learning_stop=min(now+14400,a['absolute_learning_stop']),absolute_native_stop=min(now+14100,a['absolute_native_stop']),
        finalization_seconds=60,weekend_authorization=str(DOC.parent/'AUTHORIZATION.json'),learner_device='cpu')
    p['night_max_physics']=2*p['child_max_physics'];p['night_max_commands']=2*p['child_max_commands']
    assert len(p['screen_jobs'])==55 and p['night_max_physics']==27250642 and p['night_max_commands']==936
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);T.setup(dict(p,out=str(OUT),arm='nominal'))
    q,opt=L.make(p);before=L.optimizer_state(q,opt);initial=digest(before);variables=[v for v in q.parameters() if v.requires_grad]
    theta=torch.cat([v.detach().flatten() for v in variables]).clone()
    def forbidden(*args,**kwargs):raise AssertionError('T691 preflight forbids optimizer/native calls')
    torch.optim.Adam.step=forbidden
    if 'mujoco' in sys.modules:sys.modules['mujoco'].mj_step=forbidden
    last=[0.]
    def guard():
        assert time.time()<start+300
        if time.time()-last[0]>5:S.Wait.Raw.resources();last[0]=time.time()
    panel,unused,keep=L.data(p,q,guard);numeric={};gradients={};times={}
    for arm in p['arms']:
        pc=dict(p,arm=arm);tick=time.time();numeric[arm]=L.entry_numeric(q,keep,pc)
        q.zero_grad(set_to_none=True);value,parts=L.loss(q,keep,pc);value.backward()
        gradients[arm]=torch.cat([v.grad.detach().flatten() for v in variables]).clone();times[arm]=time.time()-tick
    assert float((gradients['nominal']-gradients['induced']).abs().max())<1e-12
    # One temporary represented first-Adam formula is used only to verify
    # induced labels produce a distinct restoring gradient away from theta0.
    g=gradients['nominal'];gc=g*min(1.,1./max(float(g.norm()),1e-30));formula=-p['fit_lr']*gc/(gc.abs()+p['Adam_eps'])
    try:
        k=0
        with torch.no_grad():
            for v in variables:v.copy_((theta+formula)[k:k+v.numel()].reshape_as(v));k+=v.numel()
        away={}
        for arm in p['arms']:
            q.zero_grad(set_to_none=True);L.loss(q,keep,dict(p,arm=arm))[0].backward()
            away[arm]=torch.cat([v.grad.detach().flatten() for v in variables]).clone()
        difference=float((away['nominal']-away['induced']).norm());assert difference>1e-12
    finally:
        k=0
        with torch.no_grad():
            for v in variables:v.copy_(theta[k:k+v.numel()].reshape_as(v));k+=v.numel()
        q.zero_grad(set_to_none=True)
    assert digest(L.optimizer_state(q,opt))==initial and not opt.state
    clone,cloneopt=L.make(p);L.load(clone,cloneopt,before);assert digest(L.optimizer_state(clone,cloneopt))==initial
    assert T.N.P.frozen==T.N.L.frozen and T.R.N.worker==T.N.worker
    resources=S.Wait.Raw.resources();peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    assert peak<a['memory_bytes_cap'] and T.R.used(ROOT/'stage/runs')+p['max_bytes']<p['global_bytes']
    assert max(times.values())*64*2<1100,'Estimated full live+shadow learning does not fit fixed1200s'
    pre=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',all384_nominal_and_all32_actual_induced_histories=True,
        induced_success=29,induced_collision=3,old_function_labels_not_success=True,new6_unique_relative_T501_only3_new_vs387=True,
        new_source_model_owncarry_locations_and_actual_commands=True,initial_two_gradients_equal=True,numeric=numeric,
        first_virtual_step_restoring_gradient_difference=difference,matched_old_perID_weight=True,local_old_targets_float64_prequantization=True,
        runtime_gradient_seconds=times,peak_rss_bytes=peak,all_state_and_Adam_exact_restore=True,loader_canonical_worker_bound=True,
        all_prior_processes_exited=True,new_native=0,new_optimizer=0,resources=resources,elapsed_seconds=time.time()-start)
    save(DOC/'PREFLIGHT.json',pre)
    deps={Path(f) for f in original['inputs']}|set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC.parent/'T691_OCCUPANCY_PRIMARY_REVIEW.md'}
    deps|={previous/f for f in ['protocol.json','FINAL_REPORT.json','SEALED.json','DRIVER_STARTED.json','executed/summary.json','executed/audit.json','executed/SCREEN.json']}
    deps|={transfer/f for f in ['protocol.json','summary.json','audit.json','FINAL_REPORT.json','SEALED.json']}
    deps|={Path(source['model']['path'])}|{Path(r['path']) for r in induced}|{Path(r['path']) for r in novel}
    for module in list(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):deps.add(Path(f).resolve())
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    p['inputs']={str(f):sha(f) for f in sorted(deps)};p['preflight_seconds']=time.time()-start
    save(HERE/'protocol.json',p);assert (HERE/'protocol.json').stat().st_size<2**21
    save(DOC/'BUDGET.json',{k:p[k] for k in ['arms','Actor_steps','new_Value','night_max_physics','night_max_commands','night_main_seconds','audit_total_seconds','child_main_seconds','child_audit_seconds','child_bytes','max_bytes','absolute_all_stop','preflight_seconds']})
    print(json.dumps(pre),flush=True)

if __name__=='__main__':main()
