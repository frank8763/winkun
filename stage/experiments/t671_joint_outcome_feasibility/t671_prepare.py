"""Audited first-direction equivalence and finite single-candidate admission."""
from pathlib import Path
import ast,copy,json,os,sys,time
import t671_runtime as S
ROOT,HERE,OUT,DOC=S.ROOT,S.HERE,S.OUT,S.DOC
T,L,read,sha,digest,torch,np=S.T,S.L,S.read,S.sha,S.digest,S.torch,S.np
from status_io import save


def main():
    assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True)
    source=ROOT/'stage/runs/t669_old_task_onpolicy_signal';gate=ROOT/'stage/runs/t670_outcome_gradient_gate'
    result=read(gate/'RESULTS.json');assert read(gate/'audit.json')['state']=='PASS'
    assert result['checks']['numeric_all'] and result['checks']['all_four_nonzero'] and result['checks']['new_goal_ascent']
    assert not result['checks']['direction_changed'] and not result['checks']['raw_has_old_conflict']
    assert read(gate/'FINAL_REPORT.json')['all_scientific_processes_stopped']
    assert not Path('/proc',str(read(gate/'DRIVER_STARTED.json')['pid'])).exists()
    for folder in [source,gate]:
        for pid in read(folder/'SEALED.json')['scientific_pids']:assert not Path('/proc',str(pid)).exists()
    p=copy.deepcopy(read(source/'protocol.json'));p.pop('registration_epoch',None)
    auth=read(DOC.parent/'AUTHORIZATION.json');now=time.time();ids=read(gate/'protocol.json')['old_ids']
    by={j['id']:j for j in p['evaluation_jobs']};screen=p['screen_jobs']+[by[k] for k in ids]
    keys=p['key_jobs']+[by[k] for k in ids];assert len(screen)==40 and len({j['id'] for j in screen})==40
    assert len(keys)==24 and len({j['id'] for j in keys})==24
    p.update(id='T671_SINGLE_JOINT_OUTCOME_FEASIBILITY',authorization_approved=False,out=str(OUT),arm='joint_outcome_feasibility',
        initial=read(source/'INITIAL_MODEL.json'),old_summary=str(source/'summary.json'),old_ids=ids,
        fresh_rounds=read(gate/'protocol.json')['fresh_rounds'],raw_sigma=[.05,.05],screen_jobs=screen,key_jobs=keys,
        required_screen_ids=sorted(set(p['original_screen_success_ids'])|set(ids)),Actor_steps=4,new_Value=0,
        old_surrogate_tolerance=1e-12,Adam_eps=1e-8,max_commands=437,max_bytes=96*2**20,child_bytes=96*2**20,
        main_seconds=5700,audit_seconds=600,learning_seconds=600,full_evaluation_reserve_seconds=4500,
        absolute_learning_stop=min(now+900,auth['absolute_learning_stop']),
        absolute_native_stop=min(now+5700,auth['absolute_native_stop']),absolute_all_stop=min(now+7200,auth['absolute_all_stop']),
        finalization_seconds=60,new_optimizer=4,weekend_authorization=str(DOC.parent/'AUTHORIZATION.json'),learner_device='cpu')
    p['max_physics']=sum(j['case']['step_cap'] for j in p['evaluation_jobs']+keys)
    # Exact cached features preserve all adapter gradients; no step is taken.
    os.sched_setaffinity(0,{14});torch.set_num_threads(1);T.setup(p);q,opt=L.make(p)
    started=time.time()
    def guard():assert time.time()<started+300
    old,field,keep=L.data(p,q,guard);variables=[v for v in q.parameters() if v.requires_grad]
    opt.zero_grad(set_to_none=True)
    for g in field+old:(L.pg(q,g,p)/len(field+old)).backward()
    keep_loss,_=L.M.retention(q,keep,False);(p['replay_weight']*keep_loss).backward()
    actual=torch.cat([v.grad.flatten() for v in variables]).numpy()
    with np.load(gate/'gradients.npz',allow_pickle=False) as z:
        expected=-(z['field_gradients'].sum(0)+z['old_gradients'].sum(0))/28+p['replay_weight']*z['nominal_gradient']
    error=float(np.abs(actual-expected).max());assert error<1e-10
    before=digest(L.optimizer_state(q,opt));shadow,shadowopt=L.make(p);assert digest(L.optimizer_state(shadow,shadowopt))==before
    assert not opt.state and not shadowopt.state and q.phash()==p['initial']['policy_hash']
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    res=S.Wait.Raw.resources();assert T.R.used(ROOT/'stage/runs')+p['max_bytes']<10*2**30
    pre=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',actual_initial_loss_gradient_matches_T670=True,max_gradient_error=error,
        fresh_Adam_no_steps=True,live_shadow_initial_state_exact=True,full_cached_causal_features_and_density=True,
        no_projection_because_initial_constraint_inactive=True,not_a_matched_algorithm_comparison=True,
        screen40_required21_and_exact24=True,max_physics=p['max_physics'],max_commands=437,new_native=0,new_optimizer=0,resources=res)
    save(DOC/'PREFLIGHT.json',pre)
    deps={Path(f) for f in p['inputs']}|set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC.parent/'AUTHORIZATION.json',DOC.parent/'OUTCOME_CONSTRAINT_PRIMARY_REVIEW.md',gate/'RESULTS.json',gate/'audit.json',gate/'gradients.npz',Path(p['initial']['path'])}
    for f,h in p['inputs'].items():assert sha(f)==h,f
    for r in read(p['old_summary'])['rows']:deps.add(Path(r['path']))
    for r in read(p['fresh_rounds'])[0]['training_rows']:deps.add(Path(r['path']))
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):deps.add(Path(f).resolve())
    p['inputs']={str(f):sha(f) for f in sorted(deps)}
    save(HERE/'protocol.json',p);save(DOC/'BUDGET.json',{k:p[k] for k in ['max_physics','max_commands','max_bytes','main_seconds','audit_seconds','Actor_steps','new_Value','absolute_learning_stop','absolute_native_stop','absolute_all_stop']})
    print(json.dumps(pre))


if __name__=='__main__':main()
