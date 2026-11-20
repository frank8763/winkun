"""Fixed old-counterexample outcome sampling, no optimization or seed sweep."""
from pathlib import Path
import ast,copy,os,sys,time,json
import t669_runtime as S
from status_io import save
ROOT,HERE,OUT,DOC=S.ROOT,S.HERE,S.OUT,S.DOC
T,read,sha=S.T,S.read,S.sha


def main():
    assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True)
    src=ROOT/'stage/runs/t668_observed_departure_recovery'
    assert read(src/'audit.json')['state']=='PASS' and read(src/'FINAL_REPORT.json')['state']=='COMPLETE'
    assert len(read(src/'summary.json')['qualified_recovery_ids'])==2
    for pid in read(src/'SEALED.json')['scientific_pids']:assert not Path('/proc',str(pid)).exists()
    assert not Path('/proc',str(read(src/'DRIVER_STARTED.json')['pid'])).exists()
    old=read(ROOT/'stage/runs/t665_remaining_verification/protocol.json');p=copy.deepcopy(old);p.pop('registration_epoch',None)
    keys=['dev_task6_20','dev_task6_21','dev_task6_23','hold_task6_06'];jobs={j['id']:j for j in p['evaluation_jobs']}
    selected=[jobs[k] for k in keys];assert all(k not in [j['id'] for j in old['training_jobs']] for k in keys)
    now=time.time();auth=read(DOC.parent/'AUTHORIZATION.json')
    p.update(id='T669_TRUE_OLD_TASK_ONPOLICY_SUPPORT',authorization_approved=False,out=str(OUT),arm='old_onpolicy_signal',
        kind='data_collection',raw_sigma=[.05,.05],old_signal_jobs=selected,signal_seed=66920261120,
        max_physics=1493376,max_commands=70,max_bytes=128*2**20,child_bytes=128*2**20,main_seconds=1800,audit_seconds=600,
        absolute_learning_stop=min(now+1800,auth['absolute_learning_stop']),absolute_native_stop=min(now+1800,auth['absolute_native_stop']),
        absolute_all_stop=min(now+3600,auth['absolute_all_stop']),finalization_seconds=60,learner_device='cpu',
        new_Actor=0,new_Value=0,new_optimizer=0,weekend_authorization=str(DOC.parent/'AUTHORIZATION.json'))
    assert 16*sum(j['case']['step_cap'] for j in selected)==p['max_physics']
    os.sched_setaffinity(0,{14});T.torch.set_num_threads(1);T.setup(p)
    blob,phash,weights=T.M.variant_payload(p)
    q=T.M.L.Actor(p).eval().requires_grad_(False);q.load_state_dict(T.M.P.checkpoint(p['initial']['path'])['joint_state']['actor'])
    assert q.phash()==phash and T.digest(q.state_dict())==weights
    assert T.R.N.Pool.__init__.__globals__['worker'] is T.N.worker
    assert len({tuple([p['signal_seed'],i]) for i in range(32)})==32
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    res=S.Wait.Raw.resources();assert T.R.used(ROOT/'stage/runs')+p['max_bytes']<10*2**30
    pre=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',previous_recovery_2of4_and_audit_checked=True,
        all_four_retreat_ids_absent_from_previous_fresh_training=True,sigma05_variant_weights_exact_T501=True,
        variant_policy_hash=phash,variant_bytes=len(blob),canonical_worker_installed=True,
        fixed32_unique_seed_units=True,new_native=0,new_optimizer=0,resources=res)
    save(DOC/'PREFLIGHT.json',pre)
    deps={Path(f) for f in old['inputs']}
    for f,h in old['inputs'].items():assert sha(f)==h,f
    deps|=set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC.parent/'AUTHORIZATION.json',src/'summary.json',src/'audit.json',DOC.parent/'t668/FINDINGS.md'}
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):deps.add(Path(f).resolve())
    p['inputs']={str(f):sha(f) for f in sorted(deps)}
    save(HERE/'protocol.json',p);save(DOC/'BUDGET.json',{k:p[k] for k in ['max_physics','max_commands','max_bytes','main_seconds','audit_seconds','absolute_native_stop','absolute_all_stop','new_optimizer']})
    print(json.dumps(pre))


if __name__=='__main__':main()
