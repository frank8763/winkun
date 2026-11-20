"""Admit only unexecuted T663 work after the T664R helper read failure."""
from pathlib import Path
import ast,copy,gzip,importlib.util,json,os,sys,tempfile,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t665_remaining_verification';DOC=ROOT/'stage/docs/day_20261120/t665'
OLD=ROOT/'stage/runs/t664r_audit_throttle'
import t665_runtime as R
import t665_wait as W
from status_io import read_status,save
read,sha=R.read,R.sha


def main():
    assert not OUT.exists()
    old=read(OLD/'protocol.json');ledger=read(OLD/'LEDGER.json');sealed=read(OLD/'SEALED.json')
    assert sealed['all_scientific_processes_stopped']
    assert all(not Path('/proc',str(pid)).exists() for pid in sealed['scientific_pids'])
    assert not (OLD/'ppo_collision_s005').exists()
    assert read(OLD/'FINAL_REPORT_FINAL.json')['state']=='FAILED'
    assert 'JSONDecodeError' in (OLD/'affinity.log').read_text()
    # Execute the actual legacy read statement with an in-progress write.
    tree=ast.parse((ROOT/'stage/experiments/t664r_audit_throttle/affinity_guard.py').read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='status' for t in n.targets))
    with tempfile.TemporaryDirectory() as d:
        f=Path(d)/'status.json';f.write_text('')
        try:exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'actual_legacy_status_reader','exec'),dict(json=json,sf=f))
        except json.JSONDecodeError:pass
        else:raise AssertionError('Legacy failure not reproduced')
        def finish_write(delay):f.write_text('{"child_pid":1234}')
        assert read_status(f,sleep=finish_write)==dict(child_pid=1234)
        f.write_text('persistent invalid')
        try:read_status(f,attempts=2,sleep=lambda delay:None)
        except json.JSONDecodeError:pass
        else:raise AssertionError('Persistent corrupt state must fail closed')
        save(f,dict(child_pid=4321));assert read_status(f)==dict(child_pid=4321)
    pending=read(ROOT/'stage/docs/day_20261120/t664r/PENDING_FULL407.json')
    pending['source_cases']=str(OLD/'ppo_binary_s005_remaining/cases.jsonl.gz')
    p=copy.deepcopy(old);p.pop('registration_epoch',None)
    pri=old['prior_accounting'];agg=copy.deepcopy(pri)
    agg.update(prior_physics_upper=pri['prior_physics_upper']+ledger['physics'],
        prior_commands_upper=pri['prior_commands_upper']+ledger['commands'],
        prior_main_seconds_upper=pri['prior_main_seconds_upper']+ledger['main_seconds']+10,
        prior_audit_seconds=pri['prior_audit_seconds']+ledger['audit_seconds'],
        all_old_scientific_pids_gone=True,pids=sorted(set(pri['pids']+sealed['scientific_pids'])))
    for who,side in [('live','live'),('shadow','shadow'),('independent_history','independent')]:
        for kind,label in [('actor','Actor'),('value','Value')]:agg['prior_optimizer'][who][kind]+=ledger[label+'_'+side]
    olddirs=[ROOT/'stage/runs/t663_daytime_factorial_relay',ROOT/'stage/runs/t664_compact_day_relay',OLD]
    agg['prior_bytes']=sum(f.stat().st_size for d in olddirs for f in d.rglob('*') if f.is_file())
    p.update(id='T665_UNEXECUTED_CONDITIONAL407_AND_LAST_FIXED_ARM',out=str(OUT),authorization_approved=False,
        night_max_physics=240000000-agg['prior_physics_upper'],night_max_commands=6500-agg['prior_commands_upper'],
        night_main_seconds=int(28000-agg['prior_main_seconds_upper']),audit_total_seconds=int(3600-agg['prior_audit_seconds']),
        Actor_each_cap=min(48-v['actor'] for v in agg['prior_optimizer'].values()),
        Value_each_cap=min(192-v['value'] for v in agg['prior_optimizer'].values()),
        max_bytes=7*2**30-agg['prior_bytes'],prior_accounting=agg,
        first_start_epoch=time.time(),arm_configurations=[old['arm_configurations'][-1]],pending_completion=pending)
    p['arms']=['pending_full407','ppo_collision_s005'];p['new_optimizer_preflight']=0
    remaining=[j for j in p['evaluation_jobs'] if j['id'] in pending['required_remaining_ids']]
    upper=sum(j['case']['step_cap'] for j in remaining)
    assert upper==pending['remaining_physics_upper']==9482765 and len(remaining)==371
    p['completion_template']=dict(kind='completion',arm='pending_full407',actor_cap=0,value_cap=0,
        main_seconds=4200,audit_seconds=300,max_physics=upper,max_commands=377,
        child_bytes=32*2**20,max_bytes=32*2**20,compact_bytes=32768,full_bytes=4*2**20,fresh_bytes=4*2**20)
    R.torch.set_num_threads(1);os.sched_setaffinity(0,{14});resources=W.Raw.resources()
    q=dict(p,raw_sigma=pending['raw_sigma'],initial=pending['model'])
    R.setup(dict(q,out=str(OUT/'fixture'),arm='fixture'))
    screen=R.completion_inputs(q);assert len(screen)==36
    actor=R.M.L.frozen(q,pending['model']['path']);assert actor.phash()==pending['model']['policy_hash']
    # Canonical worker and request plumbing are inherited without native calls.
    R.setup(dict(q,out=str(OUT/'fixture'),arm='fixture'))
    assert R.R.N.Pool.__init__.__globals__['worker'] is R.N.worker
    R.verified_reuse(dict(p,**p['arm_configurations'][0]),read(ROOT/'stage/runs/t663_daytime_factorial_relay/ppo_binary_s005/INITIAL_MODEL.json'))
    assert R.R.used(ROOT/'stage/runs')+p['max_bytes']<p['global_bytes']
    assert time.time()+4200+5400+1800<p['absolute_native_stop']
    save(DOC/'BOUNDARY_RECONCILIATION.json',agg)
    auth={k:p[k] for k in ['night_max_physics','night_max_commands','night_main_seconds','audit_total_seconds','Actor_each_cap','Value_each_cap','max_bytes','global_bytes','absolute_all_stop']}
    auth.update(approved=True,new_optimizer_preflight=0,original_authorization=p['original_authorization'],
        scope='Only original unexecuted conditional407/last fixed collision arm/finite coverage; original totals and deadlines unchanged')
    save(DOC/'AUTHORIZATION.json',auth)
    result=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',new_native=0,new_optimizer=0,
        actual_legacy_empty_read_reproduced=True,transient_read_retry_pass=True,persistent_corruption_rejected=True,
        atomic_status_publication=True,all36_and_exact20_validated=True,candidate_loader_hash_exact=True,
        original_unexecuted_arm_same_policy64_reuse_verified=True,canonical_worker_verified=True,resources=resources)
    save(DOC/'PREFLIGHT.json',result);save(DOC/'ENTRY_PREFLIGHT.json',result)
    deps={Path(f) for f in old['inputs']}|set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'AUTHORIZATION.json',DOC/'PREFLIGHT.json',DOC/'ENTRY_PREFLIGHT.json',DOC/'BOUNDARY_RECONCILIATION.json'}
    for f,h in old['inputs'].items():assert sha(f)==h,f
    for d in [OLD,ROOT/'stage/runs/t663_daytime_factorial_relay/ppo_binary_s005']:
        deps.update(f for f in d.rglob('*') if f.is_file() and f.name not in ['driver.log','affinity.log','backup.log'])
    for m in list(sys.modules.values()):
        f=getattr(m,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):deps.add(Path(f).resolve())
    p['inputs']={str(f):sha(f) for f in sorted(deps)}
    save(HERE/'protocol.json',p)
    save(DOC/'BUDGET.json',{k:p[k] for k in auth if k in p}|dict(completion_template=p['completion_template'],rl_template=p['rl_template'],prior_accounting=agg))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
