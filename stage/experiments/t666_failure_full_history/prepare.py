"""Zero-step path, provenance and finite-budget admission for T666."""
from pathlib import Path
import ast,copy,gzip,json,lzma,os,sys,time
import t666_runtime as S
from status_io import save
ROOT,HERE,OUT,DOC=S.ROOT,S.HERE,S.OUT,S.DOC
T,read,sha=S.T,S.read,S.sha


def main():
    assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True)
    oldout=ROOT/'stage/runs/t665_remaining_verification';old=read(oldout/'protocol.json')
    final=read(oldout/'FINAL_REPORT_FINAL.json');sealed=read(oldout/'SEALED.json')
    assert final['error'] is None and not final['audit_pending'] and final['all_scientific_processes_stopped']
    assert all(not Path('/proc',str(pid)).exists() for pid in sealed['scientific_pids'])
    assert not Path('/proc',str(read(oldout/'DRIVER_STARTED.json')['pid'])).exists()
    auth=read(DOC.parent/'AUTHORIZATION.json');assert auth['approved']
    p=copy.deepcopy(old);p.pop('registration_epoch',None);p.pop('_root_sha256',None)
    now=time.time();p.update(id='T666_FIXED_FOUR_FORGETTING_FULL_HISTORY',out=str(OUT),arm='failure_full_history',
        authorization_approved=False,kind='diagnostic',max_physics=373344,max_commands=22,max_bytes=128*2**20,child_bytes=128*2**20,
        main_seconds=1800,audit_seconds=600,backup_seconds=600,source_seconds=240,finalization_seconds=60,
        absolute_learning_stop=min(now+1800,auth['absolute_learning_stop']),
        absolute_native_stop=min(now+1800,auth['absolute_native_stop']),
        absolute_all_stop=min(now+3600,auth['absolute_all_stop']),learner_device='cpu',
        new_Actor=0,new_Value=0,new_optimizer=0,weekend_authorization=str(DOC.parent/'AUTHORIZATION.json'))
    base=old['initial'];baseline=[json.loads(x) for x in lzma.decompress((ROOT/'stage/runs/t501_joint_command_repair/joint/cases.jsonl.t595.xz').read_bytes()).decode().splitlines()]
    low=read(ROOT/'stage/runs/t664r_audit_throttle/ppo_binary_s005_remaining/ROUNDS.json')[0]['proposal']
    coll=read(oldout/'ppo_collision_s005/ROUNDS.json')[0]['proposal']
    lowrows=read(oldout/'pending_full407/summary.json')['rows']
    collrows=[json.loads(x) for x in gzip.decompress((oldout/'ppo_collision_s005/cases.jsonl.gz').read_bytes()).decode().splitlines()]
    pairs=[];jobs={j['id']:j for j in p['evaluation_jobs']};contract=read(p['reference_contract']);refs={r['task_id']:r for r in contract['reference_rows']}
    for key in ['dev_task6_20','dev_task6_21','dev_task6_23','hold_task6_06']:
        islow=key in ['dev_task6_20','dev_task6_21'];model=low if islow else coll
        candidate=next(r for r in (lowrows if islow else collrows) if r['job_id']==key and (islow or r['role'].startswith('rest407_00_')))
        ref=copy.deepcopy(next(r for r in baseline if r['job_id']==key and r['role']=='AUTONOMOUS_JOINT_COMMAND_FULL383'))
        source=refs[key];ref.update(path=source['source_path'],sha256=source['sha256'])
        assert ref['success'] and not ref['collision'] and not candidate['success']
        assert sha(ref['path'])==ref['sha256'] and sha(candidate['path'])==candidate['sha256']
        assert candidate['request']['job']==jobs[key]
        assert candidate['policy_hash']==model['policy_hash'] and candidate['loaded_model_sha256']==model['sha256']
        pairs.append(dict(id=key,job=jobs[key],reference=base,candidate=model,reference_row=ref,candidate_row=candidate))
    p['pairs']=pairs
    p['model_sigmas']={m['path']:([.2,.2] if which=='reference' else [.05,.05]) for x in pairs for which in ['reference','candidate'] for m in [x[which]]}
    assert 4*sum(x['job']['case']['step_cap'] for x in pairs)==p['max_physics']
    # The exact native loader accepts each model with its own exploration
    # metadata; deterministic mean behavior is unchanged by sigma metadata.
    os.sched_setaffinity(0,{14});T.torch.set_num_threads(1);S.setup(p)
    checked=[]
    for pair in pairs:
        for which in ['reference','candidate']:
            model=pair[which];assert sha(model['path'])==model['sha256']
            child=dict(p,raw_sigma=[.2,.2] if which=='reference' else [.05,.05])
            q=T.M.L.frozen(child,model['path']);assert q.phash()==model['policy_hash']
            if which=='reference':
                r=pair['reference_row'];T.N.inspect(child,T.R.E.N.S.arrays(r['path']),r)
            checked.append(dict(id=pair['id'],which=which,hash=model['policy_hash']))
    # T501 and sigma=.05 models share identical inference implementation.
    # Each request has sigma only in frozen policy metadata, so workers need
    # per-model settings (see the canonical worker wrapper in native.py).
    res=S.Wait.Raw.resources();assert T.R.used(ROOT/'stage/runs')+p['max_bytes']<10*2**30
    result=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',new_native=0,new_optimizer=0,
        all_old_science_and_driver_exited=True,eight_model_loaders_checked=checked,
        four_reference_full_histories_checked=True,four_candidate_compact_provenance_checked=True,
        sum_all16_horizons=373344,commands_with_start_stop=22,resources=res)
    save(DOC/'PREFLIGHT.json',result)
    dependencies={Path(f) for f in old['inputs'] if not str(f).endswith('stage/docs/RESEARCH_STATUS.md')}
    for f,h in old['inputs'].items():assert sha(f)==h,f
    for pair in pairs:
        for which in ['reference','candidate']:
            dependencies.add(Path(pair[which]['path']));dependencies.add(Path(pair[which+'_row']['path']))
    dependencies|=set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC.parent/'AUTHORIZATION.json'}
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):dependencies.add(Path(f).resolve())
    p['inputs']={str(f):sha(f) for f in sorted(dependencies)}
    save(HERE/'protocol.json',p)
    save(DOC/'BUDGET.json',{k:p[k] for k in ['max_physics','max_commands','max_bytes','main_seconds','audit_seconds','backup_seconds','absolute_learning_stop','absolute_native_stop','absolute_all_stop','new_Actor','new_Value','new_optimizer']})
    print(json.dumps(result))


if __name__=='__main__':main()
