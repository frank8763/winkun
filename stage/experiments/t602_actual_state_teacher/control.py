"""Zero-native preparation, fixed posthoc entry rule and budget reallocation."""
from pathlib import Path
import sys,copy,time,datetime,ast
import t602_runtime as R
ROOT,HERE,OUT,DOC=R.ROOT,R.HERE,R.OUT,R.DOC
L=R.L;read,save,sha,digest,torch,np=R.read,R.save,R.sha,R.digest,R.torch,R.np


def prepare():
    assert not OUT.exists() and not (DOC/'PREFLIGHT.json').exists();DOC.mkdir(parents=True,exist_ok=True)
    old=ROOT/'stage/runs/t600_task_risk_pair';p=copy.deepcopy(read(old/'protocol.json'));s=read(old/'summary.json');assert s['state']=='COMPLETE' and read(old/'audit.json')['state']=='PASS'
    source=p['joint_recovery_row'];a=R.E.N.S.arrays(source['path']);teacher_path=source['request']['model']
    teacher=L.frozen(p,teacher_path);teacher_model=dict(path=teacher_path,sha256=sha(teacher_path),policy_hash=teacher.phash())
    assert teacher_model['sha256']==source['model_sha256']
    jobs=[];descriptive=[]
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for arm in s['result']['arms']:
        row=arm['recovery'];b=R.E.N.S.arrays(row['path']);assert sha(row['path'])==row['sha256']
        n=min(len(a['states']),len(b['states']));k=int(b['student_takeover_first']);xy=np.linalg.norm(a['poses'][:n,:2]-b['poses'][:n,:2],axis=1)
        found=next(i for i in range(k,n-4) if (xy[i:i+5]>=.10).all());first=found-1
        t=np.r_[0,np.cumsum(b['native_counts'])[:-1]]*.002
        job=copy.deepcopy(source['request']['job']);job.update(source_path=row['path'],student_takeover_first=first)
        jobs.append(job)
        q=L.frozen(p,arm['proposal']['path']);h=digest(q.state_dict())
        with torch.no_grad():pred=q.sequence(torch.tensor(a['states'])).clamp(-1,1).float().numpy()
        assert digest(q.state_dict())==h
        windows=[]
        for lo,hi in [(15,20),(20,25),(25,30),(30,35),(35,40),(40,46)]:
            mask=(t[:n]>=lo)&(t[:n]<hi)
            windows.append(dict(absolute_sim_seconds=[lo,hi],teacher_on_success_drive_steer=a['actions'][:n][mask].mean(0).tolist(),student_on_success_drive_steer=pred[:n][mask].mean(0).tolist(),student_live_drive_steer=b['actions'][:n][mask].mean(0).tolist(),max_xy_difference_m=float(xy[mask].max())))
        descriptive.append(dict(method=arm['method'],entry_decision=first,entry_clock_s=float(t[first]),position_difference_before_m=float(xy[first]),
            selection_rule='POSTHOC first >=.10m for5consecutive decisions, choose one decision before; no outcome-based threshold search',windows=windows))
    p.update(id='T602_FIXED_TEACHER_FROM_ACTUAL_STUDENT_ERROR_STATES',out=str(OUT),teacher_model=teacher_model,qualification_jobs=jobs,inputs={},protected_files={},authorization_approved=False,
        main_seconds=600,learning_seconds=0,audit_seconds=300,transport_seconds=180,source_seconds=120,
        absolute_learning_stop=datetime.datetime.fromisoformat('2026-11-16T20:20:00+08:00').timestamp(),
        absolute_native_stop=datetime.datetime.fromisoformat('2026-11-16T20:40:00+08:00').timestamp(),
        absolute_all_stop=datetime.datetime.fromisoformat('2026-11-16T21:00:00+08:00').timestamp(),
        max_commands=8,max_physics=2*sum(j['case']['step_cap'] for j in jobs),max_bytes=int(5.5*2**20),child_bytes=int(5.5*2**20),
        new_Actor_Adam_cap=0,shadow_Adam_cap=0,audit_discarded_Adam_cap=0,preflight_discarded_Adam_cap=0,new_RL=0,new_Critic=0,
        scope='Fixed T598 learned teacher only; two T600 actual failed-state histories prefix-replayed, original reset/goal/history/native clock not reset; teacher carry recomputed from actual history. Fullsuffix safeGOAL required qualification;2full+2compactexact, no newstudent learning/onpolicy/407. No alternative teacher/seed/entry grid.')
    upper=2*p['full_bytes']+2*p['compact_bytes']+1024*1024;glob=R.Legacy.R.used(ROOT/'stage/runs')+p['max_bytes']+131072
    assert upper<=p['max_bytes'] and glob<2**30 and p['max_physics']==256448
    assert s['physics']+p['max_physics']<=5246336 and s['commands']+8<=114
    total_used=R.Legacy.R.used(old)+R.Legacy.R.used(ROOT/'stage/runs/t601_takeover_distribution_diagnosis')+p['max_bytes']
    assert total_used<=10*2**20
    schedule=R.schedule_probe(p)
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    save(DOC/'ENTRY_ANALYSIS_POSTHOC.json',descriptive);save(DOC/'SCHEDULE_PREFLIGHT.json',schedule)
    save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',new_native=0,new_gradients=0,new_optimizer=0,teacher_sha_matches_qualified_original=True,entry_rules_posthoc_no_outcome_search=True,actual_runtime_counter_pass=True,own_upper=upper,global_worst=glob,combined_T600_T601_newrun_worst=total_used,original_combined_limit=10*2**20))
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
    for f in [*HERE.glob('*.py'),DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC/'ENTRY_ANALYSIS_POSTHOC.json',DOC/'SCHEDULE_PREFLIGHT.json',old/'summary.json',old/'audit.json',Path(teacher_path),Path(source['path'])]+[Path(j['source_path']) for j in jobs]:p['inputs'][str(f)]=sha(f)
    for j in jobs:p['inputs'][j['exact_snapshot']]=sha(j['exact_snapshot'])
    m=p['base_model']
    while m:p['inputs'][m['path']]=m['sha256'];p['protected_files'][m['path']]=m['sha256'];m=m.get('parent')
    p['protected_files'][teacher_path]=sha(teacher_path)
    save(DOC/'PROPOSAL.json',p);save(DOC/'BUDGET.json',dict(state='PREPARED_NOT_REGISTERED_NOT_LAUNCHED',max_physics=256448,max_commands=8,own_bytes=p['max_bytes'],own_upper=upper,global_worst=glob,combined_T600_T601_newrun_worst=total_used,combined_original10MiB_limit=10*2**20,new_optimizer=0,new_native_in_preparation=0,combined_physics_if_max=s['physics']+256448,combined_commands_if_max=s['commands']+8,main_seconds=600,audit_seconds=300,transport_seconds=180,source_seconds=120,total_admission_seconds=1230,all_stop='21:00',scope_reallocation='PENDING'))
    print('T602_PREPARED_ZERO_NATIVE_ZERO_OPTIMIZER',glob,descriptive[0]['entry_clock_s'],descriptive[1]['entry_clock_s'],flush=True)


def register():
    assert not OUT.exists();p=read(DOC/'PROPOSAL.json');a=read(DOC/'AUTHORIZATION.json')
    assert a['approved'] and a['max_physics']==256448 and a['max_commands']==8 and a['all_stop_epoch']==p['absolute_all_stop']
    assert time.time()+1230<p['absolute_all_stop'] and time.time()+600<p['absolute_native_stop']
    assert R.Legacy.R.used(ROOT/'stage/runs')+p['max_bytes']+131072<2**30
    for f,h in p['inputs'].items():assert sha(f)==h,f
    p['authorization_approved']=True;p['inputs'][str(DOC/'AUTHORIZATION.json')]=sha(DOC/'AUTHORIZATION.json');OUT.mkdir()
    blob=__import__('json').dumps(p,separators=(',',':'),allow_nan=False)+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob)
    print('T602_REGISTERED_NOT_LAUNCHED',flush=True)


if __name__=='__main__':globals()[sys.argv[1]]()
