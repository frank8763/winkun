"""One verified early recovery fit, full384 retention, same-candidate407 gate."""
from pathlib import Path
import fcntl,gzip,io,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t641_verified_recovery_learning';DOC=ROOT/'stage/docs/day_20261118/t641'
sys.path.insert(0,str(ROOT/'stage/experiments/t638_memory_ready_recovery'))
import t638_runtime as Mem
R=Mem.R;Mem.HERE=HERE;Mem.OUT=OUT;Mem.DOC=DOC
R.HERE=HERE;R.OUT=OUT;R.DOC=DOC;R.D.load=Mem.load
read,save,sha,digest,torch,np=R.read,R.save,R.sha,R.digest,R.torch,R.np


def manifest(data):
    return [dict(id=g['id'],source_sha256=g['source_sha256'],history_sha256=g['history_sha256'],
        features_sha256=digest(g['f']),logits_sha256=digest(g['logits']),target_sha256=digest(g['target']),decisions=len(g['f'])) for g in data]


def teacher_check(p):
    # Separate process: teacher loader patches cannot affect student inference.
    sys.path.insert(0,str(ROOT/'stage/experiments/t640_first_divergence_recovery'))
    import t640_runtime as Teacher
    pc=Teacher.B.read(p['teacher_configuration']);pc.update(read(p['teacher_protocol']))
    row=p['recovery_row'];assert sha(row['path'])==row['sha256']
    a=Teacher.B.R.B.R.E.N.S.arrays(row['path'])
    assert all(digest(v)==row['array_hashes'][key] for key,v in a.items())
    Teacher.B.N.P.install();Teacher.B.N.inspect(pc,[(a,row)])
    assert Teacher.B.R.safe(row) and int(a['student_takeover_first'])==5
    assert not a['ownership'][5:].any() and a['ownership'][:5].all()
    assert read(p['teacher_audit'])['state']=='PASS'
    return dict(state='PASS',source_sha256=row['sha256'],
        real_prefix_and_teacher_64plus32plus32_memory_reconstructed=True,success=True,new_native=0,new_optimizer=0)


def verify_teacher():
    p=R.plan();R.verify(p);save(OUT/'TEACHER_INPUT_AUDIT.json',teacher_check(p))


def preflight_teacher():
    assert not OUT.exists();print(json.dumps(teacher_check(R.plan())))


def main():
    p=R.plan();R.verify(p);R.setup(p);R.E.N.offline();R.E.L.install_stop_handlers()
    assert read(OUT/'TEACHER_INPUT_AUDIT.json')['state']=='PASS'
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'launch.json').exists();save(OUT/'launch.json',dict(pid=os.getpid(),epoch=time.time()))
    rt=R.Runtime(p);obj=shadow=before=None;error=None
    result=dict(qualified_recovery=True,teacher_row=p['recovery_row'],update=None,proposal=None,precheck=None,full407=None,
        accepted=False,retained=p['initial'],new_RL=0,new_Critic=0)
    try:
        data=R.D.load(p,rt.guard);save(OUT/'FULL384_MEMORY_MANIFEST.json',dict(state='ALL384_FULL_HISTORY_VERIFIED_IN_RAM',histories=manifest(data),restored_disk_cache_bytes=0))
        a=R.E.N.S.arrays(p['recovery_row']['path']);assert sha(p['recovery_row']['path'])==p['recovery_row']['sha256']
        correction=R.M.correction(p,a,p['teacher_first'])
        assert time.time()+p['learning_seconds']<p['absolute_learning_stop']
        R.gpu_guard();os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        gpu_lock=(ROOT/'stage/runs/.stage_gpu.lock').open('a+b');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        keep=R.M.pack(data,'cuda');new={k:v.cuda() for k,v in correction.items()}
        obj=R.M.make(p);shadow=R.M.make(p);before=R.M.state(obj);assert digest(before)==digest(R.M.state(shadow))
        learning_start=time.time();last=[0.]
        def learn_guard():
            rt.guard();assert time.time()<min(learning_start+p['learning_seconds'],p['absolute_learning_stop'])-5
            if time.time()-last[0]>5:last[0]=time.time();R.gpu_guard()
        u=R.M.fit(obj,keep,new,p,learn_guard,R.event('live'));v=R.M.fit(shadow,keep,new,p,learn_guard,R.event('shadow'))
        assert u==v and digest(R.M.state(obj))==digest(R.M.state(shadow))
        mature=R.M.state(obj);mature_hash=digest(mature)
        R.M.load(obj,before);assert digest(R.M.state(obj))==digest(before)
        R.M.load(obj,mature);assert digest(R.M.state(obj))==mature_hash
        proposal=R.M.export(obj,p,OUT/'proposal.pt.gz')
        result.update(update=u,proposal=proposal,trained_state_hash=mature_hash)
        save(OUT/'LEARNING_COMPLETE.json',dict(update=u,proposal=proposal,Actor_Adam_rollback_exact=True))
        obj[0].cpu();shadow[0].cpu();del keep,new;torch.cuda.empty_cache()
        fcntl.flock(gpu_lock,fcntl.LOCK_UN);gpu_lock.close()
        rt.pool=R.N.Pool(dict(p,out=str(OUT),native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        vv=rt.request([p['target_job']],proposal,'pre_target',True)
        vv+=rt.request(p['sentinel_jobs'],proposal,'pre_old6')
        pre=[r for a,r in vv];result['precheck']=pre
        if all(R.safe(r) for r in pre):
            assert time.time()+p['full_evaluation_reserve_seconds']<rt.deadline
            by={r['job_id']:r for r in pre};assert len(by)==7
            remaining=[j for j in p['evaluation_jobs'] if j['id'] not in by];assert len(remaining)==400
            rest=rt.request(remaining,proposal,'joint_rest400')
            by.update({r['job_id']:r for a,r in rest});rows=[by[j['id']] for j in p['evaluation_jobs']]
            assert len(rows)==407 and len({r['job_id'] for r in rows})==407
            keys=[by[key] for key in p['key_replay_ids']]
            rt.request([r['request']['job'] for r in keys],proposal,'exact_final7',True,keys)
            ids={r['job_id'] for r in rows if R.safe(r)}
            lost=sorted(set(p['before_success_ids'])-ids);gained=sorted(ids-set(p['before_success_ids']))
            accepted=not lost and p['target_job']['id'] in ids and bool(gained)
            result.update(full407=dict(success=len(ids),success_ids=sorted(ids),lost_prior=lost,gained_prior=gained),
                accepted=accepted,retained=proposal if accepted else p['initial'])
            save(OUT/'JOINT407_ROWS.json',dict(rows=rows,prechecks_reused=7,additional_unique_native_tasks=400,model=proposal))
        if not result['accepted']:R.M.load(obj,before);R.M.load(shadow,before)
        result['retained_optimizer_state_sha256']=digest(R.M.state(obj))
    except BaseException:
        error=traceback.format_exc()
        if obj is not None and before is not None:R.M.load(obj,before)
        save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close()
        rt.commands+=3
        save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,
          physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,result=result,new_Critic=0,new_RL=0,
          Actor_updates_started=sum(json.loads(z)[:2]==['live','BEFORE'] for z in (OUT/'optimizer_calls.jsonl').read_text().splitlines()) if (OUT/'optimizer_calls.jsonl').exists() else 0))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error


def audit():
    p=R.plan();R.verify(p);R.setup(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE'
    assert not Path('/proc',str(s['pid'])).exists()
    for g in read(OUT/'worker_generations.json'):
        assert all(not Path('/proc',str(v)).exists() for v in [g['parent'],*g['workers']])
    rows=[json.loads(z) for z in gzip.decompress((OUT/'cases.jsonl.gz').read_bytes()).decode().splitlines()]
    assert len({r['path'] for r in rows})==len(rows);by_path={r['path']:r for r in rows}
    for row in rows:
        assert sha(row['path'])==row['sha256'];a=R.E.N.S.arrays(row['path'])
        if row['full']:R.N.inspect(p,a,row)
        else:assert all(digest(v)==row['array_hashes'][k] for k,v in a.items())
        if row.get('exact'):
            ref=by_path[row['reference_path']];assert row['array_hashes']==ref['array_hashes'] and row['integration_sha256']==ref['integration_sha256']
    assert sum(r['physics'] for r in rows)==s['physics']<=p['max_physics']
    assert len(rows)+6==s['commands']<=p['max_commands']
    data=R.D.load(p,lambda:None);assert manifest(data)==read(OUT/'FULL384_MEMORY_MANIFEST.json')['histories']
    teacher=R.E.N.S.arrays(p['recovery_row']['path']);correction=R.M.correction(p,teacher,p['teacher_first'])
    R.gpu_guard();os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    gpu_lock=(ROOT/'stage/runs/.stage_gpu.lock').open('a+b');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    obj=R.M.make(p);before=R.M.state(obj);keep=R.M.pack(data,'cuda');new={k:v.cuda() for k,v in correction.items()}
    deadline=min(time.time()+p['audit_seconds']-30,p['absolute_all_stop']-60)
    def guard():assert time.time()<deadline;R.gpu_guard()
    u=R.M.fit(obj,keep,new,p,guard,R.event('independent_history'));result=s['result']
    assert u==result['update'] and digest(R.M.state(obj))==result['trained_state_hash']
    assert obj[0].phash()==result['proposal']['policy_hash'] and sha(result['proposal']['path'])==result['proposal']['sha256']
    if not result['accepted']:R.M.load(obj,before)
    assert digest(R.M.state(obj))==result['retained_optimizer_state_sha256']
    calls=[json.loads(z) for z in (OUT/'optimizer_calls.jsonl').read_text().splitlines()]
    expected=[[stage,k] for k in range(p['fit_steps']) for stage in ['BEFORE','AFTER']]
    assert all([z[1:] for z in calls if z[0]==who]==expected for who in ['live','shadow','independent_history'])
    fcntl.flock(gpu_lock,fcntl.LOCK_UN);gpu_lock.close()
    if result['full407'] is not None:
        actual=[r for r in rows if r['role'] in ['pre_target','pre_old6','joint_rest400']]
        assert len(actual)==407 and len({r['job_id'] for r in actual})==407
        ordered=read(OUT/'JOINT407_ROWS.json')['rows'];byid={r['job_id']:r for r in actual}
        assert ordered==[byid[j['id']] for j in p['evaluation_jobs']]
        assert all(r['loaded_model_sha256']==result['proposal']['sha256'] for r in ordered)
        ids={r['job_id'] for r in ordered if R.safe(r)};lost=sorted(set(p['before_success_ids'])-ids);gained=sorted(ids-set(p['before_success_ids']))
        assert result['full407']==dict(success=len(ids),success_ids=sorted(ids),lost_prior=lost,gained_prior=gained)
        assert result['accepted']==(not lost and p['target_job']['id'] in ids and bool(gained))
    assert sha(p['initial']['path'])==p['initial']['sha256']
    save(OUT/'audit.json',dict(state='PASS',new_native=0,historical_Actor_Adam=u['Actor_Adam'],
       independent_full384_history_features_and_Adam_rebuilt=True,all_saved_full_history_and_compact_arrays_verified=True,
       same_candidate407_prechecks_counted_once=True,compact407_not_full_pose_reconstruction=True,original_T501_unchanged=True))


if __name__=='__main__':
    def no_network(*args,**kwargs):raise RuntimeError('Science uses verified RAM only')
    R.D.remote_bytes=no_network;R.E.N.offline()
    globals()[sys.argv[1]]()
