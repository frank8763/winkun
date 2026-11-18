"""One recovery qualification, one full-coverage fit, conditional complete407."""
from pathlib import Path
import ast,copy,fcntl,gzip,io,json,os,subprocess,sys,time,traceback
import t637_data as D
import t637_learning as M
import t637_native as N
B,L,P,E,torch,np=D.B,D.L,D.P,D.E,D.torch,D.np
ROOT=D.ROOT;HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t637_joint_coverage_recovery';DOC=ROOT/'stage/docs/day_20261118/t637'
read,save,sha,digest=D.read,D.save,D.sha,D.digest


def plan():return read(HERE/'protocol.json')
def used(f):return B.used(f)
def safe(r):return bool(r['complete'] and r['success'] and not r['collision'] and not r.get('field_boundary',False))


def setup(p):
    B.setup(p);P.frozen=L.frozen
    # All output redirects are process-local and only affect this new run.
    N.F.M.OUT=OUT


def verify(p):
    assert p['authorization_approved'] and time.time()<p['absolute_all_stop']
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    assert used(OUT)<=p['max_bytes'] and used(ROOT/'stage/runs')-used(OUT)+p['max_bytes']+131072<2**30
    assert D.B.shutil.disk_usage(ROOT).free>=40*2**30
    for f,h in p['inputs'].items():assert sha(f)==h,f


def gpu_guard():
    jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip().splitlines()
    assert all(int(v)==os.getpid() for v in jobs if v.strip()),'Other GPU compute exists; yield'


class Runtime:
    def __init__(self,p):
        self.p=p;self.start=time.time();self.deadline=min(self.start+p['main_seconds'],p['absolute_native_stop'])
        self.counts=N.G.R.M.mp.get_context('spawn').RawArray('q',4);self.commands=3;self.rows=[];self.pool=None;self.last=0.

    def guard(self):
        assert time.time()<self.deadline-5 and sum(self.counts)<=self.p['max_physics'] and self.commands+3<=self.p['max_commands']
        if time.time()-self.last>5:
            self.last=time.time();verify(self.p)
            save(OUT/'status.json',dict(state='ACTIVE',pid=os.getpid(),epoch=time.time(),physics=sum(self.counts),commands=self.commands,completed=len(self.rows)))

    def request(self,jobs,model,role,full=False,references=None):
        values=[]
        for off in range(0,len(jobs),3):
            self.guard();js=jobs[off:off+3]
            assert sum(self.counts)+sum(j['case']['step_cap'] for j in js)<=self.p['max_physics']
            assert self.commands+len(js)+3<=self.p['max_commands'];self.commands+=len(js)
            req=[]
            for i,j in enumerate(js):
                key=f'{len(self.rows)+i:04d}_{role}'
                req.append(dict(model=model['path'],job=copy.deepcopy(j),seed=None,query=False,record_key=key,
                   full_path=str(OUT/'episodes'/(key+'.npz')) if full else None))
            for i,((a,r),request) in enumerate(zip(self.pool.request(req),req)):
                fullpath=request['full_path']
                path=Path(fullpath) if full else OUT/'evaluation'/(request['record_key']+'.npz')
                if not full:
                    path.parent.mkdir(exist_ok=True);assert not path.exists()
                    E.N.write_npz(path,{k:a[k] for k in ['actions','native_counts','end_flags']},self.p['compact_bytes'])
                r.update(job_id=request['job']['id'],request=request,seed=None,role=role,path=str(path),sha256=sha(path),full=full,
                         loaded_model_sha256=model['sha256'],field_boundary=bool((np.abs(a['poses'][:,:2])>self.p['field_limit_m']).any()))
                assert r['policy_hash']==model['policy_hash']
                if references is not None:
                    old=references[off+i]
                    r.update(reference_path=old['path'],exact=True)
                self.rows.append(r)
                with gzip.open(OUT/'cases.jsonl.gz','ab') as f:f.write((json.dumps(r)+'\n').encode())
                # On any assertion the original raw worker return is already
                # durable. Save full failure arrays if this was compact-only.
                try:
                    N.inspect(self.p,a,r)
                    if references is not None:
                        assert r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
                except BaseException:
                    if not full:E.N.write_npz(OUT/f'contract_failure_{len(self.rows)}.npz',E.N.S.packed(a),self.p['full_bytes'])
                    raise
                values.append((a,r))
        return values


def event(who):
    def emit(stage,k):
        with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([who,stage,k])+'\n')
    return emit


def main():
    p=plan();verify(p);setup(p)
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'launch.json').exists();save(OUT/'launch.json',dict(pid=os.getpid(),epoch=time.time()))
    rt=Runtime(p);error=None;result=dict(qualified_recovery=False,update=None,proposal=None,precheck=None,full407=None,accepted=False,retained=p['initial'])
    obj=shadow=before=None
    try:
        # Fetch/validate all historical data before offline native sessions.
        data=D.load(p,rt.guard)
        manifest=[dict(id=g['id'],source_sha256=g['source_sha256'],history_sha256=g['history_sha256'],features_sha256=digest(g['f']),logits_sha256=digest(g['logits']),target_sha256=digest(g['target']),decisions=len(g['f'])) for g in data]
        save(OUT/'FULL384_MEMORY_MANIFEST.json',dict(state='ALL384_FULL_HISTORY_VERIFIED_IN_RAM',histories=manifest,restored_disk_cache_bytes=0))
        E.N.offline();E.L.install_stop_handlers()
        rt.pool=N.Pool(dict(p,out=str(OUT),native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        vv=rt.request([p['target_job']],p['initial'],'baseline',True);a,row=vv[0]
        B.compare(row,p['target_reference'])
        rt.request([p['target_job']],p['initial'],'exact_baseline',True,[row])
        mixed=dict(p['target_job'],gap_teacher_first=p['teacher_first'],source_path=row['path'])
        vv=rt.request([mixed],p['initial'],'qualified_teacher',True);teacher,teacher_row=vv[0]
        rt.request([mixed],p['initial'],'exact_teacher',True,[teacher_row])
        result['qualified_recovery']=safe(teacher_row);result['teacher_row']=teacher_row
        save(OUT/'RECOVERY_ADMISSION.json',dict(passed=result['qualified_recovery'],teacher='GAP',first=p['teacher_first'],row=teacher_row,not_student_success=True))
        if result['qualified_recovery']:
            assert time.time()+p['learning_seconds']<p['absolute_learning_stop']
            gpu_guard();os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
            gpu_lock=(ROOT/'stage/runs/.stage_gpu.lock').open('a+b');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            correction=M.correction(p,teacher,p['teacher_first'])
            keep=M.pack(data,'cuda');new={k:v.cuda() for k,v in correction.items()};obj=M.make(p);shadow=M.make(p)
            before=M.state(obj);assert digest(before)==digest(M.state(shadow));learning_start=time.time();last=[0.]
            def learn_guard():
                rt.guard();assert time.time()<min(learning_start+p['learning_seconds'],p['absolute_learning_stop'])-5
                if time.time()-last[0]>5:last[0]=time.time();gpu_guard()
            u=M.fit(obj,keep,new,p,learn_guard,event('live'));v=M.fit(shadow,keep,new,p,learn_guard,event('shadow'))
            assert u==v and digest(M.state(obj))==digest(M.state(shadow))
            mature=M.state(obj);mature_hash=digest(mature);M.load(obj,before);assert digest(M.state(obj))==digest(before)
            M.load(obj,mature);assert digest(M.state(obj))==mature_hash
            proposal=M.export(obj,p,OUT/'proposal.pt.gz')
            result.update(update=u,proposal=proposal,trained_state_hash=mature_hash)
            save(OUT/'LEARNING_COMPLETE.json',dict(update=u,proposal=proposal,Actor_Adam_rollback_exact=True))
            obj[0].cpu();shadow[0].cpu();del keep,new;torch.cuda.empty_cache()
            fcntl.flock(gpu_lock,fcntl.LOCK_UN);gpu_lock.close()
            pre=rt.request([p['target_job']],proposal,'new_target',True)
            ss=rt.request(p['sentinel_jobs'],proposal,'old_sentinels')
            result['precheck']=[r for a,r in pre+ss]
            if all(safe(r) for a,r in pre+ss):
                # A candidate cannot be retained on seven sentinels. Reserve
                # the entire joint evaluation before entering it.
                assert time.time()+p['full_evaluation_reserve_seconds']<rt.deadline
                vv=rt.request(p['evaluation_jobs'],proposal,'full407');rows=[r for a,r in vv]
                keys=[r for r in rows if r['job_id'] in p['key_replay_ids']]
                rt.request([r['request']['job'] for r in keys],proposal,'exact_final_keys',True,keys)
                ids={r['job_id'] for r in rows if safe(r)};lost=sorted(set(p['before_success_ids'])-ids);gained=sorted(ids-set(p['before_success_ids']))
                accepted=not lost and p['target_job']['id'] in ids and bool(gained)
                result.update(full407=dict(success=len(ids),success_ids=sorted(ids),lost_prior=lost,gained_prior=gained),accepted=accepted,
                    retained=proposal if accepted else p['initial'])
            if not result['accepted']:M.load(obj,before);M.load(shadow,before)
            result['retained_optimizer_state_sha256']=digest(M.state(obj))
    except BaseException:
        error=traceback.format_exc()
        if obj is not None and before is not None:M.load(obj,before)
        save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close()
        rt.commands+=3
        save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,physics=sum(rt.counts),commands=rt.commands,
             elapsed_s=time.time()-rt.start,result=result,new_Critic=0,new_RL=0,
             Actor_updates_started=sum(1 for z in (OUT/'optimizer_calls.jsonl').read_text().splitlines() if json.loads(z)[:2]==['live','BEFORE']) if (OUT/'optimizer_calls.jsonl').exists() else 0))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error


def audit():
    p=plan();verify(p);setup(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE'
    ids=[s['pid']]+[v for g in read(OUT/'worker_generations.json') for v in [g['parent'],*g['workers']]]
    assert all(not Path('/proc',str(v)).exists() for v in ids)
    rows=[json.loads(z) for z in gzip.decompress((OUT/'cases.jsonl.gz').read_bytes()).decode().splitlines()]
    assert len({r['path'] for r in rows})==len(rows)
    by={r['path']:r for r in rows}
    for r in rows:
        assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path'])
        if r['full']:N.inspect(p,a,r)
        else:assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
        if r.get('exact'):assert r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256']
    assert sum(r['physics'] for r in rows)==s['physics']<=p['max_physics']
    assert len(rows)+6==s['commands']<=p['max_commands']
    baseline=next(r for r in rows if r['role']=='baseline');B.compare(baseline,p['target_reference'])
    result=s['result'];history_updates=0
    if result['proposal'] is not None:
        deadline=time.time()+p['audit_seconds']-30
        guard=lambda:(_ for _ in ()).throw(TimeoutError('T637 audit bound')) if time.time()>=min(deadline,p['absolute_all_stop']-30) else None
        data=D.load(p,guard)
        manifest=[dict(id=g['id'],source_sha256=g['source_sha256'],history_sha256=g['history_sha256'],features_sha256=digest(g['f']),logits_sha256=digest(g['logits']),target_sha256=digest(g['target']),decisions=len(g['f'])) for g in data]
        assert manifest==read(OUT/'FULL384_MEMORY_MANIFEST.json')['histories']
        teacher=E.N.S.arrays(result['teacher_row']['path']);correction=M.correction(p,teacher,p['teacher_first'])
        gpu_guard();os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        gpu_lock=(ROOT/'stage/runs/.stage_gpu.lock').open('a+b');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        obj=M.make(p);before=M.state(obj);keep=M.pack(data,'cuda');new={k:v.cuda() for k,v in correction.items()}
        def guarded():guard();gpu_guard()
        u=M.fit(obj,keep,new,p,guarded,event('independent_history'))
        assert u==result['update'] and digest(M.state(obj))==result['trained_state_hash']
        assert obj[0].phash()==result['proposal']['policy_hash'] and sha(result['proposal']['path'])==result['proposal']['sha256']
        if not result['accepted']:M.load(obj,before)
        assert digest(M.state(obj))==result['retained_optimizer_state_sha256'];history_updates=u['Actor_Adam']
        calls=[json.loads(z) for z in (OUT/'optimizer_calls.jsonl').read_text().splitlines()]
        expected=[[stage,k] for k in range(p['fit_steps']) for stage in ['BEFORE','AFTER']]
        assert all([z[1:] for z in calls if z[0]==who]==expected for who in ['live','shadow','independent_history'])
        fcntl.flock(gpu_lock,fcntl.LOCK_UN);gpu_lock.close()
    if result['full407'] is not None:
        final=[r for r in rows if r['role']=='full407'];assert len(final)==407
        success={r['job_id'] for r in final if safe(r)};lost=sorted(set(p['before_success_ids'])-success);gained=sorted(success-set(p['before_success_ids']))
        assert result['full407']==dict(success=len(success),success_ids=sorted(success),lost_prior=lost,gained_prior=gained)
        assert result['accepted']==(not lost and p['target_job']['id'] in success and bool(gained))
    assert sha(p['initial']['path'])==p['initial']['sha256']
    save(OUT/'audit.json',dict(state='PASS',native=0,historical_Actor_Adam=history_updates,all_saved_full_history_and_compact_arrays_verified=True,
       independent_full384_features_and_Adam_reconstructed=result['proposal'] is not None,
       compact407_not_independent_full_pose_reconstruction=True,original_model_unchanged=True))


if __name__=='__main__':globals()[sys.argv[1]]()
