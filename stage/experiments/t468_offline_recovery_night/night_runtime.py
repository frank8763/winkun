"""Bounded offline night rounds; fixed training-only teachers, pure Actor tests."""
from pathlib import Path
import contextlib,copy,importlib.util,json,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t468_offline_recovery_night';DOC=ROOT/'stage/docs/overnight_20261110'
sys.path.insert(0,str(HERE))
import night_learning as N
J,H,M,P,E,R,torch,np=N.J,N.H,N.M,N.P,N.E,N.R,N.torch,N.np
read,save,sha,digest=N.read,N.save,N.sha,N.digest
spec=importlib.util.spec_from_file_location('stage468_pinned467',ROOT/'stage/experiments/t467_actual_occupancy_audit/run.py')
U=importlib.util.module_from_spec(spec);spec.loader.exec_module(U)
ACTIVE=OUT
IDS=['dev_task6_00','dev_task6_21','dev_task6_17','dev_task6_22','dev_task6_26','wide117_80_dev6_LEFT_13']


def rootplan():return read(OUT/'protocol.json')
def verify(p,full=False):
    root=rootplan();assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert M.used(OUT)<=root['max_bytes'] and M.used()<=2**30
    assert M.used()-M.used(OUT)+root['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    if ACTIVE!=OUT:assert M.used(ACTIVE)<=p['max_bytes']
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f

def setup(folder):
    global ACTIVE
    ACTIVE=Path(folder);M.OUT,M.verify,M.Pool=ACTIVE,verify,Pool;M.setup_runtime()


def teacher(p,job):
    if job['id']=='dev_task6_00':
        q=E.safe_initial(E.N.science(p),p).eval().requires_grad_(False)
        return q,'safeT255reset',digest(q.state_dict())
    assert job['id']=='dev_task6_21'
    q=N.frozen(p,p['recovery_teacher_model']['path'])
    return q,'T461_actual_history_carry',p['recovery_teacher_model']['sha256']

def teacher_command(q,kind,x,hidden):
    with torch.no_grad():
        if kind=='safeT255reset':return q.base_actions(torch.tensor(x[None])).float(),hidden
        command,_,hidden,_=q.step(torch.tensor(x[None]),hidden)
        return command,hidden


def worker(pipe,index,p,counts):
    E.N.offline();R.C.ban_learning();os.sched_setaffinity(0,{index});torch.set_num_threads(1);torch.use_deterministic_algorithms(True);E.set_out(p['out'])
    folder=Path(p['out'])/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True)
    log=(folder/'console.log').open('a');stdout,stderr=sys.stdout,sys.stderr;s=w=None
    def guard(physical=False):
        E.L.check_stop();assert time.time()<min(p['native_stop_epoch'],p['absolute_native_stop'])-5 and sum(counts)<p['max_physics']
    def charge():counts[index+1]+=1;assert sum(counts)<=p['max_physics']
    try:
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            module,native,create,w,_=E.N.S.P.C.B.setup(folder/'native',E.N.science(p));original=copy.deepcopy(module.ACTIVE);E.L.install_stop_handlers()
            pipe.send(dict(state='READY',pid=os.getpid(),cpu=index))
            while True:
                req=pipe.recv()
                if req is None:break
                assert req.get('seed') is None
                q=N.frozen(p,req['model']);j=req['job'];module.ACTIVE=copy.deepcopy(original)
                module.ACTIVE['circle_shift_x_m']=j.get('circle_shift_x_m',original['circle_shift_x_m'])
                before=counts[index+1];s=E.L.Session(j,native,create,w,guard,charge,q);s.hidden=torch.zeros((1,1,64),dtype=torch.float64)
                if req.get('cancel_probe'):
                    def cancel(physical=False):
                        if physical:raise E.L.StageStop('T468 zero-native cancel')
                    s.guard=cancel
                    try:s.step();raise AssertionError('Cancellation missing')
                    except E.L.StageStop:pass
                    assert not s.rows and s.steps==0;s.close();s=None;pipe.send(dict(state='CANCEL_PASS',physics=0));continue
                query=req.get('query',False);labels=[];eh=None;kind=teacher_hash=None
                first=int(j.get('teacher_first',p.get('teacher_first',20)))
                if query:expert,kind,teacher_hash=teacher(p,j)
                while not s.done():
                    if query:
                        x=s.packet().copy();command,eh=teacher_command(expert,kind,x,eh);labels.append(command[0].numpy().copy())
                        if len(s.rows)>=first:s.step(command[0].numpy(),owner=1,actor_active=True)
                        else:s.step(owner=0)
                        assert np.array_equal(s.rows[-1][0],x)
                    else:s.step(owner=0)
                a=E.payload(s)
                if query:a['teacher_actions']=np.asarray(labels,np.float32)
                r=s.result();r.update(policy_hash=q.phash(),physics=counts[index+1]-before,array_digest=digest(a),array_hashes={k:digest(v) for k,v in a.items()},
                    integration_sha256=digest(a['final_native_integration']),pid=os.getpid(),cpu=index,teacher=query,teacher_first=first if query else None,
                    teacher_kind=kind,teacher_source_SHA256=teacher_hash,teacher_hidden_burns_actual_history_from_h0=kind=='T461_actual_history_carry')
                if req.get('full_path'):
                    path=Path(req['full_path']);path.parent.mkdir(parents=True,exist_ok=True);blob=E.N.R.blob(E.N.S.packed(a));assert len(blob)<=p['full_bytes']
                    if path.exists():assert path.read_bytes()==blob
                    else:path.write_bytes(blob)
                s.close();s=None;pipe.send((a,r))
    except BaseException:
        save(folder/'failure.json',dict(error=traceback.format_exc(),physics=counts[index+1]))
        if s is not None and s.rows:
            try:E.N.write_npz(folder/'partial.npz',E.N.S.packed(E.payload(s)),p['full_bytes'])
            except BaseException:save(folder/'partial_save_error.json',dict(error=traceback.format_exc()))
        try:pipe.send(dict(state='FAILED',error=traceback.format_exc()))
        except BaseException:pass
        raise
    finally:
        if s:s.close()
        if w:w.close()
        sys.stdout,sys.stderr=stdout,stderr;log.close();pipe.close()


class Pool(M.Pool):
    def __init__(self,p,counts,guard):
        self.guard,self.children,self.pipes=guard,[],[];ctx=R.M.mp.get_context('spawn')
        try:
            for i in range(8):
                a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
            self.ready=[self.receive(i) for i in range(8)];assert all(v['state']=='READY' for v in self.ready)
            f=Path(p['out'])/'worker_generations.json';gg=read(f) if f.exists() else [];gg.append(dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time()));save(f,gg)
        except BaseException:self.close();raise


def arrays(row):
    assert sha(row['path'])==row['sha256'];a=E.N.S.arrays(row['path'])
    assert row['full'] and digest(a)==row['array_digest'] and {k:digest(v) for k,v in a.items()}==row['array_hashes']
    return a

def prefix_exact(query,source,first):
    a,b=arrays(query),arrays(source)
    assert len(a['states'])>first and len(b['states'])>first
    for k in ['states','poses','history_poses','history_goals','history_times','hidden_before']:assert np.array_equal(a[k][:first+1],b[k][:first+1]),k
    for k in ['actions','native_counts']:assert np.array_equal(a[k][:first],b[k][:first]),k
    assert np.array_equal(a['initial_native_integration'],b['initial_native_integration'])


def mainround(mode,index):
    assert mode in N.MODES and int(index) in [0,1]
    folder=OUT/f'{mode}_{int(index)}';p=read(folder/'protocol.json');root=rootplan();setup(folder);verify(p,True)
    assert not (folder/'summary.json').exists();E.N.offline();E.L.install_stop_handlers();rt=M.Runtime(p)
    baseguard=rt.guard
    def guard(learning=False):
        assert time.time()<(p['absolute_learning_stop'] if learning else p['absolute_native_stop'])-5
        baseguard(learning)
    rt.guard=guard
    status,error='FAILED',None;source=[];queries=[];reused=[];completed=0;candidate=proposal=None;obj=shadow=initial=None;coverage=None
    try:
        rt.pool=Pool(dict(p,native_stop_epoch=min(rt.deadline,p['absolute_native_stop'])),rt.counts,rt.guard)
        seed=root['seed_occupancy_rows'];same=p['learning_initial']['policy_hash']==seed[0]['policy_hash']
        jobs={j['id']:j for j in p['main_jobs']+p['old275_jobs']}
        if same:
            source=seed
            for r in source:arrays(r);assert r['policy_hash']==p['learning_initial']['policy_hash']
        else:
            values=rt.request([jobs[k] for k in IDS],p['learning_initial'],'ACTUAL_STUDENT6',full=True);source=[r for a,r in values]
            rt.request([jobs[k] for k in IDS],p['learning_initial'],'ACTUAL_STUDENT6_EXACT',full=True,references=source)
        stats,_=U.label_statistics(p,source);onsets={r['job_id']:r['onset_candidate_detection_decision'] for r in stats}
        source_byid={r['job_id']:r for r in source};valid=set(p['current_success_ids']);query_jobs=[];selection=[]
        for key in IDS[:2]:
            if key in valid:continue
            first=onsets[key] if mode=='occupancy' and onsets[key] is not None else 20
            job=dict(jobs[key],teacher_first=first);old=next((r for r in root['validated_initial_recoveries'] if r['job_id']==key),None)
            reuse=same and first==20 and old is not None and old['policy_hash']==p['learning_initial']['policy_hash']
            selection.append(dict(job_id=key,teacher_first=first,onset=onsets[key],fallback=mode=='occupancy' and onsets[key] is None,reused=reuse))
            if reuse:arrays(old);queries.append(old);reused.append(old)
            else:query_jobs.append(job)
        if query_jobs:
            values=rt.request(query_jobs,p['learning_initial'],'RECOVERY_QUERY',full=True,query=True);new=[r for a,r in values]
            rt.request(query_jobs,p['learning_initial'],'RECOVERY_EXACT',full=True,references=new,query=True)
            for r in new:prefix_exact(r,source_byid[r['job_id']],r['teacher_first'])
            queries+=new
        assert 1<=len(queries)<=2
        coverage=dict(all_safe=all(r['success'] and not r['collision'] for r in queries),teacher_results=R.M.tally(queries),selection=selection,
            source6_reused=same,recovery_reused_count=len(reused),recovery_new_count=len(query_jobs))
        save(folder/'coverage_gate.json',coverage);save(folder/'support_rows.json',dict(student_rows=source,recovery_rows=queries,reused_rows=reused,label_statistics=stats))
        if coverage['all_safe']:
            obj,shadow=N.make(p),N.make(p);initial=N.state(*obj);batch=N.data(p,source,queries,mode);trace=[]
            save(folder/'support_counts.json',{k:batch[k] for k in ['reference_count','old_own_count','current_own_count','own_count','correction_count']})
            for k in range(128):
                rt.role=mode.upper()+'_FIT';rt.guard(True);v=[]
                for w,objects in [('live',obj),('shadow',shadow)]:
                    M.events(mode,w)('BEFORE',k);v.append(N.bc_step(objects,batch));M.events(mode,w)('AFTER',k)
                assert v[0]==v[1] and digest(N.state(*obj))==digest(N.state(*shadow));completed+=1;trace.append(v[0])
                if k%16==0:save(folder/'fit_progress.json',dict(steps=completed,latest=v[0],epoch=time.time()))
            save(folder/'repair_trace.json',trace);proposal=N.export(obj,folder/'proposal.pt.gz',p['model_bytes'])
            candidate,_=M.evaluate(rt,p,proposal,'FINAL329');success=set(candidate['success_ids'])
            candidate.update(preserves_current=valid<=success,repairs_both=all(key in success for key in IDS[:2]),development_admitted=len(success)==329)
        status='COMPLETE'
    except BaseException:
        error=traceback.format_exc()
        if obj is not None and initial is not None:N.load(obj,initial);N.load(shadow,initial);assert digest(N.state(*obj))==digest(initial)
        save(folder/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=8
        save(folder/'summary.json',dict(state=status,pid=os.getpid(),mode=mode,index=int(index),elapsed_s=time.time()-rt.start,physics=sum(rt.counts),commands=rt.commands,
            error=error,candidate=candidate,proposal=proposal,coverage=coverage,student_rows=source,recovery_rows=queries,reused_recovery_rows=reused,
            completed_paired_BC=completed,BC_Actor_live=completed,BC_Actor_shadow=completed,new_PPO=0,new_Value=0,
            fresh16=None,reload78=None,six_gate=False,eight_started=False,all_native_workers_stopped=True))
        save(folder/'status.json',dict(state=status,epoch=time.time()));R.M.fcntl.flock(rt.lock,R.M.fcntl.LOCK_UN);rt.lock.close()
    assert status=='COMPLETE',error


def auditrow(p,r):
    a=arrays(r);assert int(a['native_counts'].sum())==r['physics'] and not r['administrative_censor']
    P.frozen=N.frozen
    try:M.audit_full_history(dict(p,teacher_first=r['teacher_first']) if r['teacher'] else p,r,a)
    finally:P.frozen=H.ORIGINAL_FROZEN
    if r['teacher']:
        q,kind,hsh=teacher(p,r['request']['job']);hidden=None;labels=[]
        for x in a['states']:
            action,hidden=teacher_command(q,kind,x,hidden);labels.append(action[0].numpy().copy())
        assert np.array_equal(np.asarray(labels,np.float32),a['teacher_actions'])
        if 'teacher_kind' in r:assert r['teacher_kind']==kind and r['teacher_source_SHA256']==hsh
    return a


def auditround(mode,index):
    folder=OUT/f'{mode}_{int(index)}';p=read(folder/'protocol.json');setup(folder);verify(p,True);s=read(folder/'summary.json')
    assert s['state']=='COMPLETE' and not (folder/'audit.json').exists()
    ids=[s['pid']]+[pid for g in read(folder/'worker_generations.json') for pid in g['workers']];assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    rows=list(map(json.loads,(folder/'cases.jsonl').read_text().splitlines())) if (folder/'cases.jsonl').exists() else []
    replays=list(map(json.loads,(folder/'replays.jsonl').read_text().splitlines())) if (folder/'replays.jsonl').exists() else []
    assert s['physics']==sum(r['physics'] for r in rows+replays) and s['commands']==16+len(rows)+len(replays)<=p['max_commands']
    for r in rows:
        assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert {k:digest(v) for k,v in a.items()}=={k:r['array_hashes'][k] for k in a}
        assert int(a['native_counts'].sum())==r['physics'] and not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1
        assert [bool(v) for v in a['end_flags'][-1]]==[r['collision'],r['success'],r['timeout']]
        if r['full']:auditrow(p,r)
    bypath={r['path']:r for r in rows}
    for r in replays:
        old=bypath[r['reference_path']];assert r['array_digest']==old['array_digest'] and r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
    for r in s['student_rows']+s['recovery_rows']:auditrow(p,r)
    support=read(folder/'support_rows.json');assert support['student_rows']==s['student_rows'] and support['recovery_rows']==s['recovery_rows']
    stats,_=U.label_statistics(p,s['student_rows']);assert stats==support['label_statistics']
    for r in s['recovery_rows']:prefix_exact(r,next(v for v in s['student_rows'] if v['job_id']==r['job_id']),r['teacher_first'])
    calls=0
    if s['coverage']['all_safe']:
        obj=N.make(p);batch=N.data(p,s['student_rows'],s['recovery_rows'],mode);trace=[N.bc_step(obj,batch) for _ in range(128)];calls=128
        assert trace==read(folder/'repair_trace.json') and digest(N.state(*obj))==s['proposal']['joint_hash']
        assert N.base_digest(obj[0])==p['base_actor_digest'] and sha(s['proposal']['path'])==s['proposal']['sha256'] and N.frozen(p,s['proposal']['path']).phash()==obj[0].phash()
        events=list(map(json.loads,(folder/'optimizer_calls.jsonl').read_text().splitlines()));assert len(events)==512
        for w in ['live','shadow']:assert [v[2:] for v in events if v[:2]==[mode,w]]==[[where,k] for k in range(128) for where in ['BEFORE','AFTER']]
        final=[r for r in rows if r['role']=='FINAL329'];assert len(final)==len({r['job_id'] for r in final})==329
        assert R.M.tally(final[:54])==s['candidate']['wide54'] and R.M.tally(final[54:])==s['candidate']['old275']
        actual_safe={r['job_id'] for r in final if r['success'] and not r['collision']}
        assert actual_safe==set(s['candidate']['success_ids'])
        assert s['candidate']['preserves_current']==(set(p['current_success_ids'])<=actual_safe)
        assert s['candidate']['development_admitted']==(len(actual_safe)==329)
        assert all(r['loaded_model_sha256']==s['proposal']['sha256'] and r['policy_hash']==s['proposal']['policy_hash'] for r in final)
    else:assert s['completed_paired_BC']==0 and s['candidate'] is None
    save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,new_PPO=0,discarded_historical_Adam=calls,
        all_source_new_reused_full120_history_hidden_density_teacher_labels_exact=True,replayed_native_arrays_and_integration_exact=True,
        compact329_NOT_full120_reconstructed=True,all_live_shadow_ActorAdam_rebuilt=True,physics=s['physics'],commands=s['commands'],six_gate=False))
