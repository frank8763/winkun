"""One actual-student recovery aggregation round with recurrent neural teacher.

The teacher exists only while collecting four supervised recovery trajectories.
Deployed candidate remains a single causal Actor; no planner or runtime switch.
"""
from pathlib import Path
import contextlib, copy, importlib.util, json, os, shutil, subprocess, sys, time, traceback
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'stage/runs/t462_recurrent_parent_correction'
DOC = ROOT/'stage/docs/review_20261110'
spec = importlib.util.spec_from_file_location('stage462_pinned461', ROOT/'stage/experiments/t461_validated_recovery_bundle/run.py')
J = importlib.util.module_from_spec(spec); spec.loader.exec_module(J)
G,H,M,P,R,E,D,F,torch,np = J.G,J.H,J.M,J.P,J.R,J.E,J.G.D,J.G.F,J.torch,J.np
read,save,sha,digest = J.read,J.save,J.sha,J.digest
state,load,export,frozen = J.state,J.load,J.export,J.frozen


def make(p, device='cuda'):
    obj = H.make(p, device)
    load(obj, P.checkpoint(p['learning_initial']['path'])['residual_state'])
    assert digest(state(*obj)) == p['learning_initial']['joint_hash']
    return obj


def verify(p,full=False):
    assert (HERE/'protocol.json').read_bytes() == (OUT/'protocol.json').read_bytes()
    assert M.used(OUT)<=p['max_bytes'] and M.used()-M.used(OUT)+p['max_bytes']<=2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30
    for f,h in (p['inputs'] if full else p['protected_files']).items(): assert sha(f)==h,f


def worker(pipe,index,p,counts):
    E.N.offline(); R.C.ban_learning(); os.sched_setaffinity(0,{index})
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); E.set_out(p['out'])
    folder=Path(p['out'])/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True)
    log=(folder/'console.log').open('a');stdout,stderr=sys.stdout,sys.stderr;s=w=None
    def guard(physical=False):
        E.L.check_stop();assert time.time()<p['native_stop_epoch']-5 and sum(counts)<p['max_physics']
    def charge(): counts[index+1]+=1;assert sum(counts)<=p['max_physics']
    try:
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            module,native,create,w,_=E.N.S.P.C.B.setup(folder/'native',E.N.science(p))
            original=copy.deepcopy(module.ACTIVE);E.L.install_stop_handlers()
            pipe.send(dict(state='READY',pid=os.getpid(),cpu=index))
            while True:
                req=pipe.recv()
                if req is None:break
                assert req.get('seed') is None
                q=frozen(p,req['model']);j=req['job']
                module.ACTIVE=copy.deepcopy(original)
                module.ACTIVE['circle_shift_x_m']=j.get('circle_shift_x_m',original['circle_shift_x_m'])
                before=counts[index+1];s=E.L.Session(j,native,create,w,guard,charge,q)
                s.hidden=torch.zeros((1,1,64),dtype=torch.float64);E.L.install_stop_handlers()
                if req.get('cancel_probe'):
                    def cancel(physical=False):
                        if physical:raise E.L.StageStop('T462 zero-native cancel')
                    s.guard=cancel
                    try:s.step();raise AssertionError('Cancellation missing')
                    except E.L.StageStop:pass
                    assert not s.rows and s.steps==0;s.close();s=None
                    pipe.send(dict(state='CANCEL_PASS',physics=0));continue
                query=req.get('query',False);labels=[];eh=None;expert=None
                if query:
                    expert=frozen(p,p['research_initial']['path'])
                    assert expert.phash()==p['research_initial']['policy_hash']
                while not s.done():
                    if query:
                        x=s.packet().copy()
                        with torch.no_grad():command,_,eh,_=expert.step(torch.tensor(x[None]),eh)
                        labels.append(command[0].numpy().copy())
                        if len(s.rows)>=p['teacher_first']:s.step(command[0].numpy(),owner=1,actor_active=True)
                        else:s.step(owner=0)
                        assert np.array_equal(s.rows[-1][0],x)
                    else:s.step(owner=0)
                a=E.payload(s)
                if query:a['teacher_actions']=np.asarray(labels,np.float32)
                r=s.result();r.update(policy_hash=q.phash(),physics=counts[index+1]-before,
                    array_digest=digest(a),array_hashes={k:digest(v) for k,v in a.items()},
                    integration_sha256=digest(a['final_native_integration']),pid=os.getpid(),cpu=index,
                    teacher=query,teacher_first=p['teacher_first'] if query else None,
                    recurrent_teacher_model_sha256=p['research_initial']['sha256'] if query else None,
                    teacher_hidden_burns_actual_student_full_history_from_h0=query)
                if req.get('full_path'):
                    path=Path(req['full_path']);path.parent.mkdir(parents=True,exist_ok=True)
                    blob=E.N.R.blob(E.N.S.packed(a));assert len(blob)<=p['full_bytes']
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
                a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close()
                self.children.append(c);self.pipes.append(a)
            self.ready=[self.receive(i) for i in range(8)];assert all(r['state']=='READY' for r in self.ready)
            f=Path(p['out'])/'worker_generations.json';gg=read(f) if f.exists() else []
            gg.append(dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time()));save(f,gg)
        except BaseException:self.close();raise


def setup_runtime():
    M.OUT,M.verify,M.Pool=OUT,verify,Pool;M.setup_runtime()


def data(p,new_own,recoveries):
    anchor=frozen(p,p['learning_initial']['path']).cuda()
    valid=set(p['current_success_ids'])
    refs=[g for g in M.dataset(p) if g['id'] in valid]
    groups=F.anchors(anchor,refs)
    oldown=[r for r in p['own_rows']+p['new_original_own5'] if r['job_id'] in valid]
    own=F.anchors(anchor,[dict(id=r['job_id'],task=10,x=(a:=E.N.S.arrays(r['path']))['states'],y=a['actions']) for r in oldown+new_own])
    corrections=[D.correction(r) for r in recoveries]
    assert len(oldown)==14 and len(own)==20 and len(corrections)==4
    assert all(g['id'] in valid for g in groups+own)
    return groups,own,corrections


def prepare():
    assert not OUT.exists(),'Never restart'
    prior=ROOT/'stage/runs/t461_validated_recovery_bundle';s=read(prior/'summary.json')
    assert s['state']=='COMPLETE' and read(prior/'audit.json')['state']=='PASS'
    assert read(DOC/'T461_FINAL_REPORT.json')['remote']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert all(not Path('/proc',str(pid)).exists() for pid in read(prior/'SEALED.json')['scientific_pids'])
    assert not Path('/proc',str(read(prior/'driver_started.json')['pid'])).exists()
    base=read(prior/'protocol.json');p=copy.deepcopy(base)
    rows=list(map(json.loads,(prior/'cases.jsonl').read_text().splitlines()))
    failure_ids=sorted(r['job_id'] for r in rows if not r['success'])
    assert failure_ids==['dev_task6_17','dev_task6_22','dev_task6_26','wide117_80_dev6_LEFT_13']
    jobs={j['id']:j for j in p['main_jobs']+p['old275_jobs']}
    own_ids=['dev_task6_00','dev_task6_03','dev_task6_09','dev_task6_14','dev_task6_21','dev_task6_24']
    p.update(id='T462_CURRENT_STUDENT_RECURRENT_PARENT_CORRECTION',out=str(OUT),
        learning_initial=s['proposal'],current_success_ids=s['candidate']['success_ids'],
        current_cases=[r for r in rows if r['job_id'] in set(failure_ids+own_ids)],recovery_jobs=[jobs[k] for k in failure_ids],preservation_jobs=[jobs[k] for k in own_ids],
        teacher_first=20,warmup_steps=128,maximum_teacher_trajectories=4,maximum_teacher_exact_replays=4,
        maximum_parent_preservation_trajectories=0,maximum_parent_exact_replays=0,
        maximum_current_success_trajectories=6,maximum_current_success_replays=6,
        max_bytes=14*2**20,max_physics=8000000,max_commands=432,
        main_seconds=2700,learning_seconds=1200,audit_seconds=600,backup_seconds=600,all_seconds=4500,
        mechanism='One currentT461325success iteration:6own actual successful fullhistories+exact,4current4sec recurrentoriginal328teacher recovery+exact allsafe before128BC; preservecurrent-success reference/20own targets, frozenbase1074residual/matureAdam.',
        new_mechanism='Change actual occupancy, teacher to recurrent neural successfulparent, and retention target to currentT461 success functions; exclude known failed functions. Teacherhiddenburnactualhistory.',
        training='128BC+128shadow on4newvalidatedrecoveries, same1074residualandmatureT461Adam, originalbasefrozen.0PPO/Value, no runtime teacher/router.',
        comparison='One finite DAgger-style coverage/teacher/target-stage jointpilot, not same-data128continuation, single-factor causal result, novelty or safety guarantee.',
        user_authorization='Nov10 user continue toward8m; solve4remaining with a finite actual current-state learning mechanism, preserveoriginalmodels andgate329/fresh16/reload78 before8.',
        final_selection='Fixed128th proposal full329; current325 and original328+00 all required. Any one recovery fails =>0BC stop; no teacher/time/seed replacement.',
        fresh16=None,reload78=None,six_gate=False,eight_started=False)
    p['protected_files']={**base['protected_files'],s['proposal']['path']:s['proposal']['sha256']}
    p['worst_physics']=sum(j['case']['step_cap'] for j in p['main_jobs']+p['old275_jobs'])+2*sum(j['case']['step_cap'] for j in p['recovery_jobs']+p['preservation_jobs'])
    assert p['worst_physics']<p['max_physics']
    files=set(base['inputs'])|{str(f) for f in HERE.glob('*.py')}|set(p['protected_files'])
    files|={str(DOC/'T462_PLAN.md'),str(DOC/'T462_PRIMARY_SOURCES.md'),str(DOC/'T461_FINDINGS_AND_NEXT.md'),
        str(prior/'protocol.json'),str(prior/'summary.json'),str(prior/'audit.json'),str(prior/'cases.jsonl'),str(prior/'success_ID_comparison.json'),
        str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T461.json')}
    p['inputs']={f:sha(f) for f in sorted(files)};p['constructor_reads']=sorted(files)
    storage=18*p['full_bytes']+329*p['compact_bytes']+p['model_bytes']+4*2**20
    assert storage<=p['max_bytes'] and M.used()+p['max_bytes']<2**30
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);verify(p,True)
    save(DOC/'T462_BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',own_max=p['max_bytes'],storage_worst=storage,
        global_full=M.used()-M.used(OUT)+p['max_bytes'],max_physics=p['max_physics'],worst_physics=p['worst_physics'],
        maximum_all_commands=389,max_commands=432,estimated_main_minutes=[12,30],hard_cap_NOT_queued_duration=True,
        main_seconds=2700,learning_seconds=1200,audit_seconds=600,all_seconds=4500,BC_live_max=128,BC_shadow_max=128,
        new_PPO=0,fresh16_reload78_NOT_reserved=True,unique_bytes_deleted=0))
    print('T462_REGISTERED',p['worst_physics'],flush=True)


def preflight():
    p=read(OUT/'protocol.json');verify(p,True);assert not (OUT/'preflight.json').exists();setup_runtime()
    a,b=make(p),make(p);initial=state(*a)
    fixture=data(p,p['new_original_own5'][:]+p['own_rows'][:1],[p['initial_correction']]*4)
    # Fixture ownIDs must correspond to current success; first original own is safe.
    va,vb=H.step(a,fixture),H.step(b,fixture)
    assert va==vb and digest(state(*a))==digest(state(*b))
    assert va['reference218']<1e-24 and va['own10']<1e-24
    load(a,initial);load(b,initial);assert digest(state(*a))==digest(initial)==digest(state(*b))
    expert=frozen(p,p['research_initial']['path']);x=E.N.S.arrays(p['initial_correction']['path'])['states'][:64]
    eh=None;outputs=[]
    with torch.no_grad():
        for row in x:
            _,loc,eh,_=expert.step(torch.tensor(row[None]),eh);outputs.append(loc[0].numpy())
        assert np.max(np.abs(np.asarray(outputs)-expert.sequence(torch.tensor(x)).numpy()))<1e-10
    counts=R.M.mp.get_context('spawn').RawArray('q',9);folder=OUT/'preflight_native';folder.mkdir()
    pool=Pool(dict(p,out=str(folder),native_stop_epoch=time.time()+180,max_physics=1),counts,lambda:None)
    try:
        replies=pool.request([dict(model=p['learning_initial']['path'],job=j,seed=None,cancel_probe=True) for j in p['old8_jobs']])
        assert all(r['state']=='CANCEL_PASS' for r in replies) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',native=0,commands=24,discarded_supervised_clone_Adam=2,
        current325model_and_mature_residualAdam_exact=True,current_success_anchor_initial_loss_zero=True,
        actual_teacher_h0_sequence_vs_step_verified=True,live_shadow_and_rollback_exact=True,
        fixture_NOT_actual4recoveries_or6successes=True,eight_workers_cancelled_and_stopped=True))
    print('T462_PREFLIGHT_PASS',flush=True)


def main():
    p=read(OUT/'protocol.json');verify(p,True)
    assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'summary.json').exists()
    setup_runtime();E.N.offline();E.L.install_stop_handlers();rt=M.Runtime(p)
    status,error,candidate,proposal='FAILED',None,None,None
    gate,own,queries,completed=None,[],[],0;obj=shadow=initial=None
    try:
        rt.pool=Pool(dict(p,native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        values=rt.request(p['preservation_jobs'],p['learning_initial'],'CURRENT_OWN_SUCCESS6',full=True)
        old={r['job_id']:r for r in p['current_cases']}
        F.check_parent_reproduction(values,old);own=[r for a,r in values]
        assert all(r['success'] and not r['collision'] for r in own)
        rt.request(p['preservation_jobs'],p['learning_initial'],'CURRENT_OWN_SUCCESS6_EXACT',full=True,references=own)
        values=rt.request(p['recovery_jobs'],p['learning_initial'],'RECURRENT_PARENT_RECOVERY4',full=True,query=True)
        queries=[r for a,r in values]
        for a,r in values:
            previous=E.N.S.arrays(old[r['job_id']]['path'])
            assert np.array_equal(a['actions'][:20],previous['actions'][:20])
            assert np.array_equal(a['native_counts'][:20],previous['native_counts'][:20])
            assert digest(a['initial_native_integration'])==old[r['job_id']]['array_hashes']['initial_native_integration']
        rt.request(p['recovery_jobs'],p['learning_initial'],'RECURRENT_PARENT_RECOVERY4_EXACT',full=True,references=queries,query=True)
        gate=dict(all4_safe=all(r['success'] and not r['collision'] for r in queries),teacher_results=R.M.tally(queries),
            current6_success_historical_exact=True)
        save(OUT/'coverage_gate.json',gate);save(OUT/'coverage_rows.json',dict(own=own,queries=queries))
        if gate['all4_safe']:
            obj,shadow=make(p),make(p);initial=state(*obj);train=data(p,own,queries);trace=[]
            for k in range(128):
                rt.role='CURRENT_RECURRENT_CORRECTION_BC';rt.guard(True);values=[]
                for which,objects in [('live',obj),('shadow',shadow)]:
                    M.events('recurrent_BC',which)('BEFORE',k);values.append(H.step(objects,train));M.events('recurrent_BC',which)('AFTER',k)
                assert values[0]==values[1] and digest(state(*obj))==digest(state(*shadow))
                completed+=1;trace.append(values[0])
                if k%16==0:save(OUT/'fit_progress.json',dict(steps=completed,latest=values[0],epoch=time.time()))
            save(OUT/'repair_trace.json',trace);proposal=export(obj,OUT/'proposal.pt.gz',p['model_bytes'])
            candidate,_=M.evaluate(rt,p,proposal,'RECURRENT_CORRECTION_FINAL329')
            success=set(candidate['success_ids'])
            candidate.update(preserves_current325=set(p['current_success_ids'])<=success,
                preserves_original328=set(p['initial_success_ids'])<=success,
                repairs00=p['failed_job']['id'] in success,development_admitted=len(success)==329)
        status='COMPLETE'
    except BaseException:
        error=traceback.format_exc()
        if obj is not None and initial is not None:
            load(obj,initial);load(shadow,initial);assert digest(state(*obj))==digest(initial)
        save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=8
        ev=list(map(json.loads,(OUT/'optimizer_calls.jsonl').read_text().splitlines())) if (OUT/'optimizer_calls.jsonl').exists() else []
        counts={w:sum(r[1:3]==[w,'AFTER'] for r in ev) for w in ['live','shadow']}
        save(OUT/'summary.json',dict(state=status,pid=os.getpid(),elapsed_s=time.time()-rt.start,physics=sum(rt.counts),commands=rt.commands+p['preflight_commands'],error=error,
            candidate=candidate,proposal=proposal,gate=gate,own_rows=own,query_rows=queries,
            Actor_BC_live=counts['live'],Actor_BC_shadow=counts['shadow'],completed_paired_BC=completed,
            PPO_Actor_live=0,PPO_Value_live=0,new_native_RL_training=0,retained_model=p['research_initial'],
            research_initial_not_promoted=p['learning_initial'],fresh16=None,reload78=None,six_gate=False,eight_started=False,all_native_workers_stopped=True))
        save(OUT/'status.json',dict(state=status,epoch=time.time()))
        R.M.fcntl.flock(rt.lock,R.M.fcntl.LOCK_UN);rt.lock.close()
    assert status=='COMPLETE',error
    print('T462_MAIN_COMPLETE',flush=True)


def audit():
    p=read(OUT/'protocol.json');verify(p,True);s=read(OUT/'summary.json')
    assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists()
    ids=[s['pid']]+[pid for g in read(OUT/'worker_generations.json') for pid in g['workers']]
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    rows=list(map(json.loads,(OUT/'cases.jsonl').read_text().splitlines()))
    replays=list(map(json.loads,(OUT/'replays.jsonl').read_text().splitlines()))
    assert s['physics']==sum(r['physics'] for r in rows+replays) and s['commands']==24+16+len(rows)+len(replays)
    for r in rows:
        assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path'])
        assert {k:digest(v) for k,v in a.items()}=={k:r['array_hashes'][k] for k in a}
        assert int(a['native_counts'].sum())==r['physics'] and np.abs(a['actions']).max()<=1
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1 and not r['administrative_censor']
        assert [bool(v) for v in a['end_flags'][-1]]==[r['collision'],r['success'],r['timeout']]
        if r['full']:
            assert digest(a)==r['array_digest'] and np.array_equal(a['states'][1:],a['next_states'][:-1])
            P.frozen=frozen
            try:M.audit_full_history(p,r,a)
            finally:P.frozen=H.ORIGINAL_FROZEN
            if r['teacher']:
                teacher=frozen(p,p['research_initial']['path']);h=None
                with torch.no_grad():
                    for i,x in enumerate(a['states']):
                        command,_,h,_=teacher.step(torch.tensor(x[None]),h)
                        assert np.array_equal(command[0].numpy(),a['teacher_actions'][i])
                assert r['recurrent_teacher_model_sha256']==p['research_initial']['sha256']
    bypath={r['path']:r for r in rows}
    for r in replays:
        old=bypath[r['reference_path']]
        assert r['array_digest']==old['array_digest'] and r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
    calls=0
    if s['gate']['all4_safe']:
        obj=make(p);train=data(p,s['own_rows'],s['query_rows'])
        trace=[H.step(obj,train) for _ in range(128)];calls=128
        assert trace==read(OUT/'repair_trace.json') and digest(state(*obj))==s['proposal']['joint_hash']
        assert H.base_digest(obj[0])==p['base_actor_digest']
        assert sha(s['proposal']['path'])==s['proposal']['sha256'] and frozen(p,s['proposal']['path']).phash()==obj[0].phash()
        ev=list(map(json.loads,(OUT/'optimizer_calls.jsonl').read_text().splitlines()));assert len(ev)==512
        for w in ['live','shadow']:
            assert [r[2:] for r in ev if r[:2]==['recurrent_BC',w]]==[[where,k] for k in range(128) for where in ['BEFORE','AFTER']]
        final=[r for r in rows if r['role']=='RECURRENT_CORRECTION_FINAL329']
        assert len(final)==len({r['job_id'] for r in final})==329
        assert R.M.tally(final[:54])==s['candidate']['wide54'] and R.M.tally(final[54:])==s['candidate']['old275']
        assert all(r['loaded_model_sha256']==s['proposal']['sha256'] and r['policy_hash']==s['proposal']['policy_hash'] for r in final)
        success=set(s['candidate']['success_ids'])
        save(OUT/'success_ID_comparison.json',dict(lost_current325=sorted(set(p['current_success_ids'])-success),
            repaired_current4=sorted(success-set(p['current_success_ids'])),lost_original328=sorted(set(p['initial_success_ids'])-success)))
    else:assert s['Actor_BC_live']==s['Actor_BC_shadow']==0 and s['candidate'] is None
    assert len(s['own_rows'])==6 and len(s['query_rows'])==4 and len(replays)==10
    save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),discarded_historical_residual_Adam=calls,
        complete_full120_hidden_history_density_reward_and_recurrent_teacher_burnin_verified=True,
        ten_native_exact_replays=True,current6_successes_are_historical_policy_job_repeats=True,
        final329_compact_NOT_full120_reconstructed=True,physics=s['physics'],commands=s['commands'],
        new_native=0,new_PPO=0,fresh16=None,reload78=None,six_gate=False,eight_started=False))
    print('T462_AUDIT_PASS',flush=True)


def launch():
    p=read(OUT/'protocol.json');verify(p,True)
    assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    now=time.time()
    with (OUT/'driver_console.log').open('xb') as f:
        c=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T462','--why=One bounded recurrent neural teacher aggregation round',
            'timeout','--signal=TERM','--kill-after=2',str(p['all_seconds']),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=c.pid,epoch=now,learning_stop=now+p['learning_seconds'],native_stop=now+p['main_seconds'],all_stop=now+p['all_seconds'],protocol_sha256=sha(OUT/'protocol.json'),source_remote=src['remote_commit']))
    print('T462_ACTUALLY_LAUNCHED',c.pid,flush=True)


if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        if OUT.exists():save(OUT/f'error_{sys.argv[1]}_{os.getpid()}.json',dict(error=traceback.format_exc(),epoch=time.time()))
        raise
