"""One bounded current-policy two-failure recovery update; no eight-metre launch."""
from pathlib import Path
import contextlib,copy,json,os,shutil,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t464_two_failure_recovery';DOC=ROOT/'stage/docs/review_20261110'
sys.path.insert(0,str(HERE))
import t464_learning as Y
X,L,J,H,M,P,E,D,F,R,torch,np=Y.X,Y.L,Y.J,Y.H,Y.M,Y.P,Y.E,Y.D,Y.F,Y.R,Y.torch,Y.np
read,save,sha,digest=Y.read,Y.save,Y.sha,Y.digest
make,state,load,export,frozen,bc_step,data=Y.make,Y.state,Y.load,Y.export,Y.frozen,Y.bc_step,Y.data

def verify(p,full=False):
    root=read(OUT/'protocol.json')
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert M.used(OUT)<=root['main_partition_bytes'] and M.used()-M.used(OUT)+root['max_bytes']<=2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f

def setup_runtime():
    M.OUT,M.verify,M.Pool=OUT,verify,Pool;M.setup_runtime()

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
                        if physical:raise E.L.StageStop('T464 zero-native cancel')
                    s.guard=cancel
                    try:s.step();raise AssertionError('Cancellation missing')
                    except E.L.StageStop:pass
                    assert not s.rows and s.steps==0;s.close();s=None
                    pipe.send(dict(state='CANCEL_PASS',physics=0));continue
                query=req.get('query',False);labels=[];eh=None;expert=None
                if query:
                    expert=frozen(p,p['recovery_teacher_model']['path'])
                    assert expert.phash()==p['recovery_teacher_model']['policy_hash']
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
                    recurrent_teacher_model_sha256=p['recovery_teacher_model']['sha256'] if query else None,
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


def prepare():
    assert not OUT.exists(),'Never restart an ended or registered directory'
    prior=ROOT/'stage/runs/t463_evening_2000';root=read(prior/'protocol.json');report=read(DOC/'T463_FINAL_REPORT.json')
    assert report['remote']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert all(not Path('/proc',str(pid)).exists() for pid in read(prior/'SEALED.json')['scientific_pids'])
    assert not Path('/proc',str(read(prior/'driver_started.json')['pid'])).exists()
    initial=read(prior/'ppo/summary.json')['candidate'];assert len(initial['success_ids'])==327
    assert read(prior/'ppo/audit.json')['state']=='PASS'
    source_rows=list(map(json.loads,(prior/'ppo/cases.jsonl').read_text().splitlines()))
    final=[r for r in source_rows if r['role']=='FINAL329'];failed=sorted(r['job_id'] for r in final if not r['success'])
    assert failed==['dev_task6_00','dev_task6_21']
    p=copy.deepcopy(root['science']);jobs={j['id']:j for j in p['main_jobs']+p['old275_jobs']}
    ownids=['dev_task6_17','dev_task6_22','dev_task6_26','wide117_80_dev6_LEFT_13']
    teacher=read(ROOT/'stage/runs/t461_validated_recovery_bundle/summary.json')['proposal']
    teacher_evidence=read(ROOT/'stage/runs/t462_recurrent_parent_correction/summary.json')['own_rows']
    assert all(any(r['job_id']==key and r['success'] and r['policy_hash']==teacher['policy_hash'] for r in teacher_evidence) for key in failed)
    p.update(id='T464_TWO_CURRENT_FAILURE_RECOVERY',out=str(OUT),learning_initial=initial['model'],
        current_success_ids=initial['success_ids'],recovery_teacher_model=teacher,current_cases=[r for r in final if r['job_id'] in failed+ownids],
        preservation_jobs=[jobs[k] for k in ownids],recovery_jobs=[jobs[k] for k in failed],
        teacher_first=20,warmup_steps=128,maximum_current_success_unique=4,maximum_teacher_unique=2,
        max_bytes=67*2**20,main_partition_bytes=24*2**20,gate_partition_bytes=43*2**20,
        max_physics=8000000,max_commands=381,overall_physics_max=12000000,overall_commands_max=525,
        main_seconds=1800,learning_seconds=900,audit_seconds=600,gate_seconds=1800,gate_audit_seconds=600,backup_seconds=900,
        model_bytes=768*1024,full_bytes=448*1024,compact_bytes=4096,preflight_commands=24,
        absolute_learning_stop=root['absolute_learning_stop'],absolute_native_stop=root['absolute_native_stop'],absolute_all_stop=root['absolute_all_stop'],
        prior_window_physics=report['physics'],prior_window_commands=report['commands'],
        window_total_physics_max=root['max_physics'],window_total_commands_max=root['max_commands'],
        user_authorization='Nov10 explicitly fix the two failures; deliver results for user review before any8m. Same20:00hardstop and original30M/1600window totals.',
        mechanism='One current327-policy4sec actual occupancy correction using frozenT461 neural teacher. Bothsafe/exact plus4currentnewsuccesses/exact before128BC, original52866frozen1074residual matureActorAdam.',
        training_semantics='Demonstration-assisted supervised repair only; no newPPO/Value update. PriorPPOValue remains archived in exact input, excluded from activeActorAdam state.',
        comparison='New data/teacher/currentretention jointpilot, not an extension ofT463same-dataPPO or singlefactor/novelty/safetyproof.',
        final_selection='One fixed128th final329; everycurrent327success and bothfailedIDs required. Anyrecoveryfails =>0BC. No seed/teacher/time/radius/epoch grid.',
        eight_started=False,eight_permission='Neverlaunch8m here, even if six gate passes. Awaituser review.',
        protected_files={**root['protected_files'],initial['model']['path']:initial['model']['sha256'],teacher['path']:teacher['sha256']})
    # Explicitly preserve both original source joint and the active subset digest.
    c=P.checkpoint(p['learning_initial']['path']);assert digest(c['residual_state'])==p['learning_initial']['joint_hash']
    p['initial_actor_residual_joint']=digest({k:c['residual_state'][k] for k in ['actor','residual_opt']})
    p['initial_value_archived_not_loaded']=all(k in c['residual_state'] for k in ['value','value_opt'])
    p['worst_main_physics']=sum(j['case']['step_cap'] for j in p['main_jobs']+p['old275_jobs'])+2*sum(j['case']['step_cap'] for j in p['preservation_jobs']+p['recovery_jobs'])
    p['worst_gate_physics']=sum(j['case']['step_cap'] for j in p['old8_jobs']+p['main_jobs']+2*p['fresh_jobs'])
    p['worst_main_commands']=24+16+329+12;p['worst_gate_commands']=32+16+78
    assert p['worst_main_physics']<8000000 and p['worst_gate_physics']<4000000
    assert report['physics']+p['worst_main_physics']+p['worst_gate_physics']<=p['window_total_physics_max']
    assert report['commands']+p['worst_main_commands']+p['worst_gate_commands']<=p['window_total_commands_max']
    assert time.time()+1800+600+1800+600+900<p['absolute_all_stop']
    assert time.time()+900<p['absolute_learning_stop'] and time.time()+1800+1800<p['absolute_native_stop']
    files=set(root['inputs'])|{str(f) for f in HERE.glob('*.py')}|set(p['protected_files'])
    files|={str(DOC/'T464_PLAN.md'),str(DOC/'T464_T463_FAILURE_FINDINGS.md'),str(DOC/'T463_FINAL_REPORT.json'),
        str(prior/'ppo/summary.json'),str(prior/'ppo/audit.json'),str(prior/'ppo/cases.jsonl'),str(prior/'protocol.json'),
        str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T463.json')}
    p['inputs']={f:sha(f) for f in sorted(files)};p['constructor_reads']=sorted(files)
    assert M.used()+p['max_bytes']<=2**30
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);verify(p,True)
    save(DOC/'T464_BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',global_full=M.used()-M.used(OUT)+p['max_bytes'],
        total_reserved_bytes=p['max_bytes'],main_reserved_bytes=p['main_partition_bytes'],gate_reserved_bytes=p['gate_partition_bytes'],
        main_worst_physics=p['worst_main_physics'],gate_worst_physics=p['worst_gate_physics'],main_commands=p['worst_main_commands'],gate_commands=p['worst_gate_commands'],
        combined_window_worst_physics=report['physics']+p['worst_main_physics']+p['worst_gate_physics'],combined_window_worst_commands=report['commands']+p['worst_main_commands']+p['worst_gate_commands'],
        original_window_max_physics=root['max_physics'],original_window_max_commands=root['max_commands'],fresh16_reload78_fully_reserved=True,
        absolute_all_stop=p['absolute_all_stop'],new_PPO=0,new_Value=0,BC_live=128,BC_shadow=128))
    print('T464_REGISTERED',flush=True)

def preflight():
    p=read(OUT/'protocol.json');verify(p,True);assert not (OUT/'preflight.json').exists();setup_runtime()
    a,b=make(p),make(p);initial=state(*a);assert digest(initial)==p['initial_actor_residual_joint']
    valid=set(p['current_success_ids']);old=read(ROOT/'stage/runs/t462_recurrent_parent_correction/summary.json')['own_rows']
    fixture_own=[r for r in old if r['job_id'] in valid][:4];assert len(fixture_own)==4
    fixture=data(p,fixture_own,p['aggregate_recovery10'][:2]);va,vb=bc_step(a,fixture),bc_step(b,fixture)
    assert va==vb and digest(state(*a))==digest(state(*b))
    assert va['reference218']<1e-24 and va['own10']<1e-24
    model=export(a,OUT/'preflight_model.pt.gz',p['model_bytes']);assert frozen(p,model['path']).phash()==a[0].phash()
    load(a,initial);load(b,initial);assert digest(state(*a))==digest(initial)==digest(state(*b))
    teacher=frozen(p,p['recovery_teacher_model']['path']);xx=E.N.S.arrays(fixture_own[0]['path'])['states'][:64]
    h=None;loc=[]
    with torch.no_grad():
        for x in xx:_,mean,h,_=teacher.step(torch.tensor(x[None]),h);loc.append(mean[0].numpy())
        err=float(np.max(np.abs(np.asarray(loc)-teacher.sequence(torch.tensor(xx)).numpy())));assert err<1e-10
    counts=R.M.mp.get_context('spawn').RawArray('q',9);folder=OUT/'preflight_native';folder.mkdir()
    pool=Pool(dict(p,out=str(folder),native_stop_epoch=time.time()+180,max_physics=1),counts,lambda:None)
    try:
        replies=pool.request([dict(model=p['learning_initial']['path'],job=j,seed=None,cancel_probe=True) for j in p['old8_jobs']])
        assert all(r['state']=='CANCEL_PASS' for r in replies) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',native=0,commands=24,discarded_BC_Adam=2,
        source_fulljoint_and_extracted_ActorAdam_verified=True,original52866frozen1074trainable=True,
        current_success_function_initial_loss_zero=True,teacher_h0_sequence_step_error=err,live_shadow_rollback_serialization_exact=True,
        four_own_two_recovery_fixture_NOT_actual_navigation=True,eight_workers_cancelled_and_stopped=True))
    print('T464_PREFLIGHT_PASS',flush=True)

def main():
    p=read(OUT/'protocol.json');verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'summary.json').exists()
    setup_runtime();E.N.offline();E.L.install_stop_handlers();rt=M.Runtime(p)
    status,error,candidate,proposal='FAILED',None,None,None;own=[];queries=[];gate=None;obj=shadow=initial=None;completed=0
    try:
        rt.pool=Pool(dict(p,native_stop_epoch=min(rt.deadline,p['absolute_native_stop'])),rt.counts,rt.guard)
        values=rt.request(p['preservation_jobs'],p['learning_initial'],'CURRENT_OWN_SUCCESS4',full=True)
        historical={r['job_id']:r for r in p['current_cases']};F.check_parent_reproduction(values,historical);own=[r for a,r in values]
        assert all(r['success'] and not r['collision'] for r in own)
        rt.request(p['preservation_jobs'],p['learning_initial'],'CURRENT_OWN_SUCCESS4_EXACT',full=True,references=own)
        values=rt.request(p['recovery_jobs'],p['learning_initial'],'T461_NEURAL_RECOVERY2',full=True,query=True);queries=[r for a,r in values]
        for a,r in values:
            previous=E.N.S.arrays(historical[r['job_id']]['path'])
            assert np.array_equal(a['actions'][:20],previous['actions'][:20]) and np.array_equal(a['native_counts'][:20],previous['native_counts'][:20])
            assert digest(a['initial_native_integration'])==historical[r['job_id']]['array_hashes']['initial_native_integration']
        rt.request(p['recovery_jobs'],p['learning_initial'],'T461_NEURAL_RECOVERY2_EXACT',full=True,references=queries,query=True)
        gate=dict(all2_safe=all(r['success'] and not r['collision'] for r in queries),teacher_results=R.M.tally(queries),current4_historical_exact=True)
        save(OUT/'coverage_gate.json',gate);save(OUT/'coverage_rows.json',dict(own=own,queries=queries))
        if gate['all2_safe']:
            obj,shadow=make(p),make(p);initial=state(*obj);batch=data(p,own,queries);trace=[]
            save(OUT/'support_counts.json',{k:batch[k] for k in ['reference_count','old_own_count','own_count','correction_count']})
            for k in range(128):
                rt.role='CURRENT_TWO_FAILURE_BC';rt.guard(True);v=[]
                for w,objects in [('live',obj),('shadow',shadow)]:
                    M.events('two_failure_BC',w)('BEFORE',k);v.append(bc_step(objects,batch));M.events('two_failure_BC',w)('AFTER',k)
                assert v[0]==v[1] and digest(state(*obj))==digest(state(*shadow));completed+=1;trace.append(v[0])
                if k%16==0:save(OUT/'fit_progress.json',dict(steps=completed,latest=v[0],epoch=time.time()))
            save(OUT/'repair_trace.json',trace);proposal=export(obj,OUT/'proposal.pt.gz',p['model_bytes'])
            candidate,_=M.evaluate(rt,p,proposal,'TWO_FAILURE_FINAL329');success=set(candidate['success_ids'])
            candidate.update(preserves_current327=set(p['current_success_ids'])<=success,repairs_both=all(j['id'] in success for j in p['recovery_jobs']),development_admitted=len(success)==329)
        status='COMPLETE'
    except BaseException:
        error=traceback.format_exc()
        if obj is not None and initial is not None:
            load(obj,initial);load(shadow,initial);assert digest(state(*obj))==digest(initial)
        save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=8
        save(OUT/'summary.json',dict(state=status,pid=os.getpid(),elapsed_s=time.time()-rt.start,physics=sum(rt.counts),commands=rt.commands+p['preflight_commands'],error=error,
            candidate=candidate,proposal=proposal,gate=gate,own_rows=own,query_rows=queries,completed_paired_BC=completed,
            BC_Actor_live=completed,BC_Actor_shadow=completed,new_PPO=0,new_Value=0,fresh16=None,reload78=None,six_gate=False,eight_started=False,
            retained_model=p['research_initial'],input_research_model_NOT_promoted=p['learning_initial'],all_native_workers_stopped=True))
        save(OUT/'status.json',dict(state=status,epoch=time.time()));R.M.fcntl.flock(rt.lock,R.M.fcntl.LOCK_UN);rt.lock.close()
    assert status=='COMPLETE',error
    print('T464_MAIN_COMPLETE',flush=True)

def audit():
    p=read(OUT/'protocol.json');verify(p,True);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists()
    ids=[s['pid']]+[v for g in read(OUT/'worker_generations.json') for v in g['workers']];assert all(not Path('/proc',str(v)).exists() for v in ids)
    rows=list(map(json.loads,(OUT/'cases.jsonl').read_text().splitlines()));replays=list(map(json.loads,(OUT/'replays.jsonl').read_text().splitlines()))
    assert s['physics']==sum(r['physics'] for r in rows+replays) and s['commands']==24+16+len(rows)+len(replays)<=p['max_commands']
    for r in rows:
        assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert {k:digest(v) for k,v in a.items()}=={k:r['array_hashes'][k] for k in a}
        assert int(a['native_counts'].sum())==r['physics'] and not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1 and not r['administrative_censor']
        assert [bool(v) for v in a['end_flags'][-1]]==[r['collision'],r['success'],r['timeout']]
        if r['full']:
            assert digest(a)==r['array_digest'];P.frozen=frozen
            try:M.audit_full_history(p,r,a)
            finally:P.frozen=H.ORIGINAL_FROZEN
            if r['teacher']:
                teacher=frozen(p,p['recovery_teacher_model']['path']);h=None
                with torch.no_grad():
                    for i,x in enumerate(a['states']):
                        command,_,h,_=teacher.step(torch.tensor(x[None]),h);assert np.array_equal(command[0].numpy(),a['teacher_actions'][i])
                assert r['recurrent_teacher_model_sha256']==p['recovery_teacher_model']['sha256']
    bypath={r['path']:r for r in rows}
    for r in replays:
        prev=bypath[r['reference_path']];assert r['array_digest']==prev['array_digest'] and r['array_hashes']==prev['array_hashes'] and r['integration_sha256']==prev['integration_sha256']
    calls=0
    if s['gate']['all2_safe']:
        obj=make(p);batch=data(p,s['own_rows'],s['query_rows']);trace=[bc_step(obj,batch) for _ in range(128)];calls=128
        assert trace==read(OUT/'repair_trace.json') and digest(state(*obj))==s['proposal']['joint_hash']
        assert Y.base_digest(obj[0])==p['base_actor_digest'] and sha(s['proposal']['path'])==s['proposal']['sha256']
        assert frozen(p,s['proposal']['path']).phash()==obj[0].phash()
        events=list(map(json.loads,(OUT/'optimizer_calls.jsonl').read_text().splitlines()));assert len(events)==512
        for w in ['live','shadow']:assert [v[2:] for v in events if v[:2]==['two_failure_BC',w]]==[[where,k] for k in range(128) for where in ['BEFORE','AFTER']]
        final=[r for r in rows if r['role']=='TWO_FAILURE_FINAL329'];assert len(final)==len({r['job_id'] for r in final})==329
        assert R.M.tally(final[:54])==s['candidate']['wide54'] and R.M.tally(final[54:])==s['candidate']['old275']
        assert all(r['loaded_model_sha256']==s['proposal']['sha256'] and r['policy_hash']==s['proposal']['policy_hash'] for r in final)
        after=set(s['candidate']['success_ids']);save(OUT/'success_ID_comparison.json',dict(repaired=sorted(after-set(p['current_success_ids'])),lost=sorted(set(p['current_success_ids'])-after)))
    else:assert s['completed_paired_BC']==0 and s['candidate'] is None
    assert len(s['own_rows'])==4 and len(s['query_rows'])==2 and len(replays)==6
    save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),discarded_historical_Adam=calls,new_native=0,new_PPO=0,
        all_full_history_hidden_teacher_density_reward_SHA_exact=True,all_live_shadow_ActorAdam_rebuilt=True,
        six_native_exact_replays=True,own4_are_same_policy_job_historical_repeats=True,compact329_NOT_full120_reconstructed=True,
        physics=s['physics'],commands=s['commands'],six_gate=False,eight_started=False))
    print('T464_AUDIT_PASS',flush=True)

def install_gate():
    X.OUT,X.HERE,X.DOC=OUT,HERE,DOC

def gatemain():install_gate();X.gatemain()
def gateaudit():install_gate();X.gateaudit()

def launch():
    p=read(OUT/'protocol.json');verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    now=time.time();assert now+1800+600+1800+600+900<p['absolute_all_stop'] and now+900<p['absolute_learning_stop']
    with (OUT/'driver_console.log').open('xb') as f:
        c=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T464','--why=Bounded two failure correction, user review before8m',
            'timeout','--signal=TERM','--kill-after=2',str(int(p['absolute_all_stop']-now)),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=c.pid,epoch=now,all_stop=p['absolute_all_stop'],learning_stop=min(now+900,p['absolute_learning_stop']),native_stop=min(now+1800,p['absolute_native_stop']),
        protocol_sha256=sha(OUT/'protocol.json'),source_remote=src['remote_commit']))
    print('T464_ACTUALLY_LAUNCHED',c.pid,flush=True)

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        if OUT.exists():save(OUT/f'error_{sys.argv[1]}_{os.getpid()}.json',dict(error=traceback.format_exc(),epoch=time.time()))
        raise
