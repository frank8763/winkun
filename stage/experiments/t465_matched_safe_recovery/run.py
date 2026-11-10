"""Matched original-safe teacher recovery at the failed current-state handover."""
from pathlib import Path
import copy,importlib.util,json,os,shutil,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t465_matched_safe_recovery';DOC=ROOT/'stage/docs/review_20261110'
spec=importlib.util.spec_from_file_location('stage465_pinned464_learning',ROOT/'stage/experiments/t464_two_failure_recovery/t464_learning.py')
Y=importlib.util.module_from_spec(spec);spec.loader.exec_module(Y)
X,L,J,H,M,P,E,D,F,R,torch,np=Y.X,Y.L,Y.J,Y.H,Y.M,Y.P,Y.E,Y.D,Y.F,Y.R,Y.torch,Y.np
read,save,sha,digest=Y.read,Y.save,Y.sha,Y.digest
make,state,load,export,frozen,bc_step,data=Y.make,Y.state,Y.load,Y.export,Y.frozen,Y.bc_step,Y.data

def worker(pipe,index,p,counts):
    # The entire query implementation is unchanged frozen T449 code: safeT255
    # base_actions on each actual120 state, without recurrent teacher carry.
    P.frozen=frozen;M.worker(pipe,index,p,counts)

def verify(p,full=False):
    root=read(OUT/'protocol.json')
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert M.used(OUT)<=root['main_partition_bytes'] and M.used()-M.used(OUT)+root['max_bytes']<=2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f

def setup_runtime():
    M.OUT,M.verify,M.Pool=OUT,verify,Pool;M.setup_runtime()

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
    assert not OUT.exists(),'Never restart'
    prior=ROOT/'stage/runs/t464_two_failure_recovery';s=read(prior/'summary.json');old=read(prior/'protocol.json');report=read(DOC/'T464_FINAL_REPORT.json')
    assert s['state']=='COMPLETE' and s['completed_paired_BC']==0 and read(prior/'audit.json')['state']=='PASS'
    assert report['remote']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert all(not Path('/proc',str(v)).exists() for v in read(prior/'SEALED.json')['scientific_pids'])
    assert not Path('/proc',str(read(prior/'driver_started.json')['pid'])).exists()
    qrows={r['job_id']:r for r in s['query_rows']};assert not qrows['dev_task6_00']['success'] and qrows['dev_task6_21']['success']
    p=copy.deepcopy(old);jobs={j['id']:j for j in p['old275_jobs']+p['main_jobs']}
    p.update(id='T465_MATCHED_CURRENT_STATE_SAFE_TEACHER',out=str(OUT),previous_own4=s['own_rows'],previous_valid21=qrows['dev_task6_21'],
        previous_failed00=qrows['dev_task6_00'],recovery_jobs=[jobs['dev_task6_00']],
        max_bytes=61*2**20,main_partition_bytes=18*2**20,max_commands=355,preflight_commands=0,
        registered_actual_main_commands=347,overall_commands_max=473,registered_gate_commands=126,
        maximum_current_success_unique=0,existing_success_unique_reused=4,maximum_teacher_unique=1,maximum_teacher_exact_replays=1,
        prior_window_physics=report['window_physics'],prior_window_commands=report['window_commands'],
        mechanism='Matched single00safeT255reset teacher diagnostic at identical4sec currentPPO327prefix, replacing failedT461carry teacher control. Reuse alreadyexact4own/21validcorrection; if00safe/exact one128BC+shadow then329.',
        training_semantics='0newPPO/Value.2supervisedrecovery trajectories from explicitly different archived training teachers, purestudentdeployment.2BCpreflightclones0pipe0native.',
        comparison='Teacherweights and recurrentexecution differ jointly, not hidden-only causal claim. One fixednewteacher, no further teacher/time/seed search.',
        preflight_contract_reuse='Identical T463 genericM.worker/cancel contract already passed, no newpipe/cancel/geometry commands. CPUteacher/savedstate/fullActorAdam shadow tests only.',
        eight_permission='Awaituser review even after six_gate; neverlaunch8m.',eight_started=False)
    p['validated_reset00']=next(r for r in p['aggregate_recovery10'] if r['job_id']=='dev_task6_00')
    p['worst_main_physics']=sum(j['case']['step_cap'] for j in p['main_jobs']+p['old275_jobs'])+2*p['recovery_jobs'][0]['case']['step_cap']
    p['worst_main_commands']=347;p['worst_gate_commands']=126
    assert p['prior_window_commands']+347+126==p['window_total_commands_max']==1600
    assert p['prior_window_physics']+p['worst_main_physics']+p['worst_gate_physics']<=p['window_total_physics_max']
    assert time.time()+1800+600+1800+600+900<p['absolute_all_stop'] and time.time()+3600<p['absolute_native_stop'] and time.time()+900<p['absolute_learning_stop']
    files=set(old['inputs'])|{str(f) for f in HERE.glob('*.py')}|{r['path'] for r in s['own_rows']+s['query_rows']}
    files|={str(DOC/'T465_PLAN.md'),str(DOC/'T464_FINAL_REPORT.json'),str(prior/'protocol.json'),str(prior/'summary.json'),str(prior/'audit.json'),str(prior/'cases.jsonl'),str(prior/'replays.jsonl'),
        str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T464.json'),str(ROOT/'stage/runs/t463_evening_2000/preflight.json')}
    p['inputs']={f:sha(f) for f in sorted(files)};p['constructor_reads']=sorted(files)
    assert M.used()+p['max_bytes']<=2**30
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);verify(p,True)
    save(DOC/'T465_BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',full_global_bytes=M.used()-M.used(OUT)+p['max_bytes'],
        reserved_bytes=p['max_bytes'],main_registered_commands=347,internal_main_guard_ceiling355_NOT_extra8=True,gate_registered_commands=126,
        zero_new_preflight_pipe_commands=True,window_worst_commands=1600,window_worst_physics=p['prior_window_physics']+p['worst_main_physics']+p['worst_gate_physics'],
        main_worst_physics=p['worst_main_physics'],gate_worst_physics=p['worst_gate_physics'],original30M1600unchanged=True,absolute_all_stop=p['absolute_all_stop']))
    print('T465_REGISTERED',flush=True)

def preflight():
    p=read(OUT/'protocol.json');verify(p,True);assert not (OUT/'preflight.json').exists()
    a,b=make(p),make(p);initial=state(*a);assert digest(initial)==p['initial_actor_residual_joint']
    fixture=data(p,p['previous_own4'],[p['validated_reset00'],p['previous_valid21']]);va,vb=bc_step(a,fixture),bc_step(b,fixture)
    assert va==vb and digest(state(*a))==digest(state(*b)) and va['reference218']<1e-24 and va['own10']<1e-24
    model=export(a,OUT/'preflight_model.pt.gz',p['model_bytes']);assert frozen(p,model['path']).phash()==a[0].phash()
    load(a,initial);load(b,initial);assert digest(state(*a))==digest(initial)==digest(state(*b))
    teacher=E.safe_initial(E.N.science(p),p).eval().requires_grad_(False);xx=E.N.S.arrays(p['previous_failed00']['path'])['states'][:21]
    with torch.no_grad():
        aa=[teacher.base_actions(torch.tensor(x[None])).float().numpy() for x in xx]
        bb=[teacher.base_actions(torch.tensor(x[None])).float().numpy() for x in xx]
    assert all(np.array_equal(a,b) and np.isfinite(a).all() and np.abs(a).max()<=1 for a,b in zip(aa,bb))
    reuse=read(ROOT/'stage/runs/t463_evening_2000/preflight.json');assert reuse['state']=='PASS' and reuse['eight_workers_cancelled']
    save(OUT/'preflight.json',dict(state='PASS',native=0,commands=0,discarded_BC_Adam=2,
        full_mature_ActorAdam_shadow_rollback_and_serialization=True,safeT255reset_batch1_deterministic=True,
        unchanged_generic_worker_inherits_existing_cancel_contract=True,new_worker_cancellation_NOT_executed=True,
        reused_data_SHA_fullarray_validation=True,actual_new00recovery_NOT_yet_executed=True))
    print('T465_PREFLIGHT_PASS',flush=True)

def main():
    p=read(OUT/'protocol.json');verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'summary.json').exists()
    setup_runtime();E.N.offline();E.L.install_stop_handlers();rt=M.Runtime(p)
    status,error,candidate,proposal='FAILED',None,None,None;queries=[];gate=None;obj=shadow=initial=None;completed=0
    try:
        rt.pool=Pool(dict(p,native_stop_epoch=min(rt.deadline,p['absolute_native_stop'])),rt.counts,rt.guard)
        values=rt.request(p['recovery_jobs'],p['learning_initial'],'MATCHED_SAFE_T255_RECOVERY00',full=True,query=True);queries=[r for a,r in values]
        a,r=values[0];previous=E.N.S.arrays(p['previous_failed00']['path'])
        for key in ['states','poses','history_poses','history_goals','history_times','hidden_before']:assert np.array_equal(a[key][:21],previous[key][:21]),key
        for key in ['actions','native_counts']:assert np.array_equal(a[key][:20],previous[key][:20]),key
        assert digest(a['initial_native_integration'])==p['previous_failed00']['array_hashes']['initial_native_integration']
        rt.request(p['recovery_jobs'],p['learning_initial'],'MATCHED_SAFE_T255_RECOVERY00_EXACT',full=True,references=queries,query=True)
        mixed=[queries[0],p['previous_valid21']];gate=dict(all2_safe=all(r['success'] and not r['collision'] for r in mixed),
            teacher_results=R.M.tally(mixed),new00_safe=queries[0]['success'],reused21_safe=p['previous_valid21']['success'],matched_student_4sec_prefix_fullstates_exact=True)
        save(OUT/'coverage_gate.json',gate);save(OUT/'coverage_rows.json',dict(own_reused=p['previous_own4'],queries_mixed=mixed))
        if gate['all2_safe']:
            obj,shadow=make(p),make(p);initial=state(*obj);batch=data(p,p['previous_own4'],mixed);trace=[]
            save(OUT/'support_counts.json',{k:batch[k] for k in ['reference_count','old_own_count','own_count','correction_count']})
            for k in range(128):
                rt.role='MATCHED_SAFE_TWO_FAILURE_BC';rt.guard(True);v=[]
                for w,objects in [('live',obj),('shadow',shadow)]:
                    M.events('matched_safe_BC',w)('BEFORE',k);v.append(bc_step(objects,batch));M.events('matched_safe_BC',w)('AFTER',k)
                assert v[0]==v[1] and digest(state(*obj))==digest(state(*shadow));completed+=1;trace.append(v[0])
                if k%16==0:save(OUT/'fit_progress.json',dict(steps=completed,latest=v[0],epoch=time.time()))
            save(OUT/'repair_trace.json',trace);proposal=export(obj,OUT/'proposal.pt.gz',p['model_bytes'])
            candidate,_=M.evaluate(rt,p,proposal,'TWO_FAILURE_FINAL329');success=set(candidate['success_ids'])
            candidate.update(preserves_current327=set(p['current_success_ids'])<=success,repairs_both=all(k in success for k in ['dev_task6_00','dev_task6_21']),development_admitted=len(success)==329)
        status='COMPLETE'
    except BaseException:
        error=traceback.format_exc()
        if obj is not None and initial is not None:
            load(obj,initial);load(shadow,initial);assert digest(state(*obj))==digest(initial)
        save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=8
        assert rt.commands<=p['registered_actual_main_commands']
        save(OUT/'summary.json',dict(state=status,pid=os.getpid(),elapsed_s=time.time()-rt.start,physics=sum(rt.counts),commands=rt.commands,error=error,
            candidate=candidate,proposal=proposal,gate=gate,own_rows=p['previous_own4'],query_rows=queries,reused_valid21=p['previous_valid21'],
            completed_paired_BC=completed,BC_Actor_live=completed,BC_Actor_shadow=completed,new_PPO=0,new_Value=0,
            fresh16=None,reload78=None,six_gate=False,eight_started=False,all_native_workers_stopped=True))
        save(OUT/'status.json',dict(state=status,epoch=time.time()));R.M.fcntl.flock(rt.lock,R.M.fcntl.LOCK_UN);rt.lock.close()
    assert status=='COMPLETE',error
    print('T465_MAIN_COMPLETE',flush=True)

def audit():
    p=read(OUT/'protocol.json');verify(p,True);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists()
    ids=[s['pid']]+[v for g in read(OUT/'worker_generations.json') for v in g['workers']];assert all(not Path('/proc',str(v)).exists() for v in ids)
    rows=list(map(json.loads,(OUT/'cases.jsonl').read_text().splitlines()));replays=list(map(json.loads,(OUT/'replays.jsonl').read_text().splitlines()))
    assert s['physics']==sum(r['physics'] for r in rows+replays) and s['commands']==16+len(rows)+len(replays)<=347
    safe=E.safe_initial(E.N.science(p),p).eval().requires_grad_(False)
    for r in rows:
        assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert {k:digest(v) for k,v in a.items()}=={k:r['array_hashes'][k] for k in a}
        assert int(a['native_counts'].sum())==r['physics'] and not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1 and not r['administrative_censor']
        assert [bool(v) for v in a['end_flags'][-1]]==[r['collision'],r['success'],r['timeout']]
        if r['full']:
            assert digest(a)==r['array_digest'];P.frozen=frozen
            try:M.audit_full_history(p,r,a)
            finally:P.frozen=H.ORIGINAL_FROZEN
            assert r['teacher']
            with torch.no_grad():
                for i,x in enumerate(a['states']):assert np.array_equal(safe.base_actions(torch.tensor(x[None])).float().numpy()[0],a['teacher_actions'][i])
    for r in replays:
        prev=next(v for v in rows if v['path']==r['reference_path']);assert r['array_digest']==prev['array_digest'] and r['array_hashes']==prev['array_hashes'] and r['integration_sha256']==prev['integration_sha256']
    # Reused current own and21 were already exact native repeats inT464. Recheck
    # their source arrays and neural-teacher labels, not newphysical trials.
    for r in p['previous_own4']+[p['previous_valid21']]:
        assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert digest(a)==r['array_digest']
        P.frozen=frozen
        try:M.audit_full_history(p,r,a)
        finally:P.frozen=H.ORIGINAL_FROZEN
    carry=frozen(p,p['recovery_teacher_model']['path']);h=None;reuse=E.N.S.arrays(p['previous_valid21']['path'])
    with torch.no_grad():
        for i,x in enumerate(reuse['states']):
            command,_,h,_=carry.step(torch.tensor(x[None]),h);assert np.array_equal(command[0].numpy(),reuse['teacher_actions'][i])
    calls=0
    if s['gate']['all2_safe']:
        obj=make(p);batch=data(p,p['previous_own4'],[s['query_rows'][0],p['previous_valid21']]);trace=[bc_step(obj,batch) for _ in range(128)];calls=128
        assert trace==read(OUT/'repair_trace.json') and digest(state(*obj))==s['proposal']['joint_hash']
        assert Y.base_digest(obj[0])==p['base_actor_digest'] and sha(s['proposal']['path'])==s['proposal']['sha256'] and frozen(p,s['proposal']['path']).phash()==obj[0].phash()
        events=list(map(json.loads,(OUT/'optimizer_calls.jsonl').read_text().splitlines()));assert len(events)==512
        for w in ['live','shadow']:assert [v[2:] for v in events if v[:2]==['matched_safe_BC',w]]==[[where,k] for k in range(128) for where in ['BEFORE','AFTER']]
        final=[r for r in rows if r['role']=='TWO_FAILURE_FINAL329'];assert len(final)==len({r['job_id'] for r in final})==329
        assert R.M.tally(final[:54])==s['candidate']['wide54'] and R.M.tally(final[54:])==s['candidate']['old275']
        assert all(r['loaded_model_sha256']==s['proposal']['sha256'] and r['policy_hash']==s['proposal']['policy_hash'] for r in final)
        after=set(s['candidate']['success_ids']);save(OUT/'success_ID_comparison.json',dict(repaired=sorted(after-set(p['current_success_ids'])),lost=sorted(set(p['current_success_ids'])-after)))
    else:assert s['completed_paired_BC']==0 and s['candidate'] is None
    assert len(s['query_rows'])==1 and len(replays)==1
    save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,discarded_historical_Adam=calls,new_PPO=0,
        new00_safeT255reset_labels_and_fullhistory_exact=True,reused21_T461carry_labels_and_four_own_SHA_exact=True,
        reused_support_NOT_new_independent_or_newphysical=True,one_new_native_exact_replay=True,
        all_live_shadow_ActorAdam_rebuilt=True,compact329_NOT_full120_reconstructed=True,physics=s['physics'],commands=s['commands'],six_gate=False))
    print('T465_AUDIT_PASS',flush=True)

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
        c=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T465','--why=Bounded two failure correction, user review before8m',
            'timeout','--signal=TERM','--kill-after=2',str(int(p['absolute_all_stop']-now)),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=c.pid,epoch=now,all_stop=p['absolute_all_stop'],learning_stop=min(now+900,p['absolute_learning_stop']),native_stop=min(now+1800,p['absolute_native_stop']),
        protocol_sha256=sha(OUT/'protocol.json'),source_remote=src['remote_commit']))
    print('T465_ACTUALLY_LAUNCHED',c.pid,flush=True)

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        if OUT.exists():save(OUT/f'error_{sys.argv[1]}_{os.getpid()}.json',dict(error=traceback.format_exc(),epoch=time.time()))
        raise
