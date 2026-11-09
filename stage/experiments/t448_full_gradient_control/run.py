"""Fixed paired finite-sum versus cyclic BC estimator, same mature start."""
from pathlib import Path
import copy,importlib.util,json,os,shutil,subprocess,sys,time,traceback,io
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t448_full_gradient_control';DOC=ROOT/'stage/docs/research_rescue_20261109'
sys.path.insert(0,str(ROOT/'stage/experiments/t446_plain_joint'))
import t446_learning as P
spec=importlib.util.spec_from_file_location('stage448_pinned446',ROOT/'stage/experiments/t446_plain_joint/run.py');R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
torch,np,E,L=P.torch,P.np,P.E,P.L
read,save,sha,digest=P.read,P.save,P.sha,P.digest

def used(folder=None):return sum(f.stat().st_size for f in (Path(folder) if folder else ROOT/'stage/runs').rglob('*') if f.is_file())
def verify(p,full=False):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    assert used(OUT)<=p['max_bytes'] and used()-used(OUT)+p['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    for f,h in (p['inputs'] if full else p['protected_files']).items():assert sha(f)==h,f

def data(p):return P.dataset(p,p['teacher_rows'])
def objects(p):
    obj=P.make(p);P.load(obj,P.checkpoint(p['initial_plain_model']['path'])['joint_state']);assert digest(P.state(*obj))==p['initial_plain_model']['joint_hash'];return obj

def worker(pipe,index,p,counts):
    # Picklable top-level entry owned by this script; the pinned runtime is
    # imported independently inside every spawned child, never pickled itself.
    R.worker(pipe,index,p,counts)

class Pool(R.Pool):
    def __init__(self,p,counts,guard):
        self.guard=guard;self.children=[];self.pipes=[];ctx=R.M.mp.get_context('spawn')
        try:
            for i in range(8):
                a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
            save(Path(p['out'])/'workers.json',dict(parent=os.getpid(),workers=[c.pid for c in self.children]));self.ready=[self.receive(i) for i in range(8)];assert all(v['state']=='READY' for v in self.ready)
        except BaseException:self.close();raise

def full_step(obj,groups,step=True):
    q,v,ao,vo=obj;q.zero_grad(set_to_none=True);loss_value=0.
    for task in range(10):
        selected=[g for g in groups if g['task']==task];weight=1/16 if task<8 else 1/4
        for off in range(0,len(selected),8):
            chunk=selected[off:off+8];x=torch.nn.utils.rnn.pad_sequence([g['x'] for g in chunk]);y=torch.nn.utils.rnn.pad_sequence([g['y'] for g in chunk]);loc=q.sequence(x)
            loss=sum(P.action_loss(loc[:len(g['x']),i],y[:len(g['x']),i]).mean() for i,g in enumerate(chunk))*weight/len(selected)
            loss.backward();loss_value+=float(loss.detach())
    # Exactly one global clip and one mature Adam step after ALL219 histories.
    norm=float(torch.nn.utils.clip_grad_norm_(q.parameters(),1.,error_if_nonfinite=True))
    if step:ao.step()
    return dict(loss=loss_value,gradient_norm=norm)

def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t446_plain_joint';base=read(prior/'protocol.json');s=read(prior/'summary.json')
    assert read(prior/'audit.json')['state']=='PASS' and read(ROOT/'stage/runs/t447_fixed_credit_audit/audit.json')['state']=='PASS_FROZEN_DIAGNOSIS_ONLY'
    for group,folder in [('T446','t446_plain_joint'),('T447','t447_fixed_credit_audit')]:
        assert read(ROOT/f'stage/docs/long_distance_20261101/release_receipt_{group}.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
        assert all(not Path('/proc',str(pid)).exists() for pid in read(ROOT/'stage/runs'/folder/'SEALED.json')['scientific_pids'])
        assert not Path('/proc',str(read(ROOT/'stage/runs'/folder/'driver_started.json')['pid'])).exists()
    p=copy.deepcopy(base);p.update(id='T448_FULL_EMPIRICAL_GRADIENT_VS_CYCLIC_CONTROL',out=str(OUT),
        initial_plain_model=s['bc_model'],arms=['cyclic_control','full_empirical'],steps=64,bc_steps=64,maximum_bc_live_Adam=128,maximum_bc_shadow_Adam=128,continuation_cursor=1500,
        max_bytes=11*2**20,model_bytes=int(1.5*2**20),compact_bytes=4*1024,max_physics=14000000,max_commands=720,
        main_seconds=3600,learning_seconds=2100,audit_seconds=1200,backup_seconds=600,all_seconds=6000,
        maximum_actual_BC_live_Adam=128,maximum_BC_shadow_Adam=128,maximum_audit_discarded_Adam=128,preflight_discarded_Adam=4,
        PPO_updates=0,V_updates=0,models_frozen_at_deployment_single=True,fresh16=None,reload78=None,six_gate=False,eight_started=False,
        comparison='Same initialActor/V/matureAdam, LR, unique219 histories, group weights and64Adam. Full empirical uses more per-step data presentations/compute; NOT compute matched, NOT isolated noise effect or efficiency claim.',
        new_mechanism='Replace cyclic10-history gradient estimate with all219 histories exact grouped empirical gradient, microbatch8 onlyforRAM; oneclip/Adam afterallgroups.',
        continuation_is_matched_control_NOT_unbounded_T446_resume=True,
        parent_T447_conclusion='LEFT5 jointgradient cosine+.2367 and matureAdamlinear_delta negative; no evidence it is completely blocked. Errors across phases. No PCGrad/first2s specialweight.',
        user_authorization='Nov9 next step after review: independent fixed paired empirical-gradient mechanism, not expansion/restart of T446.')
    p['teacher_rows']=[json.loads(x) for x in (prior/'cases.jsonl').read_text().splitlines() if json.loads(x)['role']=='TEACHER_RECONSTRUCT']
    p['worst_physics']=2*sum(j['case']['step_cap'] for j in p['main_jobs']+p['old275_jobs']);assert p['worst_physics']<=p['max_physics']
    p['protected_files']={**base['protected_files'],p['initial_plain_model']['path']:p['initial_plain_model']['sha256']}
    files=set(base['inputs'])|{str(f) for f in HERE.glob('*.py')}|{str(DOC/'T448_PLAN.md'),str(DOC/'T448_PRIMARY_SOURCES.md'),str(prior/'summary.json'),str(prior/'audit.json'),str(prior/'cases.jsonl'),
        str(ROOT/'stage/runs/t447_fixed_credit_audit/diagnosis.json'),str(ROOT/'stage/runs/t447_fixed_credit_audit/audit.json'),p['initial_plain_model']['path']}|{r['path'] for r in p['teacher_rows']}
    p['inputs']={f:sha(f) for f in sorted(files)};assert used()+p['max_bytes']<2**30
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);verify(p,True)
    save(DOC/'T448_BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',own_max_bytes=p['max_bytes'],global_before=used()-used(OUT),global_full=used()-used(OUT)+p['max_bytes'],
        max_physics=p['max_physics'],worst_physics=p['worst_physics'],max_commands=720,maximum_all_pipe_commands=698,
        max_live_Adam=128,max_shadow_Adam=128,max_independent_audit_discarded_Adam=128,preflight_discarded_Adam=4,
        storage=dict(two_models=3*2**20,compact660=660*4*1024,metadata_protocol_logs=5*2**20),
        main_seconds=3600,learning_seconds=2100,audit_seconds=1200,all_seconds=6000,no_PPO=True,no_fresh_or_reload=True))
    print('T448_REGISTERED',p['worst_physics'],flush=True)

def preflight():
    p=read(OUT/'protocol.json');verify(p,True);assert not (OUT/'preflight.json').exists();dataset=data(p);groups=P.packs(dataset,'cuda')
    results={};durations={};order=P.schedule(dataset,p['continuation_cursor']+p['steps'])
    for arm in p['arms']:
        a=objects(p);b=objects(p);before=P.state(*a);start=time.monotonic()
        if arm=='full_empirical':left=full_step(a,groups);right=full_step(b,groups)
        else:left=P.bc_step(a,groups,order[p['continuation_cursor']]);right=P.bc_step(b,groups,order[p['continuation_cursor']])
        torch.cuda.synchronize();durations[arm]=time.monotonic()-start;assert left==right and digest(P.state(*a))==digest(P.state(*b))
        after=P.state(*a);P.load(a,before);assert digest(P.state(*a))==digest(before);P.load(a,after)
        assert digest(P.state(*a))==digest(after);results[arm]=left;del a,b
    diagnosis=read(ROOT/'stage/runs/t447_fixed_credit_audit/diagnosis.json');w=np.array(diagnosis['weights']);loss=sum(g['mean_surrogate_loss']*w[g['task']] for g in diagnosis['task_stats']);norm=float(np.sqrt(w@np.asarray(diagnosis['group_grad_gram'])@w))
    assert abs(results['full_empirical']['loss']-loss)<1e-12 and abs(results['full_empirical']['gradient_norm']-norm)<1e-10
    # No new model file: frozen source checkpoint suffices for cancellation.
    folder=OUT/'preflight_native';folder.mkdir();counts=R.M.mp.get_context('spawn').RawArray('q',9);pool=Pool(dict(p,out=str(folder),native_stop_epoch=time.time()+180,max_physics=1),counts,lambda:None)
    try:replies=pool.request([dict(model=p['initial_plain_model']['path'],job=j,seed=None,cancel_probe=True) for j in p['old8_jobs']]);assert all(x['state']=='CANCEL_PASS' for x in replies) and sum(counts)==0
    finally:pool.close()
    estimate=sum(durations.values())*p['steps'];assert estimate<1600,('Full live+shadow estimate too large',estimate)
    save(OUT/'preflight.json',dict(state='PASS',native=0,discarded_Actor_Adam=4,new_V_Adam=0,commands=24,results=results,
        seconds_live_plus_shadow_one_step=durations,estimated64_live_shadow_both_arms_seconds=estimate,
        full_loss_and_gradient_norm_match_T447_unbatched_group_calculation=True,full_loss_error=abs(results['full_empirical']['loss']-loss),
        full_gradient_norm_error=abs(results['full_empirical']['gradient_norm']-norm),mature_Adam_shadow_restore_exact=True,eight_workers_cancelled_and_stopped=True))
    print('T448_PREFLIGHT_PASS',estimate,flush=True)

def preflight_finalize():
    p=read(OUT/'protocol.json');verify(p,True);assert not (OUT/'preflight.json').exists()
    old=read(OUT/'preflight_failed_001/error.json');assert old['executed_prefix']['discarded_Actor_Adam']==4 and old['native']==0
    previous=read(OUT/'preflight_failed_001/protocol.json');assert sha(OUT/'preflight_failed_001/run.py.txt')==previous['inputs'][str(HERE/'run.py')]
    obj=objects(p);before=digest(P.state(*obj));groups=P.packs(data(p),'cuda');t=time.monotonic();result=full_step(obj,groups,step=False);torch.cuda.synchronize();seconds=time.monotonic()-t
    assert digest(P.state(*obj))==before
    diagnosis=read(ROOT/'stage/runs/t447_fixed_credit_audit/diagnosis.json');w=np.array(diagnosis['weights']);loss=sum(g['mean_surrogate_loss']*w[g['task']] for g in diagnosis['task_stats']);norm=float(np.sqrt(w@np.asarray(diagnosis['group_grad_gram'])@w))
    assert abs(result['loss']-loss)<1e-12 and abs(result['gradient_norm']-norm)<1e-10
    folder=OUT/'preflight_native';folder.mkdir();counts=R.M.mp.get_context('spawn').RawArray('q',9);pool=Pool(dict(p,out=str(folder),native_stop_epoch=time.time()+180,max_physics=1),counts,lambda:None)
    try:replies=pool.request([dict(model=p['initial_plain_model']['path'],job=j,seed=None,cancel_probe=True) for j in p['old8_jobs']]);assert all(x['state']=='CANCEL_PASS' for x in replies) and sum(counts)==0
    finally:pool.close()
    estimate=seconds*2*1.25*64;assert estimate<1600
    save(OUT/'preflight.json',dict(state='PASS',native=0,discarded_Actor_Adam=4,recovery_new_Adam=0,new_V_Adam=0,commands=24,
        completed_original_four_clone_Adam_prefix_reused=True,original_error_and_source_preserved=True,
        recovery_one_full_gradient_only_seconds=seconds,estimated64_full_live_shadow_seconds_with25percent_step_allowance=estimate,
        initial_cyclic_clone_timing_NOT_retained=True,full_loss_and_gradient_norm_match_T447=True,
        full_loss_error=abs(result['loss']-loss),full_gradient_norm_error=abs(result['gradient_norm']-norm),
        mature_Adam_shadow_restore_assertions_passed_in_original_prefix=True,eight_workers_cancelled_and_stopped=True,
        same_scientific_configuration=True,only_picklable_worker_entry_fixed=True))
    print('T448_PREFLIGHT_FINALIZED_ZERO_NEW_ADAM',estimate,flush=True)

def main():
    p=read(OUT/'protocol.json');verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'summary.json').exists()
    E.N.offline();E.L.install_stop_handlers();R.OUT=OUT;R.verify=verify;R.M.OUT=OUT;R.M.verify=verify;rt=R.Runtime(p);results={};state='FAILED';error=None;live=shadow_count=0
    try:
        dataset=data(p);groups=P.packs(dataset,'cuda');order=P.schedule(dataset,p['continuation_cursor']+p['steps']);rt.pool=Pool(dict(p,native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        for arm in p['arms']:
            rt.role='FIT_'+arm;obj=objects(p);shadow=objects(p);trace=[];initial=digest(P.state(*obj))
            for k in range(p['steps']):
                rt.guard(True)
                def step(which,objects):
                    nonlocal live,shadow_count
                    with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([arm,k,which,'BEFORE'])+'\n')
                    value=full_step(objects,groups) if arm=='full_empirical' else P.bc_step(objects,groups,order[p['continuation_cursor']+k])
                    with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([arm,k,which,'AFTER'])+'\n')
                    if which=='live':live+=1
                    else:shadow_count+=1
                    return value
                a=step('live',obj);b=step('shadow',shadow);assert a==b;trace.append(a)
                if k%8==0:assert digest(P.state(*obj))==digest(P.state(*shadow));save(OUT/'fit_progress.json',dict(arm=arm,steps=k+1,latest=a,epoch=time.time()))
            assert digest(P.state(*obj))==digest(P.state(*shadow));fit=P.assess(obj[0],dataset);model=P.export(obj,OUT/f'{arm}.pt.gz',p['model_bytes']);del obj,shadow;torch.cuda.empty_cache()
            final=rt.request(p['main_jobs']+p['old275_jobs'],model,'FINAL_'+arm)
            success=sorted(r['job_id'] for a,r in final if r['success'] and not r['collision'])
            result=dict(arm=arm,initial_joint_hash=initial,steps=64,trace=trace,fit=fit,model=model,wide54=R.M.tally([r for a,r in final[:54]]),old275=R.M.tally([r for a,r in final[54:]]),
                all329_safe=len(success)==329,success_ids=success,no_current_PPO=True)
            results[arm]=result;save(OUT/(arm+'_result.json'),result);save(OUT/'results.json',results)
        state='COMPLETE'
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,physics=sum(rt.counts),commands=rt.commands,live_Actor_Adam=live,shadow_Actor_Adam=shadow_count))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=8
        save(OUT/'summary.json',dict(state=state,pid=os.getpid(),elapsed_s=time.time()-rt.start,physics=sum(rt.counts),commands=rt.commands+24,
            live_Actor_Adam=live,shadow_Actor_Adam=shadow_count,V_Adam=0,PPO_updates=0,results=results,error=error,all_native_workers_stopped=True,fresh16=None,reload78=None,six_gate=False,eight_started=False))
        save(OUT/'status.json',dict(state=state,epoch=time.time()));R.M.fcntl.flock(rt.lock,R.M.fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error
    print('T448_MAIN_COMPLETE',flush=True)

def audit():
    p=read(OUT/'protocol.json');verify(p,True);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists()
    ids=[s['pid']]+read(OUT/'workers.json')['workers'];assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    E.N.offline();dataset=data(p);groups=P.packs(dataset,'cuda');order=P.schedule(dataset,p['continuation_cursor']+p['steps']);calls=0
    for arm in p['arms']:
        obj=objects(p);trace=[]
        for k in range(p['steps']):trace.append(full_step(obj,groups) if arm=='full_empirical' else P.bc_step(obj,groups,order[p['continuation_cursor']+k]));calls+=1
        result=s['results'][arm];assert trace==result['trace'] and digest(P.state(*obj))==result['model']['joint_hash'];assert P.assess(obj[0],dataset)==result['fit'];assert sha(result['model']['path'])==result['model']['sha256']
    rows=[json.loads(x) for x in (OUT/'cases.jsonl').read_text().splitlines()];assert len(rows)==658 and s['physics']==sum(r['physics'] for r in rows) and s['commands']==698
    for row in rows:
        assert sha(row['path'])==row['sha256'];a=E.N.S.arrays(row['path']);assert int(a['native_counts'].sum())==row['physics'];assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1
    events=[json.loads(x) for x in (OUT/'optimizer_calls.jsonl').read_text().splitlines()]
    for arm in p['arms']:
        for which in ['live','shadow']:assert [[v[1],v[3]] for v in events if v[0]==arm and v[2]==which]==[[k,w] for k in range(64) for w in ['BEFORE','AFTER']]
    save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,discarded_historical_Actor_Adam=calls,
        both_arms_full_Actor_V_and_mature_Adam_states_reconstructed=True,all_BC_traces_and_fit_metrics_exact=True,
        native_compact658_trajectory_arrays_hashes_and_budget_verified=True,compact_final_cases_full120_NOT_independently_reconstructed=True,
        physics=s['physics'],commands=s['commands'],PPO_updates=0,fresh16=None,reload78=None,six_gate=False))
    print('T448_AUDIT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip();assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip();now=time.time()
    with (OUT/'driver_console.log').open('xb') as f:
        c=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T448','--why=Finite full-empirical-gradient paired control',
            'timeout','--signal=TERM','--kill-after=2',str(p['all_seconds']),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=c.pid,epoch=now,learning_stop=now+p['learning_seconds'],native_stop=now+p['main_seconds'],all_stop=now+p['all_seconds'],source_remote=src['remote_commit'],protocol_sha256=sha(OUT/'protocol.json')))
    print('T448_ACTUALLY_LAUNCHED',c.pid,flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
