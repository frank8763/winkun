"""Fixed historical counterexample reconstruction and frozen closed-loop diagnosis."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t284_feedback_drift'
sys.path.insert(0,str(ROOT/'stage/experiments/t282_capped_balance'))
from balanced_learning import *
from t280_worker import generate,LimitedLog
sys.path.insert(0,str(ROOT/'stage/experiments/t278_fixed_goal_starts'))
from t278_guard import Budget
import ast,builtins,contextlib,gzip,io,json,multiprocessing as mp,os,shutil,signal,symtable,time,traceback


def save_payload(path,data,limit,meter):
    assert len(data)<=limit and not path.exists()
    meter.charge_payload(len(data));path.write_bytes(data)
    return dict(path=str(path),bytes=len(data),sha256=C.sha(path))


def actions(model,states):
    with torch.no_grad():return np.stack([model.step(torch.tensor(x[None]))[0][0].numpy() for x in states])


def diagnose(parent,candidate,a,b):
    n=min(len(a['states']),len(b['states']));xa=a['states'][:n];xb=b['states'][:n]
    assert np.array_equal(xa[0],xb[0])
    aa=actions(parent,xa);bb=actions(candidate,xb);ba=actions(candidate,xa);ab=actions(parent,xb)
    assert np.array_equal(aa,a['actions'][:n]) and np.array_equal(bb,b['actions'][:n])
    aa,bb,ba,ab=[x.astype(np.float64) for x in [aa,bb,ba,ab]]
    total=bb-aa;param_old=ba-aa;state_candidate=bb-ba;param_new=bb-ab;state_parent=ab-aa
    assert np.allclose(total,param_old+state_candidate,atol=1e-12,rtol=0)
    assert np.allclose(total,param_new+state_parent,atol=1e-12,rtol=0)
    times=.002*np.r_[0,np.cumsum(a['native_counts'].astype(np.int64))[:-1]][:n]
    assert np.array_equal(times,.002*np.r_[0,np.cumsum(b['native_counts'].astype(np.int64))[:-1]][:n])
    xy=np.linalg.norm(a['poses'][:n,:2]-b['poses'][:n,:2],axis=1)
    yaw=np.abs((a['poses'][:n,2]-b['poses'][:n,2]+np.pi)%(2*np.pi)-np.pi)
    arrays=dict(time_s=times,xy_error_m=xy,yaw_error_rad=yaw,total_action_delta=total,
        parameter_effect_at_parent_state=param_old,state_effect_under_candidate=state_candidate,
        parameter_effect_at_candidate_state=param_new,state_effect_under_parent=state_parent)
    crossings={}
    for threshold in [.001,.01,.05,.1]:
        ids=np.flatnonzero(xy>=threshold);crossings[str(threshold)]=None if not len(ids) else dict(index=int(ids[0]),time_s=float(times[ids[0]]),xy_m=float(xy[ids[0]]))
    norm=lambda x:np.linalg.norm(x,axis=1)
    ratio=float(np.linalg.norm(state_candidate)/max(np.linalg.norm(param_old),1e-12))
    reverse_ratio=float(np.linalg.norm(state_parent)/max(np.linalg.norm(param_new),1e-12))
    result=dict(common_decision_starts=n,initial120_exact=True,recorded_actions_exact=True,
        both_order_decomposition_identity_max_error=float(max(np.abs(total-param_old-state_candidate).max(),np.abs(total-param_new-state_parent).max())),
        parent_state_parameter_change_max=float(norm(param_old).max()),candidate_state_parameter_change_max=float(norm(param_new).max()),
        total_action_delta_max=float(norm(total).max()),feedback_to_parameter_energy_sqrt_ratio_on_parent=ratio,
        feedback_to_parameter_energy_sqrt_ratio_on_candidate=reverse_ratio,xy_error_max_m=float(xy.max()),yaw_error_max_rad=float(yaw.max()),
        first_xy_threshold_crossings=crossings,
        limitation='One selected historical rejected pair. Algebraic same-state parameter/state-feedback decomposition is order-dependent, not a dynamics Jacobian, Lyapunov certificate or causal intervention. No new success or training benefit.')
    return result,arrays


def main():
    assert not (OUT/'status.json').exists(),'Ended/started directories are never resumed'
    plan=json.loads((HERE/'protocol.json').read_text());base=json.loads(Path(plan['source_protocol']).read_text())
    start=time.time();meter=None;counts=mp.Array('q',1);raw=[];exports={};clone_calls=[];reason='COMPLETE';error=None;log=None
    def guard():
        assert time.time()<start+plan['wall_seconds'],'WALL'
        if meter is not None:meter.checkpoint()
    def status(state):C.save(OUT/'status.json',dict(state=state,pid=os.getpid(),updated_epoch=time.time(),main_epoch=start,absolute_deadline_epoch=start+plan['wall_seconds'],commands=0 if meter is None else meter.commands,physics_steps=int(counts[0]),clone_optimizer_calls=len(clone_calls),new_navigation_training_updates=0))
    def alarm(signum,frame):raise TimeoutError('T284 absolute wall cutoff')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(plan['wall_seconds'])
    try:
        assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
        for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
        meter=Budget(ROOT,OUT,plan,start_epoch=start);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[]));status('RECONSTRUCTING_EXISTING_REJECTED_CANDIDATE')
        oldrows=[json.loads(x) for x in Path(plan['source_cases']).read_text().splitlines()]
        items=[]
        for r in oldrows:
            if r['role']=='training' and r['id'].startswith('capped_task_balance_r0_'):
                with np.load(r['path']) as z:a={k:z[k].copy() for k in z.files}
                assert C.sha(r['path'])==r['sha256'] and all(S.digest(v)==r['array_hashes'][k] for k,v in a.items());items.append(dict(row=r,arrays=a))
        assert len(items)==16
        q=policy(base);opt=L.optimizer(q,base);shadow=copy.deepcopy(q);so=L.optimizer(shadow,base)
        def counted(optimizer,phase):
            original=optimizer.step
            def invoke(*args,**kwargs):
                clone_calls.append(phase);assert len(clone_calls)<=2
                C.save(OUT/'clone_optimizer_calls.json',dict(calls=clone_calls,new_navigation_training_updates=0))
                return original(*args,**kwargs)
            optimizer.step=invoke
        counted(opt,'historical_live_reconstruction');counted(so,'historical_shadow_reconstruction')
        parent=policy(base).eval().requires_grad_(False);x,y,off=anchors(base,parent);data=L.training_batch(items,q)
        cache=json.loads(Path(plan['source_cache']).read_text())['cache'];captured=[]
        def gate(actor):
            ph=S.digest(actor.state_dict())
            if ph==plan['candidate_policy_hash']:captured.append(copy.deepcopy(actor.state_dict()))
            return cache[ph]
        live,shadow_report=paired_balanced_update(q,opt,shadow,so,data,x,y,off,base,0,guard,gate,lambda phase,event:None)
        expected=[json.loads(l) for l in Path(plan['source_updates']).read_text().splitlines()][0]
        assert live==shadow_report==expected['updates'] and len(captured)==2 and S.digest(captured[0])==S.digest(captured[1])==plan['candidate_policy_hash']
        assert S.digest(q.state_dict())==base['policy_hash'] and S.digest(opt.state_dict())==expected['optimizer_hash']
        candidate=policy(base);candidate.load_state_dict(captured[0]);candidate.eval().requires_grad_(False)
        payload=dict(policy=candidate.state_dict(),historical_rejected_candidate=True,source_round=0,source_factor=.0625,policy_hash=plan['candidate_policy_hash'])
        buf=io.BytesIO();torch.save(payload,buf);blob=gzip.compress(buf.getvalue(),compresslevel=6,mtime=0)
        exports=save_payload(OUT/'historical_candidate.pt.gz',blob,plan['model_bytes'],meter)
        assert S.digest(L.read(exports['path'])['policy'])==plan['candidate_policy_hash']
        C.save(OUT/'reconstruction.json',dict(clone_calls=2,source_round_exact=True,candidate_policy_hash=plan['candidate_policy_hash'],parent_and_Adam_rollback_exact=True,export=exports))
        del items,data,q,shadow,opt,so,captured
        (OUT/'native').mkdir(exist_ok=True);log=LimitedLog(OUT/'native.log')
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):native,create=S.native_setup(OUT/'native')
        outputs={}
        for name,model in [('parent',parent),('candidate',candidate)]:
            for repeat in [False,True]:
                job=dict(plan['job'],id=name+('_replay' if repeat else ''),source_reset_id=plan['job']['id'],mode='deterministic',seed=0,aux_eta=0.)
                meter.begin(job['case']['step_cap']);status('NATIVE_'+job['id']);before=int(counts[0])
                with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):item=generate(job,model,native,create,start+plan['wall_seconds']-10,counts,0,before)
                meter.physics=int(counts[0]);row=item['row'];row.update(policy_hash=S.digest(model.state_dict()),replay=repeat)
                buf=io.BytesIO();np.savez_compressed(buf,**item['arrays']);row.update(save_payload(OUT/(job['id']+'.npz'),buf.getvalue(),plan['episode_bytes'],meter));raw.append(row);C.save(OUT/'cases.json',raw)
                assert row['complete'] and not row['error'],'Preserved partial/error; no continuation'
                ref=plan['references'][name]
                assert row['array_hashes']==ref['array_hashes'] and row['integration_sha256']==ref['integration_sha256']
                if repeat:
                    assert all(np.array_equal(v,outputs[name]['arrays'][k]) for k,v in item['arrays'].items())
                else:outputs[name]=item
                status('VERIFIED_'+job['id']);guard()
        analysis,arr=diagnose(parent,candidate,outputs['parent']['arrays'],outputs['candidate']['arrays'])
        buf=io.BytesIO();np.savez_compressed(buf,**arr);analysis['payload']=save_payload(OUT/'decomposition.npz',buf.getvalue(),plan['analysis_bytes'],meter)
        C.save(OUT/'analysis.json',analysis)
        assert sum(r['physics_steps'] for r in raw)==int(counts[0])<=plan['max_physics']
        for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    except BaseException:
        reason='FAILED';error=traceback.format_exc();C.save(OUT/'failure.json',dict(traceback=error,commands=0 if meter is None else meter.commands,physics_steps=int(counts[0])))
    finally:
        signal.alarm(0)
        if log is not None:log.close()
        if meter is not None:meter.close()
    result=dict(state=reason,commands=0 if meter is None else meter.commands,physics_steps=int(counts[0]),complete_payloads=len(raw),clone_optimizer_calls=len(clone_calls),navigation_training_updates=0,
        all_workers_stopped=True,elapsed_s=time.time()-start,main_epoch=start,absolute_deadline_epoch=start+plan['wall_seconds'],own_bytes=S.used(OUT),global_bytes=S.used(ROOT/'stage/runs'),
        candidate_export=exports,all_four_historical_reconstructions_exact=reason=='COMPLETE',no_new_six_m_success=True,automatic_next_training=False)
    C.save(OUT/'summary.json',result);C.save(OUT/'status.json',result);print(json.dumps(result),flush=True)


def preflight():
    p=json.loads((HERE/'protocol.json').read_text());assert not (OUT/'status.json').exists()
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    table=symtable.symtable(Path(__file__).read_text(),str(__file__),'exec');missing=set()
    def walk(t):
        for s in t.get_symbols():
            if s.is_referenced() and s.is_global() and s.get_name() not in globals() and not hasattr(builtins,s.get_name()):missing.add(s.get_name())
        for child in t.get_children():walk(child)
    walk(table);assert not missing,missing
    assert p['max_commands']*p['job']['case']['step_cap']<=p['max_physics']
    assert p['payload_bytes']+p['metadata_bytes']<=p['final_bytes']
    used=S.used(ROOT/'stage/runs');assert used-S.used(OUT)+p['final_bytes']<=p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    C.save(OUT/'preflight.json',dict(state='PASS',physics_steps=0,optimizer_calls=0,native_environments_created=0,all_nested_global_names_resolve=True,
        registered_protocol_sha256=C.sha(HERE/'protocol.json'),full_reserved_global_bytes=used-S.used(OUT)+p['final_bytes']))
    print('T284 PREFLIGHT PASS: zero native and optimizer')


if __name__=='__main__':
    if sys.argv[1:]==['--preflight']:preflight()
    else:assert not sys.argv[1:];main()
