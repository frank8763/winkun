"""One final bounded, function-constrained policy repair; Stage only.

Cached genuine on-policy score gradient plus verified recovery supervision.
The inner optimizer controls a 66-dimensional output-layer displacement.
All deployed feature/core tensors stay fixed. No runtime teacher or route.
"""
from pathlib import Path
import sys,os,json,time,copy,hashlib,gzip,io,shutil,subprocess,fcntl,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t440_final_functional_constraint';DOC=ROOT/'stage/docs/review_20261109'
sys.path.insert(0,str(ROOT/'stage/experiments/t432_offline_night'))
import t432_core as C
E,np,torch=C.E,C.np,C.torch
read,save,sha,digest=C.read,C.save,C.sha,C.digest

def lines(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]
def used():return E.N.K.S.used(ROOT/'stage/runs')
def verify(p):
    assert sha(OUT/'protocol.json')==sha(HERE/'protocol.json')
    assert time.time()<p['deadline_all'] and used()<=2**30 and E.N.K.S.used(OUT)<=p['max_bytes']
    assert shutil.disk_usage(ROOT).free>=40*2**30
    for f,h in p['inputs'].items():assert sha(f)==h,f
def frozen(p,model=None):
    q=C.frozen(p,model or p['student_start_model']).requires_grad_(False)
    assert not q.sample
    return q
def fixed(q):return digest({k:v for k,v in q.state_dict().items() if k not in ['extra.2.weight','extra.2.bias']})
def gradient(q):
    q.requires_grad_(False);q.extra[-1].requires_grad_(True)
    return [q.extra[-1].weight,q.extra[-1].bias]
def extract(q,x,b=None):
    tape=[]
    def hook(module,args):tape.append(args[0].detach().reshape(-1,32).clone())
    handle=q.extra[-1].register_forward_pre_hook(hook)
    try:
        with torch.no_grad():y=q.sequence(x,q.base_actions(x) if b is None else b).detach().clone()
    finally:handle.remove()
    a=torch.cat(tape);assert a.shape==(len(y),32)
    return torch.cat([a,torch.ones((len(a),1),dtype=a.dtype)],1),y
def constrained(v,A,epsilon):
    # tanh is globally 1-Lipschitz: raw-location change <=1.25*logit change.
    scale=torch.maximum(v.new_tensor(1.),1.25*(A@v).abs().max()/epsilon)
    return v/scale
def predict(x,y,d):
    z=torch.atanh((y/1.25).clamp(-1+1e-15,1-1e-15))
    return 1.25*torch.tanh(z+x@d)
def executed_loss(y,a):
    return torch.where(a==1,torch.relu(1-y).square(),torch.where(a==-1,torch.relu(y+1).square(),(y-a).square())).mean()
def load_data(p,q,guard):
    data=read(p['dataset_file'])['result'];anchors=data['anchors_c'];targets=data['targets']
    assert len(anchors)==62 and len(targets)==7
    preserve=[];features=[];proofs=[]
    for entry in anchors:
        guard();r=entry['row'];assert sha(entry['path'])==entry['sha256']
        a=E.N.S.arrays(entry['path']);assert digest(a)==r['array_digest']
        assert r['success'] and not r['collision'] and r['policy_hash']==p['expected_policy_hash']
        x=torch.tensor(a['states']);A,y=extract(q,x)
        assert float((y-torch.tensor(a['locations'])).abs().max())<1e-10
        preserve.append((A,y));features.append(A);proofs.append(dict(kind='current_success',id=r['job_id'],rows=len(A),sha256=entry['sha256']))
    old=E.N.old_data(q,E.N.science(p),guard);assert len(old)==211
    for item in old:
        guard();A,y=extract(q,item['x'],item['b']);preserve.append((A,y));features.append(A)
        proofs.append(dict(kind='historical_self_target',id=item['id'],rows=len(A)))
    recover=[]
    for entry in targets:
        guard();assert sha(entry['path'])==entry['sha256'] and entry['exact']
        a=E.N.S.arrays(entry['path']);assert digest(a)==entry['row']['array_digest']
        assert a['end_flags'][-1,1] and not a['end_flags'][:,0].any()
        first=entry['first'];assert not a['ownership'][:first].any() and a['ownership'][first:].all()
        A,y=extract(q,torch.tensor(a['states']));recover.append((A[first:],y[first:],torch.tensor(a['actions'][first:],dtype=torch.float64)))
        proofs.append(dict(kind='verified_recovery_suffix',id=entry['id'],first=first,rows=len(A)-first,sha256=entry['sha256']))
    train=[r for r in lines(p['sampling_cases']) if r['role']=='TRAIN_0'];assert len(train)==4
    items=[]
    for r in train:
        guard();assert r['policy_hash']==q.phash() and sha(r['path'])==r['sha256']
        a=E.N.S.arrays(r['path']);assert digest(a)==r['array_digest']
        E.audit_array(q,a,p,0,r['seed']);items.append(a)
    variables=gradient(q);batch,returns,errors=E.L.blocks(q,items,0,p)
    q.zero_grad(set_to_none=True);loss=0.
    for x,b,raw,atom,oldlog,advantage in batch:
        ratio=(E.L.logp(q.sequence(x,b),raw,atom,p)-oldlog).exp()
        coefficient=advantage/(4*400)
        loss=loss-torch.minimum(ratio*coefficient,ratio.clamp(1-p['ppo_clip'],1+p['ppo_clip'])*coefficient).sum()
    grads=torch.autograd.grad(loss,variables)
    pg=torch.cat([grads[0].T,grads[1][None]],0).detach();assert pg.shape==(33,2) and torch.isfinite(pg).all()
    q.requires_grad_(False);q.zero_grad(set_to_none=True)
    A=torch.cat(features);assert A.shape[1]==33 and torch.isfinite(A).all()
    return dict(A=A,preserve=preserve,recover=recover,pg=pg,
        proof=dict(support_histories=273,recoveries=7,support_rows=len(A),onpolicy_cached_episodes=4,
            onpolicy_reused_from_completed_T437_not_new_samples=True,PG_shape=list(pg.shape),PG_sha256=digest(pg),
            PG_norm=float(pg.norm()),returns=returns.tolist(),density_errors=errors,
            support_matrix_sha256=digest(A),sources=proofs))
def solve(p,g,guard,event=None):
    v=torch.nn.Parameter(torch.zeros((33,2),dtype=torch.float64));opt=torch.optim.Adam([v],lr=p['inner_lr'],foreach=False,fused=False)
    base=float(sum(executed_loss(y,a) for x,y,a in g['recover'])/7);trace=[]
    for k in range(p['inner_steps']):
        guard();opt.zero_grad(set_to_none=True);d=constrained(v,g['A'],p['raw_drift_cap'])
        recovery=sum(executed_loss(predict(x,y,d),a) for x,y,a in g['recover'])/7
        linear=(g['pg']*d).sum();loss=recovery+linear+p['ridge']*d.square().sum()
        assert torch.isfinite(loss);loss.backward()
        if event:event('BEFORE',k,v,opt)
        opt.step()
        if event:event('AFTER',k,v,opt)
        with torch.no_grad():
            d=constrained(v,g['A'],p['raw_drift_cap']);maxlogit=float((g['A']@d).abs().max())
            assert 1.25*maxlogit<=p['raw_drift_cap']+1e-12
            trace.append(dict(k=k,objective=float(loss),recovery=float(recovery),linear_PG=float(linear),max_logit_change=maxlogit))
    with torch.no_grad():
        d=constrained(v,g['A'],p['raw_drift_cap']).detach().clone()
        loss=float(sum(executed_loss(predict(x,y,d),a) for x,y,a in g['recover'])/7)
        maxraw=max(float((predict(x,y,d)-y).abs().max()) for x,y in g['preserve'])
        slope=float((g['pg']*d).sum());assert maxraw<=p['raw_drift_cap']+1e-10
    feasible=loss<=base*p['recovery_ratio_gate'] and slope<0
    return d,dict(state='PASS' if feasible else 'FAIL_OFFLINE_FEASIBILITY',initial_recovery_loss=base,final_recovery_loss=loss,
        recovery_ratio=loss/base,linear_PG_change=slope,max_raw_support_change=maxraw,
        global_logit_Lipschitz_bound=1.25*float((g['A']@d).abs().max()),delta_sha256=digest(d),
        optimizer_hash=digest(opt.state_dict()),inner_steps=p['inner_steps'],new_online_RL_steps=0,
        cached_onpolicy_gradient_evaluations=1,trace=trace)
def apply(q,d):
    with torch.no_grad():q.extra[-1].weight.add_(d[:32].T);q.extra[-1].bias.add_(d[32])
def export(q,path,extra=None):
    buf=io.BytesIO();learned={k:v.detach().clone() for k,v in q.state_dict().items()}
    torch.save(dict(state=dict(learned=learned),policy_hash=q.phash(),memory_on=True,
        architecture='unchanged_single_Actor_final66_function_constrained_repair',extra=extra),buf)
    data=gzip.compress(buf.getvalue(),mtime=0);assert len(data)<=512*1024;Path(path).write_bytes(data)
    return dict(path=str(path),sha256=sha(path),policy_hash=q.phash(),learned_sha256=digest(learned))
def prepare():
    assert not OUT.exists();p=copy.deepcopy(read(ROOT/'stage/runs/t437_joint_closed_loop_rl/protocol.json'))
    prior=ROOT/'stage/runs/t437_joint_closed_loop_rl/fresh_rl'
    hard=p['deadline_all'];assert time.time()+2400+300+600<hard
    p.update(id='T440_FINAL_FUNCTIONAL_CONSTRAINED_POLICY_REPAIR',out=str(OUT),max_bytes=40*2**20,
        max_physics=14000000,max_commands=520,main_seconds=2400,audit_seconds=300,
        dataset_file=str(ROOT/'stage/runs/t432_offline_night/data/summary.json'),sampling_cases=str(prior/'cases.jsonl'),
        inner_steps=64,inner_lr=.001,ridge=.001,raw_drift_cap=.02,recovery_ratio_gate=.99,
        deadline_all=hard,deadline_native=hard-900,full_bytes=384*1024,
        new_authorization='Nov9 user explicitly requests next and stop this Stage line if it still fails. Exactly one constrained candidate, no automatic fallback.',
        cached_onpolicy_not_new_online_data=True,no_Adam_on_deployed_parameters=True,
        sole_inner_variable='33x2 final-layer displacement, radial exact support bound',
        original_reward_unchanged=True,no_success_threshold_change=True,stop_line_if_not_complete_gate=True,
        scientific_scope='One cached true policy-gradient proximal proposal plus verified recovery, fixed representation. Not new online training, not claimed original algorithm or safety proof.')
    paths=set(p['inputs'])|{str(f) for f in HERE.glob('*.py')}|{str(DOC/'T440_PLAN.md'),p['dataset_file'],p['sampling_cases']}
    for e in read(p['dataset_file'])['result']['anchors_c']:paths.add(e['path'])
    for r in lines(p['sampling_cases']):
        if r['role']=='TRAIN_0':paths.add(r['path'])
    p['inputs']={x:sha(x) for x in sorted(paths)}
    reserve=read(ROOT/'stage/docs/overnight_20261108_relay/RESERVE_CANDIDATES.json')
    protected=json.dumps(p);assert not any(x['original_path'] in protected for x in reserve['rows'])
    after=used()-reserve['expected_payload_released_bytes']+2**20
    assert after+p['max_bytes']<=2**30
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T440_BUDGET.json',dict(state='REGISTERED_STORAGE_PENDING_NOT_LAUNCHED',global_before=used(),
        lossless_expected_release=reserve['expected_payload_released_bytes'],global_after_storage_conservative=after,
        full_reserved=after+p['max_bytes'],max_bytes=p['max_bytes'],max_physics=p['max_physics'],max_commands=520,
        full78_reserve_bytes=78*p['full_bytes'],metadata_and_models_and_compact_reserve=10*2**20,
        deadline_all=hard,main_seconds=2400,audit_seconds=300,inner_Adam_live=64,inner_Adam_shadow=64,
        independent_historical_inner_Adam=128,new_online_training_episodes=0))
    assert 78*p['full_bytes']+10*2**20<=p['max_bytes']
    print('T440_REGISTERED_STORAGE_PENDING',flush=True)
def preflight():
    p=read(OUT/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);C.ban_learning()
    q=frozen(p);before=digest(q.state_dict());x=torch.tensor(E.N.S.arrays(read(p['dataset_file'])['result']['anchors_c'][0]['path'])['states'])
    A,y=extract(q,x)
    d=torch.linspace(-1,1,66,dtype=torch.float64).reshape(33,2)*1e-4
    other=frozen(p);apply(other,d);_,actual=extract(other,x)
    assert float((predict(A,y,d)-actual).abs().max())<1e-10
    for scale in [0.,1.,1e4]:
        v=d*scale;bounded=constrained(v,A,p['raw_drift_cap']);assert float((predict(A,y,bounded)-y).abs().max())<=p['raw_drift_cap']+1e-12
    assert digest(q.state_dict())==before and fixed(q)==fixed(other)
    save(OUT/'preflight.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',feature_formula_real_history_error=float((predict(A,y,d)-actual).abs().max()),
        head_only66_verified=True,all_other_tensors_fixed=True,large_and_zero_displacements_bounded=True,all_original_inputs_SHA_verified=True,
        native=0,optimizer=0,no_claim_of_closed_loop_safety=True))
    print('T440_PREFLIGHT_PASS',flush=True)
class Native:
    def __init__(self,p):
        self.p=p;self.pool=None;self.start=time.time();self.counts=C.mp.get_context('spawn').RawArray('q',5);self.commands=0;self.rows=[];self.replays=[]
        self.deadline=min(self.start+p['main_seconds'],p['deadline_native']);self.lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def guard(self):
        E.L.check_stop();assert time.time()<self.deadline-5 and sum(self.counts)<=self.p['max_physics'] and self.commands<=self.p['max_commands'];verify(self.p)
        save(OUT/'status.json',dict(state='FINAL_CANDIDATE_NATIVE',epoch=time.time(),pid=os.getpid(),physics=sum(self.counts),commands=self.commands,rows=len(self.rows)))
    def start_pool(self):
        E.set_out(OUT);cfg=dict(self.p,out=str(OUT),native_stop_epoch=self.deadline,max_physics=self.p['max_physics'])
        self.pool=C.Pool(cfg,self.counts,self.guard);self.commands=4
    def request(self,jobs,model,role,full=False,refs=None):
        output=[]
        for offset in range(0,len(jobs),4):
            batch=jobs[offset:offset+4];self.guard();assert sum(self.counts)+sum(j['case']['step_cap'] for j in batch)<=self.p['max_physics']
            self.commands+=len(batch);requests=[dict(job=j,model=model['path'],mode='student',seed=None) for j in batch]
            for i,((a,r),req) in enumerate(zip(self.pool.request(requests),requests)):
                row=dict(r,job_id=req['job']['id'],request=req,role=role,seed=None,loaded_model_sha256=sha(model['path']),physics_steps=int(a['native_counts'].sum()))
                if refs is not None:
                    old,prev=refs[offset+i];assert set(a)==set(old)
                    for k in a:assert E.equal(a[k],old[k]),(role,k)
                    assert row['array_digest']==prev['array_digest'];row.update(exact=True,all_arrays_exact=True,independent_sample=False,reference_path=prev['path']);self.replays.append(row);file='replays.jsonl'
                else:
                    keep=full(req) if callable(full) else full
                    path=OUT/'episodes'/f'{len(self.rows):04}.npz';path.parent.mkdir(exist_ok=True)
                    E.N.write_npz(path,E.N.S.packed(a) if keep else {k:a[k] for k in ['actions','native_counts','end_flags']},self.p['full_bytes'] if keep else 5*1024)
                    row.update(path=str(path),sha256=sha(path),full=bool(keep));self.rows.append(row);file='cases.jsonl'
                with (OUT/file).open('a') as f:f.write(json.dumps(row)+'\n')
                output.append((a,row))
        return output
    def close(self):
        if self.pool:self.pool.close()
        fcntl.flock(self.lock,fcntl.LOCK_UN);self.lock.close()
def fit():
    p=read(OUT/'protocol.json');verify(p);assert not (OUT/'fit_summary.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);E.N.offline();E.L.install_stop_handlers()
    start=time.time();limit=min(start+300,p['deadline_native']-1800)
    def guard():assert time.time()<limit-5;verify(p)
    q=frozen(p);original=fixed(q);g=load_data(p,q,guard);save(OUT/'dataset_proof.json',g['proof'])
    events=[]
    def event(which):
        def emit(where,k,v,opt):
            row=dict(which=which,where=where,k=k,variable_sha256=digest(v.detach()),optimizer_sha256=digest(opt.state_dict()),gradient_sha256=digest(v.grad))
            events.append(row)
            with (OUT/'inner_optimizer_calls.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        return emit
    d,result=solve(p,g,guard,event('live'));other,shadow=solve(p,g,guard,event('shadow'))
    assert result==shadow and torch.equal(d,other)
    assert [dict(r,which=None) for r in events if r['which']=='live']==[dict(r,which=None) for r in events if r['which']=='shadow']
    initial=export(q,OUT/'initial.pt.gz');apply(q,d);assert fixed(q)==original
    # Actual deployed sequence evaluation verifies the analytic bound, not only the radial surrogate.
    actual_max=0.;data=read(p['dataset_file'])['result']
    source=frozen(p)
    for item in data['anchors_c']:
        guard();a=E.N.S.arrays(item['path']);x=torch.tensor(a['states'])
        _,y=extract(q,x);actual_max=max(actual_max,float((y-torch.tensor(a['locations'])).abs().max()))
    assert actual_max<=p['raw_drift_cap']+1e-10
    model=export(q,OUT/'proposed.pt.gz',dict(functional_constraint=True,delta_sha256=digest(d)))
    buf=io.BytesIO();np.savez_compressed(buf,delta=d.numpy());(OUT/'delta.npz').write_bytes(buf.getvalue())
    save(OUT/'fit_summary.json',dict(state='COMPLETE',elapsed_s=time.time()-start,initial_model=initial,proposed_model=model,
        fit=result,actual_anchor_max_raw_change=actual_max,all_other_tensors_exact=True,inner_live=64,inner_shadow=64,new_online_RL=0,
        native_allowed=result['state']=='PASS',cached_true_policy_gradient=True,not_claimed_novel=True))
    print('T440_OFFLINE_GATE',result['state'],result['recovery_ratio'],result['linear_PG_change'],flush=True)
def native():
    p=read(OUT/'protocol.json');verify(p);fit=read(OUT/'fit_summary.json');assert fit['native_allowed'] and not (OUT/'native_summary.json').exists()
    C.ban_learning();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);E.N.offline();E.L.install_stop_handlers()
    rt=Native(p);save(OUT/'native_started.json',dict(pid=os.getpid(),epoch=rt.start,deadline=rt.deadline));state='FAILED';error=None
    result=dict(wide54=None,old275=None,fresh16=None,reload78=None,six_gate=False,eight_started=False)
    try:
        rt.start_pool();model=fit['proposed_model'];fresh_ids={j['id'] for j in p['fresh_jobs']};reference_ids={j['id'] for j in p['reload_jobs']}-fresh_ids;assert len(reference_ids)==62
        dev=rt.request(p['main_jobs']+p['old275_jobs'],model,'FINAL329',lambda req:req['job']['id'] in reference_ids)
        rows=[r for a,r in dev];result.update(wide54=E.tally(rows[:54]),old275=E.tally(rows[54:]),development_all_safe=all(r['success'] and not r['collision'] for r in rows))
        if result['development_all_safe']:
            refs={r['job_id']:(a,r) for a,r in dev if r['job_id'] in reference_ids};assert len(refs)==62
            for f,h in p['fresh_ledger'].items():assert sha(f)==h
            fresh=rt.request(p['fresh_jobs'],model,'FRESH16',True);result['fresh16']=E.tally([r for a,r in fresh])
            if all(r['success'] and not r['collision'] for a,r in fresh):
                refs.update({r['job_id']:(a,r) for a,r in fresh})
                rt.request(p['reload_jobs'],model,'RELOAD78',refs=[refs[j['id']] for j in p['reload_jobs']]);result.update(six_gate=True,reload78=dict(exact=78,all_arrays_exact=True,independent_sample=False))
        state='COMPLETE'
    except BaseException:error=traceback.format_exc();save(OUT/'native_error.json',dict(error=error,no_restart=True))
    finally:
        rt.close();save(OUT/'native_summary.json',dict(state=state,error=error,elapsed_s=time.time()-rt.start,physics=sum(rt.counts),commands=rt.commands,result=result,all_workers_stopped=True))
    assert state=='COMPLETE',error
    print('T440_NATIVE_COMPLETE',result,flush=True)
def audit():
    p=read(OUT/'protocol.json');verify(p);assert not (OUT/'audit.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);E.N.offline()
    start=time.time();fit=read(OUT/'fit_summary.json');q=frozen(p)
    def guard():assert time.time()<min(start+p['audit_seconds'],p['deadline_all']-600)-5;verify(p)
    g=load_data(p,q,guard);assert g['proof']==read(OUT/'dataset_proof.json')
    expected=lines(OUT/'inner_optimizer_calls.jsonl');events=[]
    for which in ['live','shadow']:
        def event(where,k,v,opt):
            row=dict(which=which,where=where,k=k,variable_sha256=digest(v.detach()),optimizer_sha256=digest(opt.state_dict()),gradient_sha256=digest(v.grad))
            assert row==expected[len(events)];events.append(row)
        d,r=solve(p,g,guard,event);assert r==fit['fit']
    assert len(events)==256
    apply(q,d);assert q.phash()==fit['proposed_model']['policy_hash'] and digest(q.state_dict())==fit['proposed_model']['learned_sha256']
    assert sha(fit['proposed_model']['path'])==fit['proposed_model']['sha256']
    native=read(OUT/'native_summary.json') if (OUT/'native_summary.json').exists() else None
    rows=lines(OUT/'cases.jsonl') if (OUT/'cases.jsonl').exists() else [];replays=lines(OUT/'replays.jsonl') if (OUT/'replays.jsonl').exists() else []
    if native:
        assert native['state']=='COMPLETE' and native['all_workers_stopped']
        assert not Path('/proc',str(read(OUT/'native_started.json')['pid'])).exists()
        assert all(not Path('/proc',str(pid)).exists() for pid in read(OUT/'workers.json')['workers'])
        import importlib.util
        spec=importlib.util.spec_from_file_location('stage440_pinned_policy_audit',ROOT/'stage/experiments/t427_fixed_representation/audit.py')
        H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
        for row in rows:
            guard();assert sha(row['path'])==row['sha256'];a=E.N.S.arrays(row['path'])
            if row['full']:
                assert digest(a)==row['array_digest'];H.full_policy(C,a,row,p)
            else:
                for k in a:assert digest(a[k])==row['array_hashes'][k]
        assert sum(r['physics'] for r in rows+replays)==native['physics'] and 4+len(rows)+len(replays)==native['commands']
        assert native['result']['six_gate']==(len(replays)==78)
    else:assert not fit['native_allowed']
    save(OUT/'audit.json',dict(state='PASS',elapsed_s=time.time()-start,native=0,new_training=0,discarded_historical_inner_Adam=128,
        actual_variables_gradients_optimizer_events_exact=True,functional_bound_verified=True,
        cached_onpolicy_gradient_not_new_exploration=True,six_gate=native['result']['six_gate'] if native else False,
        compact_array_hash_limit_declared=True))
    print('T440_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
