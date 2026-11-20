"""Finite-sample old-outcome score geometry; no optimizer or simulator."""
from pathlib import Path
import fcntl,gzip,itertools,json,os,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t670_outcome_gradient_gate';DOC=ROOT/'stage/docs/weekend_20261120_23/t670'
os.sched_setaffinity(0,{14})
sys.path.insert(0,str(ROOT/'stage/experiments/t665_remaining_verification'));import t665_runtime as T
from status_io import save
M,torch,np=T.M,T.torch,T.np
read,sha,digest=T.read,T.sha,T.digest


def guard(p,start,limit):
    assert time.time()<min(start+limit,p['absolute_all_stop'])-10
    while True:
        try:T.Wait.Raw.resources();return
        except AssertionError as e:
            if not T.Wait.is_overlap(e):raise
            assert time.time()<min(start+limit,p['absolute_all_stop'])-10;time.sleep(.25)


def verify(p):
    assert p['authorization_approved'] and sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    for f,h in p['inputs'].items():assert sha(f)==h,f
    assert T.R.used(ROOT/'stage/runs')<p['global_bytes'] and T.R.used(OUT)<p['max_bytes']
    assert T.R.D.B.shutil.disk_usage(ROOT).free>40*2**30
    for folder in p['sealed_sources']:
        s=read(Path(folder)/'SEALED.json');assert s['all_scientific_processes_stopped']
        assert all(not Path('/proc',str(pid)).exists() for pid in s['scientific_pids'])


def outcome_surrogate(q,group,p):
    value=next(q.parameters()).new_zeros(())
    for g in group:
        logp=M.P.L.density(M.location(q,g),g['raw'],g['atom'],p['raw_sigma'])
        value+=((logp-g['old']).exp()-1).sum()*g['adv']/(len(group)*p['score_scale'])
    return value


def flatten_gradient(loss,variables):
    return torch.cat([v.reshape(-1) for v in torch.autograd.grad(loss,variables,allow_unused=False)]).detach().numpy().copy()


def group(q,rows,p):
    ys=[M.outcome(r,'collision_cost') for r in rows];n=len(ys);assert n>1
    result=[]
    for i,row in enumerate(rows):
        assert row['policy_hash']==q.phash() and sha(row['path'])==row['sha256']
        assert row['full'] and row['seed'] is not None and not row.get('teacher')
        a=T.R.E.N.S.arrays(row['path']);assert not a['ownership'].any() and a['stochastic'].all()
        g=M.features(q,a,row['job_id'],row['sha256'])
        g.update(raw=torch.tensor(a['raw_actions']),atom=torch.tensor(a['atom']),old=torch.tensor(a['logp']),
            adv=ys[i]-(sum(ys)-ys[i])/(n-1))
        with torch.no_grad():
            assert float((M.P.L.density(M.location(q,g),g['raw'],g['atom'],p['raw_sigma'])-g['old']).abs().max())<1e-9
            assert float((M.location(q,g)-torch.tensor(a['locations'])).abs().max())<1e-10
        result.append(g)
    return result,ys


def project(d0,A,tolerance=1e-12):
    """Nearest Euclidean displacement with A d >= 0, finite active sets."""
    m=len(A);candidates=[]
    for mask in range(1<<m):
        active=[j for j in range(m) if mask&(1<<j)]
        if active:
            B=A[active];gram=B@B.T
            lam=np.linalg.lstsq(gram,-B@d0,rcond=1e-12)[0]
            if (lam < -tolerance).any():continue
            d=d0+B.T@lam
            if np.abs(B@d).max()>1e-10:continue
        else:lam=np.zeros(0);d=d0.copy()
        residual=A@d
        if (residual < -tolerance).any():continue
        candidates.append((float(np.square(d-d0).sum()),active,lam,d,residual))
    assert candidates,'No cone solution found although zero is feasible'
    best=min(candidates,key=lambda x:x[0]);distance,active,lam,d,residual=best
    dual=np.zeros(m);dual[active]=lam
    assert np.linalg.norm(d-d0-A.T@dual)<1e-10
    assert np.abs(dual*residual).max()<1e-10
    return d,dict(active=active,dual=dual.tolist(),residual=residual.tolist(),distance=distance,
        feasible_active_sets=len(candidates),stationarity_error=float(np.linalg.norm(d-d0-A.T@dual)),
        complementarity_error=float(np.abs(dual*residual).max()))


def move(q,variables,before,d,scale=1.):
    k=0
    with torch.no_grad():
        for v,b in zip(variables,before):
            v.copy_(b+torch.tensor(d[k:k+v.numel()]).reshape_as(v)*scale);k+=v.numel()
    assert k==len(d)


def compute(p,limit):
    start=time.time();guard(p,start,limit);science=dict(read(p['source_protocol']),initial=p['initial'],raw_sigma=[.05,.05])
    T.setup(science);q=M.L.frozen(science,p['initial']['path']);q.requires_grad_(False)
    q.drive_adapter.requires_grad_(True);q.steer_adapter.requires_grad_(True)
    variables=[v for v in q.parameters() if v.requires_grad];assert sum(v.numel() for v in variables)==11906
    model_hash=q.phash();oldgroups=[];oldys=[];oldgrad=[]
    rows=read(p['old_summary'])['rows'];rows=[r for r in rows if not r.get('exact')]
    for key in p['old_ids']:
        guard(p,start,limit);g,y=group(q,[r for r in rows if r['job_id']==key],science)
        assert len(g)==8 and len(set(y))>1
        oldgroups.append(g);oldys.append(y);oldgrad.append(flatten_gradient(outcome_surrogate(q,g,science),variables))
    fieldrows=read(p['fresh_rounds'])[0]['training_rows']
    fieldgrad=[];fieldys=[];fieldkeys=[]
    for job in science['field_jobs']:
        guard(p,start,limit);rr=[r for r in fieldrows if r['job_id']==job['id']]
        assert len(rr)==2
        g,y=group(q,rr,science);fieldkeys.append(job['id']);fieldys.append(y)
        fieldgrad.append(flatten_gradient(outcome_surrogate(q,g,science),variables))
    refs=M.references(science,q,lambda:guard(p,start,limit));keep=M.pack(refs)
    loss,per_task=M.retention(q,keep,False);kg=flatten_gradient(loss,variables)
    oldgrad=np.stack(oldgrad);fieldgrad=np.stack(fieldgrad);gn=fieldgrad.mean(0)
    # Match a prospective soft joint objective over 24 field + 4 old tasks.
    rewardgrad=(fieldgrad.sum(0)+oldgrad.sum(0))/(len(fieldgrad)+len(oldgrad))
    gl=-rewardgrad+science['replay_weight']*kg;clip=min(1.,1./max(np.linalg.norm(gl),1e-30));gc=gl*clip
    d0=-science['fit_lr']*gc/(np.abs(gc)+p['Adam_eps'])
    norms=np.linalg.norm(oldgrad,axis=1);A=oldgrad/np.maximum(norms[:,None],1e-30)
    d,qp=project(d0,A)
    before=[v.detach().clone() for v in variables];finite=[];nonlinear={}
    try:
        direction=d/max(np.linalg.norm(d),1e-30)
        for i,g in enumerate(oldgroups):
            guard(p,start,limit);h=p['finite_parameter_epsilon']
            move(q,variables,before,direction,h);plus=float(outcome_surrogate(q,g,science).detach())
            move(q,variables,before,direction,-h);minus=float(outcome_surrogate(q,g,science).detach())
            measured=(plus-minus)/(2*h);expected=float(oldgrad[i]@direction)
            numeric=abs(measured-expected)<=p['numeric_atol']+p['numeric_rtol']*abs(expected)
            finite.append(dict(id=p['old_ids'][i],autograd_dot=expected,finite_difference=measured,error=abs(measured-expected),passed=numeric))
        for label,step in [('soft_Adam_formula',d0),('old_outcome_projection',d)]:
            move(q,variables,before,step)
            nonlinear[label]=[float(outcome_surrogate(q,g,science).detach()) for g in oldgroups]
    finally:move(q,variables,before,np.zeros_like(d0))
    assert q.phash()==model_hash and all(v.grad is None for v in q.parameters())
    checks=dict(all_four_nonzero=bool((norms>1e-12).all()),numeric_all=all(x['passed'] for x in finite),
        KKT=bool(min(qp['residual'])>=-1e-12),direction_changed=bool(np.linalg.norm(d-d0)>1e-12),
        raw_has_old_conflict=bool((oldgrad@d0 < -1e-12).any()),
        new_goal_ascent=bool(gn@d>1e-12),projection_nonzero=bool(np.linalg.norm(d)>1e-12))
    arrays=dict(old_gradients=oldgrad,field_gradients=fieldgrad,nominal_gradient=kg,soft_Adam_displacement=d0,projected_displacement=d)
    result=dict(state='PASS' if all(checks.values()) else 'SCIENTIFIC_GATE_FAILED',checks=checks,
        source_policy_hash=model_hash,old_ids=p['old_ids'],old_outcomes=oldys,old_gradient_norms=norms.tolist(),
        old_Gram=(A@A.T).tolist(),field_ids=fieldkeys,field_outcomes=fieldys,
        mixed_field_tasks=sum(len(set(y))>1 for y in fieldys),new_gradient_norm=float(np.linalg.norm(gn)),
        nominal_gradient_norm=float(np.linalg.norm(kg)),loss_gradient_norm=float(np.linalg.norm(gl)),clip_scale=clip,
        soft_old_directional_change=(oldgrad@d0).tolist(),projected_old_directional_change=(oldgrad@d).tolist(),
        soft_new_directional_change=float(gn@d0),projected_new_directional_change=float(gn@d),
        QP=qp,numeric=finite,nonlinear_sample_surrogate=nonlinear,
        nonlinear_check_not_actual_rollout=True,Adam_formula_not_optimizer_call=True,
        arrays_hashes={k:digest(v) for k,v in arrays.items()},new_native=0,new_optimizer=0)
    return result,arrays


def execute(action):
    p=read(OUT/'protocol.json');verify(p);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time();result,arrays=compute(p,p['main_seconds'] if action=='main' else p['audit_seconds'])
    if action=='main':
        with (OUT/'gradients.npz').open('xb') as f:np.savez_compressed(f,**arrays)
        save(OUT/'RESULTS.json',result)
    else:
        assert result==read(OUT/'RESULTS.json')
        with np.load(OUT/'gradients.npz',allow_pickle=False) as stored:
            assert set(stored.files)==set(arrays)
            assert all(np.array_equal(stored[k],v) for k,v in arrays.items())
        save(OUT/'audit.json',dict(state='PASS',all_gradients_and_KKT_recomputed_exact=True,results_digest=digest(result),
            numeric_gate=result['state'],new_native=0,new_optimizer=0))
    save(OUT/(action+'_receipt.json'),dict(state='COMPLETE',pid=os.getpid(),elapsed_s=time.time()-started,new_native=0,new_optimizer=0))
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':execute(sys.argv[1])
