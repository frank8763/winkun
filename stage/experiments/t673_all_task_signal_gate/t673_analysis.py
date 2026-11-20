"""All-task finite-sample signal audit; no optimizer, simulator or candidate export."""
from pathlib import Path
import fcntl,json,os,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t673_all_task_signal_gate';DOC=ROOT/'stage/docs/weekend_20261120_23/t673'
sys.path.insert(0,str(ROOT/'stage/experiments/t670_outcome_gradient_gate'))
import t670_analysis as A
from status_io import save
M,T,torch,np=A.M,A.T,A.torch,A.np
read,sha,digest=A.read,A.sha,A.digest

def dataset(p):
    groups={}
    for f in p['summaries']:
        s=read(f);assert s['state']=='COMPLETE'
        for r in s['rows']:
            assert r['role']=='frozen_fresh' and not r.get('exact')
            groups.setdefault(r['job_id'],[]).append(r)
    assert len(groups)==407 and sum(map(len,groups.values()))==1724
    assert all(len(v)==(8 if k.startswith('t502_field_') else 4) for k,v in groups.items())
    return dict(sorted(groups.items()))

def combination(scores,y):
    y=np.asarray(y,dtype=np.float64);n=len(y);assert n>1
    adv=y-(y.sum()-y)/(n-1)
    return (adv[:,None]*scores).mean(0)

def conditional_KL(loc,initial,sigma):
    scale=loc.new_tensor(sigma)
    assert scale.shape==(2,) and bool((scale>0).all())
    return .5*((loc-initial)/scale).square().sum(-1)

def metrics(q,group,p):
    vals=[]
    with torch.no_grad():
        for g in group:
            loc=M.location(q,g);initial=g['initial']
            # Actual per-action behavior sigma, never the nominal replay scale.
            kl=conditional_KL(loc,initial,p['raw_sigma'])
            vals.append(dict(mean=float(kl.mean()),maximum=float(kl.max()),sum=float(kl.sum()),p95=float(torch.quantile(kl,.95)),
                             peak_decision=int(kl.argmax()),decisions=len(kl),source_sha256=g['source_sha256']))
        sur=float(A.outcome_surrogate(q,group,p))
    return dict(task_episode_mean_KL=sum(x['mean'] for x in vals)/len(vals),max_conditional_KL=max(x['maximum'] for x in vals),
                mean_episode_sum_KL=sum(x['sum'] for x in vals)/len(vals),max_episode_p95_KL=max(x['p95'] for x in vals),episodes=vals,sample_surrogate=sur)

def compute(p,limit):
    started=time.time();last=[0.]
    def guard(phase,key):
        A.guard(p,started,limit)
        if time.time()-last[0]>3:
            save(OUT/'status.json',dict(state='ACTIVE',action=sys.argv[1],phase=phase,task=key,child_pid=os.getpid(),epoch=time.time(),new_native=0,new_optimizer=0));last[0]=time.time()
    guard('setup','');science=read(p['science_protocol']);science.update(out=str(OUT),arm='t673')
    T.setup(science);q=M.L.frozen(science,p['initial']['path']);q.requires_grad_(False)
    q.drive_adapter.requires_grad_(True);q.steer_adapter.requires_grad_(True)
    variables=[v for v in q.parameters() if v.requires_grad];assert sum(v.numel() for v in variables)==11906
    base=q.phash();before=[v.detach().clone() for v in variables];groups=dataset(p);ids=list(groups);grads=[];support={};omissions={}
    for key,rows in groups.items():
        guard('scores',key);g,y=A.group(q,rows,science);n=len(y);scores=[]
        if len(set(y))>1:
            for item in g:
                score=M.P.L.density(M.location(q,item),item['raw'],item['atom'],science['raw_sigma']).sum()/science['score_scale']
                scores.append(A.flatten_gradient(score,variables))
            scores=np.stack(scores);grad=combination(scores,y)
            direct=A.flatten_gradient(A.outcome_surrogate(q,g,science),variables)
            assert np.max(np.abs(grad-direct))<1e-10
            contribution_norms=np.linalg.norm(scores*(np.array(y)-(sum(y)-np.array(y))/(n-1))[:,None]/n,axis=1)
            loo=[];omissions[key]=[]
            for i in range(n):
                sel=np.arange(n)!=i;v=combination(scores[sel],np.array(y)[sel]);norm=float(np.linalg.norm(v));den=norm*np.linalg.norm(grad)
                loo.append(dict(removed=i,norm=norm,cosine=float(v@grad/den) if den>1e-24 else None));omissions[key].append(v)
        else:grad=np.zeros(11906);loo=[];contribution_norms=np.zeros(n)
        grads.append(grad);support[key]=dict(outcomes=y,n=n,mixed=len(set(y))>1,gradient_norm=float(np.linalg.norm(grad)),leave_one_trajectory_out=loo,
            contribution_norms=contribution_norms.tolist(),largest_contribution_fraction=float(contribution_norms.max()/max(contribution_norms.sum(),1e-30)),
            zero_gradient_is_not_protection=len(set(y))==1)
        del g
    grads=np.stack(grads);field=np.array([k.startswith('t502_field_') for k in ids]);old=np.array([k in p['protected384'] for k in ids]);assert old.sum()==384 and field.sum()==24
    guard('nominal387','');refs=M.references(science,q,lambda:guard('nominal387',''));keep=M.pack(refs)
    loss,_=M.retention(q,keep,False);kg=A.flatten_gradient(loss,variables);del refs,keep,loss
    means={'all407':grads.mean(0),'field24':grads[field].mean(0)}
    directions={};representation_errors={}
    for name,reward in [('all407',means['all407']),('field24',means['field24'])]:
        gl=-reward+science['replay_weight']*kg
        gc=gl*min(1.,1./max(np.linalg.norm(gl),1e-30))
        formula=-science['fit_lr']*gc/(np.abs(gc)+science['Adam_eps'])
        try:
            A.move(q,variables,before,formula)
            represented=np.concatenate([(v.detach()-b).flatten().numpy().copy() for v,b in zip(variables,before)])
        finally:A.move(q,variables,before,np.zeros(11906))
        representation_errors[name]=float(np.max(np.abs(represented-formula)));assert representation_errors[name]<1e-12
        directions[name]=represented
    for key,values in omissions.items():
        for row,v in zip(support[key]['leave_one_trajectory_out'],values):
            row['directional_margins']={name:float(v@step) for name,step in directions.items()}
    directional={name:grads@v for name,v in directions.items()}
    numeric=[];nonlinear={name:{} for name in directions}
    for key,rows in groups.items():
        guard('nonlinear',key);idx=ids.index(key);g,y=A.group(q,rows,science)
        with torch.no_grad():
            for item in g:item['initial']=M.location(q,item).detach().clone()
        if key in p['numeric_ids']:
            direction=grads[idx]/max(np.linalg.norm(grads[idx]),1e-30);eps=p['finite_parameter_epsilon']
            try:
                A.move(q,variables,before,direction,eps);plus=float(A.outcome_surrogate(q,g,science).detach())
                A.move(q,variables,before,direction,-eps);minus=float(A.outcome_surrogate(q,g,science).detach())
            finally:A.move(q,variables,before,np.zeros(11906))
            actual=(plus-minus)/(2*eps);expected=float(grads[idx]@direction)
            numeric.append(dict(id=key,analytic=expected,numeric=actual,error=abs(actual-expected),passed=abs(actual-expected)<=p['numeric_atol']+p['numeric_rtol']*abs(expected)))
        for name,step in directions.items():
            try:A.move(q,variables,before,step);nonlinear[name][key]=metrics(q,g,science)
            finally:A.move(q,variables,before,np.zeros(11906))
        assert q.phash()==base
    norms=np.linalg.norm(grads,axis=1);mixed=norms>1e-12
    checks=dict(all407_full_density_carry_verified=True,all_gradient_finite=bool(np.isfinite(grads).all()),numeric_all=all(x['passed'] for x in numeric),
                numeric_count=len(numeric)==len(p['numeric_ids']),mixed_original_at_least3=int((mixed&old).sum())>=3,mixed_field_at_least1=int((mixed&field).sum())>=1)
    summaries={}
    for name,v in directions.items():
        changes=directional[name];summaries[name]=dict(displacement_norm=float(np.linalg.norm(v)),old_negative_ids=[ids[i] for i in np.where(old&(changes<-1e-12))[0]],
            old_positive=int((old&(changes>1e-12)).sum()),no_signal_original=int((old&~mixed).sum()),field_mean_directional=float(means['field24']@v),
            max_task_mean_KL=max(x['task_episode_mean_KL'] for x in nonlinear[name].values()),max_conditional_KL=max(x['max_conditional_KL'] for x in nonlinear[name].values()),
            max_mean_episode_sum_KL=max(x['mean_episode_sum_KL'] for x in nonlinear[name].values()),old_nonlinear_negative_ids=[k for k in ids if k in p['protected384'] and nonlinear[name][k]['sample_surrogate']<-1e-12])
    arrays=dict(task_gradients=grads,nominal387_gradient=kg,all407_displacement=directions['all407'],field24_displacement=directions['field24'],mixed_normalized_Gram=(grads[mixed]/norms[mixed,None])@(grads[mixed]/norms[mixed,None]).T)
    result=dict(state='SIGNAL_DIAGNOSTIC_PASS' if all(checks.values()) else 'SCIENTIFIC_GATE_FAILED',checks=checks,ids=ids,support=support,representation_errors=representation_errors,
        direction_summaries=summaries,numeric=numeric,nonlinear=nonlinear,source_policy=base,
        direction_is_first_Adam_formula_not_optimizer=True,conditional_KL_not_closed_loop_or_expected_trajectory_bound=True,
        diagnostic_does_not_authorize_learning=True,new_native=0,new_optimizer=0,array_hashes={k:digest(v) for k,v in arrays.items()})
    assert q.phash()==base and all(v.grad is None for v in q.parameters())
    return result,arrays

def execute(action):
    p=read(OUT/'protocol.json');assert p['authorization_approved'] and sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    for f,h in p['inputs'].items():assert sha(f)==h,f
    seal=read(Path(p['source_run'])/'SEALED.json');assert seal['all_scientific_processes_stopped'] and all(not Path('/proc',str(pid)).exists() for pid in seal['scientific_pids'])
    assert not Path('/proc',str(p['source_driver_pid'])).exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time();result,arrays=compute(p,p[action+'_seconds'])
    if action=='main':
        with (OUT/'gradients.npz').open('xb') as f:np.savez_compressed(f,**arrays)
        save(OUT/'RESULTS.json',result)
    else:
        assert result==read(OUT/'RESULTS.json')
        with np.load(OUT/'gradients.npz',allow_pickle=False) as f:
            assert set(f.files)==set(arrays) and all(np.array_equal(f[k],v) for k,v in arrays.items())
        save(OUT/'audit.json',dict(state='PASS',all407_support_gradient_directions_KL_exact=True,new_native=0,new_optimizer=0))
    assert T.R.used(OUT)<p['max_bytes'] and T.R.used(ROOT/'stage/runs')<p['global_bytes']
    save(OUT/(action+'_receipt.json'),dict(state='COMPLETE',elapsed_s=time.time()-start,new_native=0,new_optimizer=0));fcntl.flock(lock,fcntl.LOCK_UN)

if __name__=='__main__':execute(sys.argv[1])
