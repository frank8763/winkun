"""A single 44-halfspace actual-displacement projection; zero Actor optimizer."""
from pathlib import Path
import fcntl,os,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t674_projection_feasibility';DOC=ROOT/'stage/docs/weekend_20261120_23/t674'
sys.path.insert(0,str(ROOT/'stage/experiments/t673_all_task_signal_gate'))
import t673_analysis as B
from status_io import save
T,M,torch,np=B.T,B.M,B.torch,B.np
read,sha,digest=B.read,B.sha,B.digest

def nnls_dual(matrix,target):
    """Finite Lawson-Hanson active-set NNLS using rank-aware least squares."""
    x=np.zeros(matrix.shape[1]);passive=np.zeros(len(x),dtype=bool);iterations=0
    while True:
        w=matrix.T@(target-matrix@x)
        if not ((~passive)&(w>1e-14)).any():break
        entering=int(np.argmax(np.where(passive,-np.inf,w)));passive[entering]=True
        while True:
            iterations+=1;assert iterations<=10000,'NNLS active-set iteration bound'
            z=np.zeros_like(x);z[passive]=np.linalg.lstsq(matrix[:,passive],target,rcond=1e-12)[0]
            if (z[passive]>0).all():x=z;break
            blocked=passive&(z<=0);alpha=float(np.min(x[blocked]/(x[blocked]-z[blocked])))
            x+=alpha*(z-x);remove=passive&(x<=1e-14);x[remove]=0;passive[remove]=False
    assert x.min()>=0
    return x,float(np.linalg.norm(matrix@x-target))

def projection(d0,rows):
    assert np.isfinite(d0).all() and np.isfinite(rows).all()
    dual,residual=nnls_dual(rows.T,-d0)
    d=d0+rows.T@dual;margins=rows@d
    stationarity=float(np.linalg.norm(d-d0-rows.T@dual));comp=float(np.max(np.abs(dual*margins)))
    primal=float(.5*np.square(d-d0).sum());dual_value=float(-.5*np.square(rows.T@dual).sum()-dual@(rows@d0));gap=primal-dual_value
    assert abs(gap)<1e-10
    assert margins.min()>=-1e-10 and stationarity<1e-10 and comp<1e-10
    return d,dict(dual=dual.tolist(),margins=margins.tolist(),stationarity=stationarity,complementarity=comp,
                  rank=int(np.linalg.matrix_rank(rows)),active=int((dual>1e-12).sum()),nnls_residual=float(residual),primal=primal,dual_value=dual_value,duality_gap=gap)

def compute(p,limit):
    start=time.time();last=[0.]
    def guard(key):
        B.A.guard(p,start,limit)
        if time.time()-last[0]>3:save(OUT/'status.json',dict(state='ACTIVE',action=sys.argv[1],task=key,child_pid=os.getpid(),epoch=time.time(),new_native=0,new_optimizer=0));last[0]=time.time()
    source=read(p['prior_result']);ids=source['ids']
    with np.load(p['prior_gradients'],allow_pickle=False) as f:grad=f['task_gradients'].copy();d0=f['all407_displacement'].copy()
    old=np.array([k in p['protected384'] for k in ids]);norm=np.linalg.norm(grad,axis=1);mask=old&(norm>1e-12);assert mask.sum()==44
    rows=grad[mask]/norm[mask,None];guard('projection');d,qp=projection(d0,rows)
    science=read(p['science_protocol']);science.update(out=str(OUT),arm='t674');T.setup(science)
    q=M.L.frozen(science,p['initial']['path']);q.requires_grad_(False);q.drive_adapter.requires_grad_(True);q.steer_adapter.requires_grad_(True)
    variables=[v for v in q.parameters() if v.requires_grad];before=[v.detach().clone() for v in variables];base=q.phash()
    try:
        B.A.move(q,variables,before,d);represented=np.concatenate([(v.detach()-b).flatten().numpy().copy() for v,b in zip(variables,before)])
    finally:B.A.move(q,variables,before,np.zeros_like(d))
    assert np.max(np.abs(represented-d))<1e-12;d=represented
    nonlinear={}
    for key,rr in B.dataset(p).items():
        guard(key);group,y=B.A.group(q,rr,science)
        with torch.no_grad():
            for g in group:g['initial']=M.location(q,g).detach().clone()
        try:B.A.move(q,variables,before,d);nonlinear[key]=B.metrics(q,group,science)
        finally:B.A.move(q,variables,before,np.zeros_like(d))
    field=np.array([k.startswith('t502_field_') for k in ids]);fieldg=grad[field].mean(0)
    old_negative=[k for k in ids if k in p['protected384'] and nonlinear[k]['sample_surrogate']<-p['nonlinear_tolerance']]
    mean_field=sum(nonlinear[k]['sample_surrogate'] for k in ids if k.startswith('t502_field_'))/24
    checks=dict(genuine_raw_old_conflict=bool((rows@d0<-1e-12).any()),genuine_changed_direction=bool(np.linalg.norm(d-d0)>1e-8),
                represented_old_linear_feasible=bool((rows@d).min()>=-1e-12),KKT=qp['stationarity']<1e-10 and qp['complementarity']<1e-10,
                field_linear_ascent=bool(fieldg@d>1e-12),field_nonlinear_ascent=mean_field>1e-12,
                all_old_nonlinear_nonnegative=not old_negative,max_task_mean_latent_KL=max(x['task_episode_mean_KL'] for x in nonlinear.values())<=.01,
                finite=bool(np.isfinite(d).all()))
    result=dict(state='PROJECTION_FEASIBILITY_PASS' if all(checks.values()) else 'SCIENTIFIC_GATE_FAILED',checks=checks,QP=qp,
                original_signal_ids=[k for k,m in zip(ids,mask) if m],original_no_signal_count=int((old&~mask).sum()),
                no_signal_not_protected=True,raw_negative_count=int((rows@d0<-1e-12).sum()),old_nonlinear_negative_ids=old_negative,
                field_linear_ascent=float(fieldg@d),field_nonlinear_mean=mean_field,relative_displacement_change=float(np.linalg.norm(d-d0)/np.linalg.norm(d0)),
                displacement_norm=float(np.linalg.norm(d)),nonlinear=nonlinear,source_policy=base,new_native=0,new_optimizer=0,
                NNLS_is_dual_QP_solver_not_Actor_optimizer=True,no_guarantee_of_full384_retention=True,
                array_hashes=dict(projected_displacement=digest(d),normalized_old_constraints=digest(rows)))
    assert q.phash()==base and all(v.grad is None for v in q.parameters())
    return result,dict(projected_displacement=d,normalized_old_constraints=rows)

def execute(action):
    p=read(OUT/'protocol.json');assert p['authorization_approved'] and sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    for f,h in p['inputs'].items():assert sha(f)==h,f
    for folder in p['sealed_sources']:
        seal=read(Path(folder)/'SEALED.json');assert seal['all_scientific_processes_stopped'] and all(not Path('/proc',str(pid)).exists() for pid in seal['scientific_pids'])
        assert not Path('/proc',str(read(Path(folder)/'DRIVER_STARTED.json')['pid'])).exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time();result,arrays=compute(p,p[action+'_seconds'])
    if action=='main':
        with (OUT/'projection.npz').open('xb') as f:np.savez_compressed(f,**arrays)
        save(OUT/'RESULTS.json',result)
    else:
        assert result==read(OUT/'RESULTS.json')
        with np.load(OUT/'projection.npz',allow_pickle=False) as f:assert set(f.files)==set(arrays) and all(np.array_equal(f[k],v) for k,v in arrays.items())
        save(OUT/'audit.json',dict(state='PASS',all_QP_and_represented_nonlinear_results_exact=True,new_native=0,new_optimizer=0))
    assert T.R.used(OUT)<p['max_bytes'] and T.R.used(ROOT/'stage/runs')<p['global_bytes']
    save(OUT/(action+'_receipt.json'),dict(state='COMPLETE',elapsed_s=time.time()-start,new_native=0,new_optimizer=0));fcntl.flock(lock,fcntl.LOCK_UN)

if __name__=='__main__':execute(sys.argv[1])
