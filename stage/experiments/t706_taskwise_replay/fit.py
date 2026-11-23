"""Fixed-feature mean vs Group-DRO, same seven taskwise convex constraints."""
from pathlib import Path
import copy,fcntl,gzip,importlib.util,io,json,os,sys,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'stage/runs/t706_solver_dependencies'))
import numpy as np
from scipy import sparse
import clarabel
spec=importlib.util.spec_from_file_location('t706_fit_controller',HERE/'run.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
A=C.A;torch=A.torch;digest=A.digest;OUT=C.OUT
sys.path.insert(0,str(ROOT/'stage/experiments/t701_compact_retention'));import t701_actor as Legacy

def stats(Z,Y,groups,records,guard):
    result=[]
    for i,r in enumerate(records):
        guard();take=groups==i;z=Z[take];y=Y[take];n=len(z);assert n==r['length']
        result.append(dict(id=r['id'],G=z.T@z/(2*n),B=z.T@y/(2*n),constant=float((y*y).sum()/(2*n))))
    return result

def losses(Z,Y,groups,records,W):
    return {r['id']:float(((Z[groups==i]@W-Y[groups==i])**2).mean()) for i,r in enumerate(records)}

def psdfactor(matrix):
    eigen,v=np.linalg.eigh((matrix+matrix.T)/2)
    assert eigen.min()>=-1e-10*max(1.,float(eigen.max())),('Gram not PSD',eigen.min())
    R=np.sqrt(np.maximum(eigen,0))[:,None]*v.T
    assert np.max(np.abs(R.T@R-matrix))<=1e-9*max(1.,np.max(np.abs(matrix)))
    return R

def source_factor(s):
    G,B,c=s['G'],s['B'],s['constant'];d=len(G);Q=np.zeros((2*d+1,2*d+1));Q[:d,:d]=Q[d:2*d,d:2*d]=G
    Q[:2*d,-1]=Q[-1,:2*d]=-B.reshape(-1,order='F');Q[-1,-1]=c
    return psdfactor(Q)

def conic(stat,W0,arm,guard,emit,time_limit=220.):
    d=len(W0);n=2*d+1;old=stat[:403] if len(stat)==410 else stat[:-7];src=stat[-7:]
    oldscale=float(np.mean([np.sum(W0*(s['G']@W0))-2*np.sum(W0*s['B'])+s['constant'] for s in old]));assert oldscale>0
    G=np.mean([s['G'] for s in old],axis=0);rho=1e-10*float(np.trace(G))/d
    w0=W0.reshape(-1,order='F');P=np.zeros((n,n));q=np.zeros(n)
    for k in [0,d]:P[k:k+d,k:k+d]=2*(rho*np.eye(d)+(G if arm=='mean_taskwise' else 0))/oldscale
    q[:2*d]=-2*rho*w0/oldscale
    if arm=='dro_taskwise':q[-1]=1.
    mats=[];bs=[];cones=[];bounds=[]
    for s in src:
        bound=float(np.sum(W0*(s['G']@W0))-2*np.sum(W0*s['B'])+s['constant']);assert bound>0
        F=source_factor(s)/np.sqrt(bound);block=np.zeros((len(F)+1,n));b=np.zeros(len(F)+1);b[0]=1;block[1:,:2*d]=-F[:,:2*d];b[1:]=F[:,-1]
        mats.append(sparse.csc_matrix(block));bs.append(b);cones.append(clarabel.SecondOrderConeT(len(b)));bounds.append(bound)
    if arm=='dro_taskwise':
        for s in old:
            R=psdfactor(s['G']/oldscale);F=sparse.block_diag((R,R),format='csc')
            # ||[2Rvec(W),t-1]|| <= t+1 is exactly L_i(W)/oldscale<=t.
            mat=sparse.vstack([sparse.csc_matrix(([ -1.],([0],[n-1])),shape=(1,n)),
                sparse.hstack([-2*F,sparse.csc_matrix((2*d,1))]),
                sparse.csc_matrix(([-1.],([0],[n-1])),shape=(1,n))],format='csc')
            b=np.zeros(2*d+2);b[0]=1;b[-1]=-1
            mats.append(mat);bs.append(b);cones.append(clarabel.SecondOrderConeT(len(b)))
    mat=sparse.vstack(mats,format='csc');b=np.concatenate(bs);settings=clarabel.DefaultSettings();settings.verbose=False
    settings.max_iter=128;settings.time_limit=time_limit;settings.max_threads=1
    settings.tol_gap_abs=settings.tol_gap_rel=settings.tol_feas=1e-9
    solver=clarabel.DefaultSolver(sparse.triu(sparse.csc_matrix(P)).tocsc(),q,mat,b,cones,settings)
    iterations=set();curve=[]
    def callback(info):
        guard();k=int(info.iterations)
        if k>0 and k not in iterations:
            iterations.add(k);emit('BEFORE',k,'conic_iteration')
            curve.append(dict(iteration=k,gap_abs=float(info.gap_abs),primal_residual=float(info.res_primal),dual_residual=float(info.res_dual)))
            emit('AFTER',k,'conic_iteration')
        return False
    solver.set_termination_callback(callback);solution=solver.solve();guard()
    assert str(solution.status)=='Solved',('Fixed conic solve failed',str(solution.status),solution.iterations)
    assert int(solution.iterations)<=128 and len(iterations)==int(solution.iterations)
    x=np.asarray(solution.x);dual=np.asarray(solution.z);slack=np.asarray(solution.s)
    primal=np.max(np.abs(mat@x+slack-b))/(1+np.max(np.abs(b)))
    stationary=np.max(np.abs(P@x+q+mat.T@dual))/(1+np.max(np.abs(q)))
    gap=abs(float(x@P@x+q@x+b@dual));primalobj=float(.5*x@P@x+q@x)
    assert primal<=1e-6 and stationary<=1e-6 and gap<=1e-6*(1+abs(primalobj)),(primal,stationary,gap,primalobj)
    pos=0;worst_primal=worst_dual=0.
    for cone in cones:
        size=cone.dim;sv=slack[pos:pos+size];zv=dual[pos:pos+size];pos+=size
        worst_primal=min(worst_primal,float(sv[0]-np.linalg.norm(sv[1:])));worst_dual=min(worst_dual,float(zv[0]-np.linalg.norm(zv[1:])))
    assert worst_primal>=-1e-6 and worst_dual>=-1e-6
    W=x[:2*d].reshape((d,2),order='F')
    return W,dict(iterations=int(solution.iterations),status=str(solution.status),oldscale=oldscale,rho=rho,
        bounds=bounds,primal_residual=primal,stationarity_residual=stationary,dual_gap=gap,cone_primal_min=worst_primal,cone_dual_min=worst_dual,
        objective_without_constant=primalobj,epigraph=float(x[-1]),weights_hash=digest(torch.from_numpy(W.copy())),curve=curve,
        problem_matrix_shape=list(mat.shape),problem_nonzeros=mat.nnz,solver_version=clarabel.__version__,max_threads=1,max_iter=128,time_limit=time_limit)

def construct(p,data,guard,emit):
    torch.set_num_threads(1);q=A.frozen(p,p['repair_initial']['path']);head=q.replay_head
    with torch.no_grad():
        parts=[]
        for off in range(0,len(data['x']),2048):
            guard();h=head[1](head[0](data['x'][off:off+2048]));parts.append(torch.cat([h,torch.ones(len(h),1,dtype=torch.float64)],1))
        Z=torch.cat(parts).numpy();Y=data['y'].numpy();groups=data['groups'].numpy();records=data['records'];W0=torch.cat([head[2].weight.T,head[2].bias[None]],0).numpy().copy()
    assert np.max(np.abs(Y[groups<403]))==0. and Z.shape==(83693,109) and len(records)==410
    stat=stats(Z,Y,groups,records,guard);original=losses(Z,Y,groups,records,W0);models={};info={}
    for arm in ['mean_taskwise','dro_taskwise']:
        W,cert=conic(stat,W0,arm,guard,emit);full=losses(Z,Y,groups,records,W)
        for i,s in enumerate(stat):
            gram=float(np.sum(W*(s['G']@W))-2*np.sum(W*s['B'])+s['constant']);assert abs(gram-full[s['id']])<=1e-9
        for r in records[403:]:assert full[r['id']]<=original[r['id']]+1e-9,('Source taskwise bound',r['id'],full[r['id']],original[r['id']])
        if arm=='dro_taskwise':assert max(full[r['id']] for r in records[:403])<=cert['epigraph']*cert['oldscale']+1e-8
        emit('BEFORE',0,'candidate_assignment');model=copy.deepcopy(q)
        model.replay_head[2].weight.copy_(torch.from_numpy(W[:-1].T.copy()));model.replay_head[2].bias.copy_(torch.from_numpy(W[-1].copy()));emit('AFTER',0,'candidate_assignment')
        for k,v in q.state_dict().items():
            if k not in ['replay_head.2.weight','replay_head.2.bias']:assert torch.equal(model.state_dict()[k],v)
        models[arm]=model.eval();info[arm]=dict(certificate=cert,perID_loss=full,old_mean=float(np.mean([full[r['id']] for r in records[:403]])),old_max=max(full[r['id']] for r in records[:403]),state_hash=digest(model.state_dict()))
    return models,dict(finals=info,original_perID_loss=original,old_Actor_budget_frozen=True,no_new_training_data=True,hidden_and_capacity_unchanged=True)

def event(who):
    def emit(stage,step,kind):
        with (OUT/'fit/fit_calls.jsonl').open('a') as f:f.write(json.dumps(dict(who=who,stage=stage,step=step,kind=kind,epoch=time.time()))+'\n')
    return emit

def main(mode):
    p=C.read(OUT/'protocol.json');C.verify(p,force=True);end=min(time.time()+(900 if mode=='main' else 600),p['absolute_learning_stop'] if mode=='main' else p['absolute_all_stop'])-15
    def guard():assert time.time()<end;C.verify(p)
    torch.use_deterministic_algorithms(True);torch.set_num_threads(1);os.sched_setaffinity(0,set(range(8,16)))
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        data=torch.load(io.BytesIO(gzip.decompress((ROOT/'stage/runs/t699_matched_value/DATA.pt.gz').read_bytes())),map_location='cpu',weights_only=False)
        who='live' if mode=='main' else 'independent';models,info=construct(p,data,guard,event(who))
        if mode=='main':
            shadow,other=construct(p,data,guard,event('shadow'));assert info==other
            assert all(digest(models[k].state_dict())==digest(shadow[k].state_dict()) for k in models);del shadow
        spec=importlib.util.spec_from_file_location('t706_existing_metrics',ROOT/'stage/experiments/t701_compact_retention/fit.py');F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F);result={}
        for name,q in models.items():
            metric=F.metrics(p,q,data,guard,False);assert metric['added_tensor_bytes']==161584
            if mode=='main':
                model=Legacy.Old.export(q,p,OUT/'fit'/(name+'.pt.gz'));reload=A.frozen(p,model['path']);assert digest(q.state_dict())==digest(reload.state_dict());result[name]=dict(model=model,metrics=metric)
            else:
                saved=C.read(OUT/'fit/summary.json');assert info==saved['construction'];model=saved['finals'][name]['model'];reload=A.frozen(p,model['path']);assert digest(reload.state_dict())==digest(q.state_dict())
                for k,v in saved['finals'][name]['metrics'].items():
                    if not k.startswith(('batch1','peak_')):assert metric[k]==v,(name,k)
                result[name]=dict(independent_state_exact=True,independent_certificate_exact=True)
        C.verify(p,force=True)
        if mode=='main':C.save(OUT/'fit/summary.json',dict(state='COMPLETE',finals=result,construction=info,new_Adam=0,new_RL=0))
        else:C.save(OUT/'fit/audit.json',dict(state='PASS',finals=result,independent_full_data_source_bounds=True,new_native=0))
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main(sys.argv[1])
