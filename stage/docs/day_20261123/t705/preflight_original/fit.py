"""Two fixed 109x2 convex readout fits, no hidden changes or new data."""
from pathlib import Path
import copy,fcntl,gzip,importlib.util,io,json,os,sys,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'stage/experiments/t701_compact_retention'))
import t701_actor as Legacy
spec=importlib.util.spec_from_file_location('t705_fit_controller',HERE/'run.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
A=C.A;torch=A.torch;digest=A.digest;OUT=C.OUT

def solve(G,B,W0,rho,lam):
    matrix=G[0]+lam*G[1]+rho*torch.eye(len(W0),dtype=torch.float64)
    rhs=B[0]+lam*B[1]+rho*W0
    L=torch.linalg.cholesky(matrix);W=torch.cholesky_solve(rhs,L)
    residual=float((matrix@W-rhs).abs().max())/max(1.,float(rhs.abs().max()))
    assert residual<=1e-8 and torch.isfinite(W).all()
    return W,residual

def optimize(Z,Y,group,records,W0,guard,event):
    lengths=torch.tensor([r['length'] for r in records],dtype=torch.float64);G=[];B=[];Ys=[];weights=[]
    for isnew,nids in [(False,403),(True,7)]:
        ids=(group>=403) if isnew else (group<403);z=Z[ids];y=Y[ids];g=group[ids]
        w=1/(2*nids*lengths[g]);G.append(z.T@(w[:,None]*z));B.append(z.T@(w[:,None]*y));Ys.append((z,y));weights.append(w)
    rho=1e-10*float(torch.trace(G[0]))/len(W0);assert rho>0
    def loss(W,i):
        z,y=Ys[i];return float(((z@W-y).square()*weights[i][:,None]).sum())
    bound=loss(W0,1);curve=[]
    def trial(lam,kind):
        guard();k=len(curve);assert k<34;event('BEFORE',k,kind)
        W,res=solve(G,B,W0,rho,lam);old,new=loss(W,0),loss(W,1);reg=rho*float((W-W0).square().sum())
        row=dict(index=k,kind=kind,multiplier=lam,old_loss=old,source_loss=new,regularizer=reg,stationarity=res,weights_hash=digest(W));curve.append(row)
        event('AFTER',k,kind);return W,row
    balanced,br=trial(1.,'balanced_convex_readout')
    old,orr=trial(0.,'constrained_dual_old_endpoint')
    if orr['source_loss']<=bound:
        constrained=old;final=orr
    else:
        low=0.;high=None
        for k in range(8):
            lam=10.**k;W,row=trial(lam,'constrained_dual_bracket')
            if row['source_loss']<=bound:high=lam;constrained=W;final=row;break
            low=lam
        assert high is not None,'Fixed dual bracket exhausted; no native/no rescue'
        for k in range(24):
            lam=(low+high)/2;W,row=trial(lam,'constrained_dual_bisection')
            if row['source_loss']<=bound:high=lam;constrained=W;final=row
            else:low=lam
    objective=final['old_loss']+final['regularizer'];gap=final['multiplier']*(bound-final['source_loss'])
    assert final['source_loss']<=bound+1e-12 and gap>=-1e-12
    assert gap<=1e-4*(objective+bound+1e-12),('Fixed solver certificate failed',gap,objective,bound)
    assert objective<=loss(W0,0)+1e-10,'Original readout feasible but old objective got worse'
    return dict(balanced=balanced,constrained=constrained),dict(rho=rho,source_bound=bound,original_old_loss=loss(W0,0),original_weights_hash=digest(W0),
        old_gram_rank=int(torch.linalg.matrix_rank(G[0])),dimension=len(W0),curve=curve,solves=len(curve),
        dual_gap=gap,constrained_final=final,balanced_final=br,source_aggregate_not_perID=True,
        no_hidden_parameter_changes=True,no_native_checkpoint_selection=True)

def construct(p,d,guard,event):
    q=A.frozen(p,p['repair_initial']['path']);head=q.replay_head
    with torch.no_grad():
        parts=[]
        for start in range(0,len(d['x']),2048):
            guard();h=head[1](head[0](d['x'][start:start+2048]));parts.append(torch.cat([h,torch.ones(len(h),1,dtype=torch.float64)],1))
        Z=torch.cat(parts);W0=torch.cat([head[2].weight.T,head[2].bias[None]],0).detach().clone()
        assert float((Z[:128]@W0-head(d['x'][:128])).abs().max())<1e-12
        weights,info=optimize(Z,d['y'],d['groups'],d['records'],W0,guard,event)
    result={}
    for name,W in weights.items():
        model=copy.deepcopy(q);model.replay_head[2].weight.copy_(W[:-1].T);model.replay_head[2].bias.copy_(W[-1])
        for k,v in q.state_dict().items():
            if k not in ['replay_head.2.weight','replay_head.2.bias']:assert torch.equal(model.state_dict()[k],v),k
        result[name]=model.eval()
    return result,info

def event(who):
    def emit(stage,step,kind):
        with (OUT/'fit/fit_calls.jsonl').open('a') as f:f.write(json.dumps(dict(who=who,stage=stage,step=step,kind=kind,epoch=time.time()))+'\n')
    return emit

def main(mode):
    p=C.read(OUT/'protocol.json');C.verify(p,force=True)
    end=min(time.time()+(600 if mode=='main' else 450),p['absolute_learning_stop'] if mode=='main' else p['absolute_all_stop'])-10
    def guard():assert time.time()<end;C.M.resource_wait(end)
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True);os.sched_setaffinity(0,set(range(8,16)))
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        d=torch.load(io.BytesIO(gzip.decompress((ROOT/'stage/runs/t699_matched_value/DATA.pt.gz').read_bytes())),map_location='cpu',weights_only=False)
        assert len(d['x'])==83693 and int((d['groups']<403).sum())==81956
        who='live' if mode=='main' else 'independent';models,info=construct(p,d,guard,event(who))
        if mode=='main':
            shadow,check=construct(p,d,guard,event('shadow'));assert info==check
            assert all(digest(models[k].state_dict())==digest(shadow[k].state_dict()) for k in models);del shadow
        result={}
        from importlib.util import spec_from_file_location,module_from_spec
        ms=spec_from_file_location('t705_reused_cost_measure',ROOT/'stage/experiments/t701_compact_retention/fit.py');F=module_from_spec(ms);ms.loader.exec_module(F)
        for name,q in models.items():
            torch.set_num_threads(1);os.sched_setaffinity(0,{15});metric=F.metrics(p,q,d,guard,False)
            assert metric['added_tensor_bytes']==161584
            if mode=='main':
                model=Legacy.Old.export(q,p,OUT/'fit'/(name+'.pt.gz'))
                reload=A.frozen(p,model['path']);assert digest(reload.state_dict())==digest(q.state_dict())
                result[name]=dict(model=model,metrics=metric)
            else:
                saved=C.read(OUT/'fit/summary.json');assert info==saved['construction']
                model=saved['finals'][name]['model'];reload=A.frozen(p,model['path']);assert digest(reload.state_dict())==digest(q.state_dict())
                for key,v in saved['finals'][name]['metrics'].items():
                    if not key.startswith(('batch1','peak_')):assert metric[key]==v,(name,key)
                result[name]=dict(independent_state_exact=True,independent_metrics=True)
        C.verify(p,force=True)
        if mode=='main':C.save(OUT/'fit/summary.json',dict(state='COMPLETE',finals=result,construction=info,new_Adam=0,new_RL=0))
        else:C.save(OUT/'fit/audit.json',dict(state='PASS',finals=result,independent_construction_exact=True,solves=info['solves'],new_native=0))
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main(sys.argv[1])
