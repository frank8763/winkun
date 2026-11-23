"""128 fixed source-gradient steps: mean A-GEM vs worst old-task reference."""
from pathlib import Path
import copy,fcntl,gzip,importlib.util,io,json,os,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('t708_fit_controller',HERE/'run.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
A=C.A;torch=A.torch;digest=A.digest;OUT=C.OUT
sys.path.insert(0,str(ROOT/'stage/experiments/t701_compact_retention'));import t701_actor as Legacy

def project(gradient,reference):
    dot=(gradient*reference).sum();norm=(reference*reference).sum()
    assert bool(norm>0) or not bool(reference.any()),'Nonzero reference norm underflow'
    active=bool(dot<0) and bool(norm>0)
    value=gradient-(dot/norm)*reference if active else gradient.clone()
    assert torch.isfinite(value).all()
    assert float((value*reference).sum())>=-1e-12*max(1.,float(value.norm()*reference.norm()))
    assert abs(float((gradient*value).sum()-(value*value).sum()))<=1e-10*max(1.,float((value*value).sum()))
    return value,active,float(dot),float(norm)

def loss(W,G,B,c):return (W[None]*(G@W)).sum((1,2))-2*(W[None]*B).sum((1,2))+c

def construct(p,data,guard,emit):
    q=A.frozen(p,p['repair_initial']['path']);head=q.replay_head
    with torch.no_grad():
        pieces=[]
        for off in range(0,len(data['x']),2048):
            guard();h=head[1](head[0](data['x'][off:off+2048]));pieces.append(torch.cat([h,torch.ones(len(h),1,dtype=torch.float64)],1))
        Z=torch.cat(pieces);Y=data['y'];groups=data['groups'];records=data['records'];W0=torch.cat([head[2].weight.T,head[2].bias[None]],0).clone()
    assert Z.shape==(83693,109) and not Y[groups<403].any() and len(records)==410
    grams=[];rhs=[];const=[]
    for i,r in enumerate(records):
        guard();mask=groups==i;z=Z[mask];y=Y[mask];n=len(z);assert n==r['length']
        grams.append(z.T@z/(2*n));rhs.append(z.T@y/(2*n));const.append(float(y.square().sum()/(2*n)))
    G=torch.stack(grams);B=torch.stack(rhs);c=torch.tensor(const,dtype=torch.float64)
    Gn=G[403:].mean(0);Bn=B[403:].mean(0);Go=G[:403].mean(0)
    Gn=(Gn+Gn.T)/2
    eigen,V=torch.linalg.eigh(Gn);eigen_residual=float((Gn@V-V*eigen[None]).abs().max());assert eigen_residual<=1e-10*max(1.,float(Gn.abs().max())) and float(eigen.min())>=-1e-12*max(1.,float(eigen.max()))
    lipschitz=2*float(eigen.max());assert lipschitz>0;lr=1/lipschitz
    original=loss(W0,G,B,c);finals={};models={};tie_order=sorted(range(403),key=lambda i:records[i]['id'])
    for arm in ['agem_mean','agem_worst']:
        W=W0.clone();curve=[];previous_reference=None
        for step in range(128):
            guard();emit('BEFORE',step,'gradient_evaluation');gnew=2*(Gn@W-Bn);emit('AFTER',step,'gradient_evaluation')
            values=loss(W,G,B,c);old=values[:403];source_before=float(values[403:].mean())
            emit('BEFORE',step,'gradient_evaluation')
            if arm=='agem_mean':gref=2*(Go@W);reference_id='ALL403_ID_BALANCED_MEAN';tie_count=1
            else:
                maximum=old.max();idx=next(i for i in tie_order if bool(old[i]==maximum));gref=2*(G[idx]@W);reference_id=records[idx]['id'];tie_count=int((old==maximum).sum())
            emit('AFTER',step,'gradient_evaluation')
            direction,active,dot,norm=project(gnew,gref)
            before_hash=digest(W)
            emit('BEFORE',step,'parameter_step');W=W-lr*direction;emit('AFTER',step,'parameter_step')
            after=loss(W,G,B,c);upper=source_before-.5*lr*float(direction.square().sum());descent_residual=float(after[403:].mean())-upper
            assert torch.isfinite(W).all() and float(after[403:].mean())<=source_before+1e-12 and descent_residual<=1e-12,'Source quadratic descent failed'
            curve.append(dict(step=step,reference_id=reference_id,projection_active=active,gradient_dot=dot,reference_norm_squared=norm,tie_count=tie_count,reference_switched=previous_reference is not None and previous_reference!=reference_id,source_gradient_hash=digest(gnew),reference_gradient_hash=digest(gref),direction_hash=digest(direction),before_weights_hash=before_hash,source_descent_upper_bound=upper,source_descent_residual=descent_residual,
                projected_reference_dot=float((direction*gref).sum()),source_before=source_before,source_after=float(after[403:].mean()),
                old_mean=float(after[:403].mean()),old_max=float(after[:403].max()),weights_hash=digest(W)))
            previous_reference=reference_id
        final=loss(W,G,B,c);perID={};oldchanged=0
        for i,r in enumerate(records):
            guard();mask=groups==i;direct=float((Z[mask]@W-Y[mask]).square().mean());assert abs(direct-float(final[i]))<=1e-9
            perID[r['id']]=direct
        model=copy.deepcopy(q);model.replay_head[2].weight.copy_(W[:-1].T);model.replay_head[2].bias.copy_(W[-1])
        for k,v in q.state_dict().items():
            if k not in ['replay_head.2.weight','replay_head.2.bias']:assert torch.equal(model.state_dict()[k],v)
        models[arm]=model.eval();finals[arm]=dict(steps=128,gradient_evaluations=256,learning_rate=lr,source_lipschitz=lipschitz,source_eigen_residual=eigen_residual,source_min_eigenvalue=float(eigen.min()),curve=curve,perID_loss=perID,
            old_mean=float(final[:403].mean()),old_max=float(final[:403].max()),source_mean=float(final[403:].mean()),
            projection_count=sum(r['projection_active'] for r in curve),state_hash=digest(model.state_dict()))
    return models,dict(finals=finals,original_perID_loss={r['id']:float(original[i]) for i,r in enumerate(records)},hidden_and_parent_frozen=True,source_gradient_steps_not_RL=True,
        old_Actor_ledger_frozen=True,fullbatch_fixed_feature_AGEM_adaptation_not_full_paper_replication=True)

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
        spec=importlib.util.spec_from_file_location('t708_existing_metrics',ROOT/'stage/experiments/t701_compact_retention/fit.py');F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F);result={}
        for name,q in models.items():
            metric=F.metrics(p,q,data,guard,False);assert metric['added_tensor_bytes']==161584
            if mode=='main':
                model=Legacy.Old.export(q,p,OUT/'fit'/(name+'.pt.gz'));reload=A.frozen(p,model['path']);assert digest(q.state_dict())==digest(reload.state_dict());result[name]=dict(model=model,metrics=metric)
            else:
                saved=C.read(OUT/'fit/summary.json');assert info==saved['construction'];model=saved['finals'][name]['model'];reload=A.frozen(p,model['path']);assert digest(reload.state_dict())==digest(q.state_dict())
                for k,v in saved['finals'][name]['metrics'].items():
                    if not k.startswith(('batch1','peak_')):assert metric[k]==v,(name,k)
                result[name]=dict(independent_state_exact=True,independent_gradient_projection_and_steps_exact=True)
        C.verify(p,force=True)
        if mode=='main':C.save(OUT/'fit/summary.json',dict(state='COMPLETE',finals=result,construction=info,new_Adam=0,new_RL=0))
        else:C.save(OUT/'fit/audit.json',dict(state='PASS',finals=result,independent_full_data_loss_and_projection_verified=True,new_native=0))
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main(sys.argv[1])
