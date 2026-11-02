"""Fixed cross-fitted state baseline and exact conditional-score gradients."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t287_time_aligned_credit'))
import t287_credit as B
sys.path.insert(0,str(ROOT/'stage/experiments/t290_exact_duplicate_store'))
from store import resolve
import hashlib,io,json,numpy as np,torch


def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def load(plan):
    items=[]
    for row in plan['rows']:
        f=resolve(row['path']);assert sha(f)==row['sha256']
        with np.load(f) as z:a={k:z[k].copy() for k in z.files}
        assert len(a)==12 and all(B.S.digest(v)==row['array_hashes'][k] for k,v in a.items())
        assert np.array_equal(a['states'][1:],a['next_states'][:-1])
        items.append(dict(row=row,arrays=a,returns=B.S.monte_carlo(a['rewards'],a['native_counts'])))
    assert len(items)==32 and len({i['row']['seed'] for i in items})==8
    return items


def features(x,mu,sd):
    z=np.clip((x.astype(np.float64)-mu)/sd,-10,10)
    return np.c_[np.ones(len(z)),z,z*z]


def fit_predict(items,ridge,progress):
    families=sorted({i['row']['seed'] for i in items});pred=[None]*len(items);fits=[]
    for seed in families:
        train=[i for i in items if i['row']['seed']!=seed];test=[(k,i) for k,i in enumerate(items) if i['row']['seed']==seed]
        assert len(train)==28 and len(test)==4
        x=np.concatenate([i['arrays']['states'] for i in train]);y=np.concatenate([i['returns'] for i in train])
        w=np.concatenate([np.full(len(i['returns']),1/(28*len(i['returns']))) for i in train]);assert abs(w.sum()-1)<1e-12
        mu=np.sum(x*w[:,None],axis=0);sd=np.sqrt(np.maximum(np.sum((x-mu)**2*w[:,None],axis=0),1e-6))
        z=features(x,mu,sd);regularizer=np.eye(241)*ridge;regularizer[0,0]=0
        coeff=np.linalg.solve(z.T@(w[:,None]*z)+regularizer,z.T@(w*y))
        assert np.isfinite(coeff).all()
        for k,item in test:pred[k]=features(item['arrays']['states'],mu,sd)@coeff
        fits.append(dict(seed=seed,train_ids=[i['row']['id'] for i in train],test_ids=[i['row']['id'] for k,i in test],mu=mu,sd=sd,coeff=coeff))
        progress('VALUE_FOLD_'+str(seed))
    return pred,fits


def time_loo(items):
    out=[None]*len(items)
    for rnd in [0,1]:
        ids=[k for k,i in enumerate(items) if i['row']['id'].startswith(f'time_aligned_credit_r{rnd}_')];assert len(ids)==16
        vv=[items[k] for k in ids]
        values=B.physical_time_baselines([i['returns'] for i in vv],[i['arrays']['native_counts'] for i in vv],[i['row']['source_reset_id'] for i in vv])
        for k,value in zip(ids,values):out[k]=value
    return out


def exact_gradients(plan,items,pred,loo,progress):
    base=json.loads(Path(plan['source_protocol']).read_text());q=B.policy(base);q.eval();before=B.S.digest(q.state_dict())
    assert before==plan['policy_hash'];params=[p for p in q.parameters() if p.requires_grad];vectors=[];hashes=[];density=[]
    for j,item in enumerate(items):
        a=item['arrays'];x=torch.tensor(a['states']);actions=torch.tensor(a['actions']);_,m,_=q.sequence(x[:,None]);m=m[:,0]
        previous=torch.cat([m[:1],m[:-1]]);pa=torch.cat([actions[:1],actions[:-1]]);reset=torch.zeros(len(m),dtype=torch.bool);reset[0]=True
        lp=B.L.AR.logp(m,actions,previous,pa,reset)
        me=float((m-torch.tensor(a['means'])).abs().max().detach());le=float((lp-torch.tensor(a['logp'])).abs().max().detach())
        assert me<3e-6 and le<.005,(me,le)
        weights=B.R.GAMMA**np.r_[0,np.cumsum(a['native_counts'].astype(np.int64))[:-1]]
        for name,b in [('zero',np.zeros(len(m))),('time_loo',loo[j]),('state_value',pred[j])]:
            coefficients=torch.tensor(weights*(item['returns']-b)/base['surrogate_horizon_decisions'],dtype=torch.float64)
            loss=torch.dot(lp,coefficients)
            gg=torch.autograd.grad(loss,params,retain_graph=name!='state_value');v=torch.cat([g.detach().reshape(-1) for g in gg]);assert torch.isfinite(v).all()
            vectors.append(v);hashes.append(dict(trajectory=item['row']['id'],baseline=name,sha256=B.S.digest(v)))
        density.append(dict(id=item['row']['id'],mean_max_error=me,logp_max_error=le));progress('GRADIENT_'+str(j+1))
        del gg,v,loss,lp,m,previous,x
    assert B.S.digest(q.state_dict())==before and all(p.grad is None for p in params)
    matrix=torch.stack(vectors).double();gram=(matrix@matrix.T).numpy();assert gram.shape==(96,96)
    return gram,hashes,density


def measure(plan,items,pred,loo,gram):
    def stats(index):
        k=gram[np.ix_(index,index)];n=len(index);mean=float(k.sum()/n**2);var=float((np.trace(k)-k.sum()/n)/(n-1));return dict(mean_norm2=max(0,mean),sample_gradient_variance_trace=max(0,var))
    groups=sorted({i['row']['source_reset_id'] for i in items});families=sorted({i['row']['seed'] for i in items});metrics={}
    for axis,name in enumerate(['zero','time_loo','state_value']):
        task={}
        for task_id in groups:
            ids=[j for j,i in enumerate(items) if i['row']['source_reset_id']==task_id];assert len(ids)==8
            baselines=[np.zeros(len(items[j]['returns'])) if axis==0 else (loo[j] if axis==1 else pred[j]) for j in ids]
            mse=float(np.mean([np.mean((items[j]['returns']-b)**2) for j,b in zip(ids,baselines)]))
            task[task_id]=dict(**stats([3*j+axis for j in ids]),equal_episode_return_MSE=mse)
        combine=np.zeros((8,96))
        for k,f in enumerate(families):
            ids=[j for j,i in enumerate(items) if i['row']['seed']==f];assert len(ids)==4
            combine[k,[3*j+axis for j in ids]]=.25
        family=combine@gram@combine.T;n=8
        metrics[name]=dict(tasks=task,equal_task_return_MSE=float(np.mean([v['equal_episode_return_MSE'] for v in task.values()])),
            family_mean_gradient_norm2=max(0,float(family.sum()/n**2)),family_gradient_sample_variance_trace=max(0,float((np.trace(family)-family.sum()/n)/(n-1))))
    a=metrics['time_loo'];b=metrics['state_value'];m0=np.zeros(96);m1=np.zeros(96);m0[1::3]=1/32;m1[2::3]=1/32
    n0=float(m0@gram@m0);n1=float(m1@gram@m1);cos=float(m0@gram@m1/max(np.sqrt(max(0,n0*n1)),1e-30));ratio=np.sqrt(max(0,n1)/max(n0,1e-30))
    checks=dict(return_MSE_90pct=b['equal_task_return_MSE']<=.9*a['equal_task_return_MSE'],
        family_gradient_variance_90pct=b['family_gradient_sample_variance_trace']<=.9*a['family_gradient_sample_variance_trace'],
        each_task_variance_not_over110pct=all(b['tasks'][t]['sample_gradient_variance_trace']<=1.1*a['tasks'][t]['sample_gradient_variance_trace'] for t in groups),
        each_task_MSE_not_over110pct=all(b['tasks'][t]['equal_episode_return_MSE']<=1.1*a['tasks'][t]['equal_episode_return_MSE'] for t in groups),
        finite_sample_mean_gradient_cosine_ge080=cos>=.8,finite_sample_mean_norm_ratio_within_half_double=.5<=ratio<=2)
    return dict(metrics=metrics,mean_gradient_cosine=cos,mean_gradient_norm_ratio=ratio,registered_checks=checks,offline_screen_pass=all(checks.values()))


def synthetic():
    # Complete Bernoulli enumeration: a causal state-only baseline has zero expected score.
    p=.37;b=2.4;effect=sum((p if a else 1-p)*(a-p)*b for a in [0,1]);assert abs(effect)<1e-14
    x=np.arange(24,dtype=np.float64).reshape(8,3);mu=x.mean(0);sd=np.maximum(x.std(0),.001);assert features(x,mu,sd).shape==(8,7)
    return dict(state='PASS',baseline_expected_score=effect,synthetic_only=True,native_steps=0,optimizer_calls=0)
