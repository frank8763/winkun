"""Paired rehearsal distributions with matched per-ID total old weight."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t678_reference_target_pilot'))
import t678_learning as Base
B,M,T,torch,np=Base.B,Base.M,Base.T,Base.torch,Base.np
read,sha,digest=Base.read,Base.sha,Base.digest
make,optimizer_state,load=Base.make,Base.optimizer_state,Base.load

def feature(q,path,expected,key):
    assert sha(path)==expected;a=T.R.E.N.S.arrays(path)
    assert not a['ownership'].any()
    g=M.features(q,a,key,expected)
    return g,a

def data(p,q,guard):
    contract=read(p['reference_contract'])['reference_rows'];old=[];induced=[];novel=[]
    for row in contract:
        if row['origin']!='T501_original_success':continue
        guard();g,a=feature(q,row['source_path'],row['sha256'],row['task_id'])
        assert not a['stochastic'].any() and a['end_flags'][-1,1] and not a['end_flags'][:-1].any()
        with torch.no_grad():g['target']=M.location(q,g).clamp(-1,1).clone()
        old.append(g)
    assert len(old)==len({g['id'] for g in old})==384
    for row in p['induced_rows']:
        guard();g,a=feature(q,row['path'],row['sha256'],row['job_id'])
        assert not a['stochastic'].any() and row['full'] and not row.get('exact')
        # These are local old-policy function labels on actual external
        # histories, including collisions; no success claim is attached.
        with torch.no_grad():g['target']=M.location(q,g).clamp(-1,1).clone()
        induced.append(g)
    assert len(induced)==len({g['id'] for g in induced})==32 and {g['id'] for g in induced}<=set(p['before_success_ids'])
    source=M.L.frozen(p,p['novel_source_model']['path'])
    assert M.L.fixed_hash(source)==M.L.fixed_hash(q)
    for row in p['novel_rows']:
        guard();g,a=feature(q,row['path'],row['sha256'],row['id'])
        assert row['safe'] and not a['stochastic'].any() and a['end_flags'][-1,1] and not a['end_flags'][:-1].any()
        assert bool((np.abs(a['poses'][:,:2])<=7.).all())
        with torch.no_grad():locations=source.sequence(torch.tensor(a['states']))
        assert float((locations-torch.tensor(a['locations'])).abs().max())<1e-10
        assert np.array_equal(locations.clamp(-1,1).float().numpy(),a['actions'])
        novel.append(g)
    assert len(novel)==len({g['id'] for g in novel})==6 and not ({g['id'] for g in novel}&set(p['before_success_ids']))
    nominal=M.pack(old);onstudent=M.pack(induced);new=M.pack(novel)
    old_initial=float(M.retention(q,nominal,False)[0].detach());induced_initial=float(M.retention(q,onstudent,False)[0].detach())
    assert old_initial<1e-25 and induced_initial<1e-25
    per_old=dict(zip(nominal['ids'],nominal['lengths']));weights=[]
    for key,n in zip(nominal['ids'],nominal['lengths']):weights.extend([(0.5 if key in onstudent['ids'] else 1.)/(384*n)]*n)
    induced_weight=[]
    for key,n in zip(onstudent['ids'],onstudent['lengths']):induced_weight.extend([.5/(384*n)]*n)
    nominal['induced_arm_weight']=torch.tensor(weights,dtype=torch.float64);onstudent['induced_arm_weight']=torch.tensor(induced_weight,dtype=torch.float64)
    assert abs(float(nominal['induced_arm_weight'].sum()+onstudent['induced_arm_weight'].sum())-1)<1e-12
    for key in onstudent['ids']:
        assert .5/384+.5/384==1/384
    panel={}
    for key,rows in B.dataset(p).items():
        if key not in p['gradient_panel_ids']:continue
        guard();g,y=B.A.group(q,rows,p)
        with torch.no_grad():
            for item in g:item['initial']=M.location(q,item).detach().clone()
        panel[key]=g
    return panel,None,dict(nominal=nominal,induced=onstudent,novel=new)

def loss(q,keep,p):
    new,each=M.retention(q,keep['novel'],False)
    if p['arm']=='nominal':old,old_each=M.retention(q,keep['nominal'],False)
    else:
        assert p['arm']=='induced';old=q.head.weight.new_zeros(())
        for key in ['nominal','induced']:
            g=keep[key];per=M.P.action_loss(M.location(q,g),g['target']).mean(-1)/.2**2
            old+=(per*g['induced_arm_weight']).sum()
    return new+p['old_rehearsal_weight']*old,dict(new=float(new.detach()),old=float(old.detach()))

def entry_numeric(q,keep,p):
    variables=[v for v in q.parameters() if v.requires_grad];theta=torch.cat([v.detach().flatten() for v in variables]).clone()
    gradient=B.A.flatten_gradient(loss(q,keep,p)[0],variables)
    direction=np.random.default_rng(69120261121).standard_normal(11906);direction/=np.linalg.norm(direction)
    def represent(vv):
        k=0
        with torch.no_grad():
            for v in variables:v.copy_(vv[k:k+v.numel()].reshape_as(v));k+=v.numel()
    try:
        represent(theta+torch.tensor(direction*1e-6));plus=float(loss(q,keep,p)[0].detach())
        represent(theta-torch.tensor(direction*1e-6));minus=float(loss(q,keep,p)[0].detach())
    finally:represent(theta)
    actual=(plus-minus)/2e-6;expected=float(gradient@direction);error=abs(actual-expected)
    assert error<=1e-8+1e-4*abs(expected) and torch.equal(torch.cat([v.detach().flatten() for v in variables]),theta)
    return dict(analytic=expected,numeric=actual,error=error,gradient_norm=float(np.linalg.norm(gradient)))

def fit(p,q,opt,panel,unused,keep,guard,event):
    assert p['Actor_steps']==64 and not opt.state
    before=optimizer_state(q,opt);fixed=M.L.fixed_hash(q);variables=[v for v in q.parameters() if v.requires_grad];trace=[]
    for k in range(p['Actor_steps']):
        guard();opt.zero_grad(set_to_none=True);value,parts=loss(q,keep,p);value.backward()
        norm=float(torch.nn.utils.clip_grad_norm_(variables,1.,error_if_nonfinite=True));event('BEFORE',k);opt.step();event('AFTER',k)
        assert M.L.fixed_hash(q)==fixed and all(bool(torch.isfinite(v).all()) for v in variables)
        trace.append(dict(step=k,loss=float(value.detach()),gradient_norm=norm,parts=parts))
    values={key:B.metrics(q,g,p) for key,g in panel.items()};maxkl=max(v['task_episode_mean_KL'] for v in values.values())
    proposed=optimizer_state(q,opt);accepted=maxkl<=p['max_task_latent_KL']
    final,parts=loss(q,keep,p);old_nominal=float(M.retention(q,keep['nominal'],False)[0].detach());old_induced=float(M.retention(q,keep['induced'],False)[0].detach())
    if not accepted:load(q,opt,before);assert digest(optimizer_state(q,opt))==digest(before)
    return dict(Actor_Adam=64,retained_steps=64 if accepted else 0,accepted=accepted,arm=p['arm'],trace=trace,
        final_loss=float(final.detach()),final_parts=parts,nominal_old_loss=old_nominal,induced_old_function_loss=old_induced,
        max_actual_sigma_task_mean_KL=maxkl,nonlinear_panel=values,old_function_labels_NOT_recovery_success=True,
        all32_induced_histories_used_including_failures=True,matched_per_ID_total_old_weight=True,
        offline_fixed_rejected_policy_occupancy_NOT_iterative_DAgger=True,new_Value=0),proposed
