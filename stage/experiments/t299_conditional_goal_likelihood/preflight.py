from t299_learning import *
import builtins,dis,importlib.util,types,json,io,hashlib,time
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t299_conditional_goal_likelihood'

def actual_gradient_contract(q,items):
    result=[]
    for item in [items[0],items[4]]:
        x,y=item;k=len(x)//2;xp=x[[k]];prev=x[[k-1]]
        with torch.no_grad():_,m,_,raw=q.step(xp);_,pm,_,_=q.step(prev)
        mm=m.detach().double().requires_grad_(True);pp=pm.detach().double().requires_grad_(True);a=y[[k]].double();pa=y[[k-1]].double();reset=torch.tensor([False])
        nll=-D.logp(mm,a,pp,pa,reset).sum();gm,gp=torch.autograd.grad(nll,[mm,pp]);errors=[];eps=1e-6
        for name,v,g in [('current',mm,gm),('previous',pp,gp)]:
            for j in range(2):
                plus=v.detach().clone();minus=plus.clone();plus[0,j]+=eps;minus[0,j]-=eps
                vp=-D.logp(plus if name=='current' else mm.detach(),a,plus if name=='previous' else pp.detach(),pa,reset)
                vm=-D.logp(minus if name=='current' else mm.detach(),a,minus if name=='previous' else pp.detach(),pa,reset)
                errors.append(abs(float((vp-vm)/(2*eps))-float(g[0,j]))/(1+abs(float(g[0,j]))))
        assert max(errors)<1e-5
        leaf=raw.detach().clone().requires_grad_(True);mode=q.mode(leaf);mse=(mode-y[[k]]).square().sum();mg=torch.autograd.grad(mse,leaf)[0]
        leaf=raw.detach().clone().requires_grad_(True);mean=R.center(leaf);nn=-D.logp(mean,y[[k]],pm.detach(),y[[k-1]],reset).sum();ng=torch.autograd.grad(nn,leaf)[0]
        assert torch.isfinite(ng).all()
        if float(raw[0,0])>1:assert float(mg[0,0])==0 and abs(float(ng[0,0]))>1e-8
        result.append(dict(actual_current_previous_mean_relative_fd_error=max(errors),drive_raw=float(raw[0,0]),mode_MSE_drive_output_gradient=float(mg[0,0]),conditional_NLL_drive_output_gradient=float(ng[0,0])))
    return result

def main():
    assert not (OUT/'preflight.json').exists() and not (OUT/'launch.json').exists();p=json.loads((HERE/'protocol.json').read_text());torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    for filename in ['run.py','audit.py']:
        spec=importlib.util.spec_from_file_location('t299_'+filename[:-3],HERE/filename);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);missing=set()
        def check(code):
            for ins in dis.get_instructions(code):
                if ins.opname=='LOAD_GLOBAL' and ins.argval not in vars(mod) and ins.argval not in vars(builtins):missing.add(ins.argval)
            for co in code.co_consts:
                if isinstance(co,types.CodeType):check(co)
        check(mod.main.__code__);assert not missing,(filename,missing)
    q=policy(p);assert S.digest(q.state_dict())==p['policy_hash'];items=auxiliary(p);gradient=actual_gradient_contract(q,items);reference=D.reference_contract()
    # New objective scale determined only once, no optimizer or native steps.
    if (OUT/'gradient_scale.json').exists():scale=json.loads((OUT/'gradient_scale.json').read_text())
    else:
        scale=gradient_scale(q,items);C.save(OUT/'gradient_scale.json',scale)
    assert scale['source_policy_hash']==p['policy_hash']
    p['auxiliary_NLL_scale']=scale['factor'];p['initial_gradient_scale_sha256']=C.sha(OUT/'gradient_scale.json');p['input_hashes'][str(OUT/'gradient_scale.json')]=C.sha(OUT/'gradient_scale.json')
    for f in [HERE/'protocol.json',OUT/'protocol.json']:f.write_text(json.dumps(p,separators=(',',':'))+'\n')
    # Recompute all historical original means and stable logp on canonical batch1.
    prior=json.loads((ROOT/'stage/runs/t295_hindsight_support/protocol.json').read_text());lp_errors=[]
    for source in prior['sources']:
        with np.load(resolve(source['path'])) as z:a={k:z[k].copy() for k in z.files}
        with torch.no_grad():
            mean=exact_mean(q,torch.tensor(a['states']));act=torch.tensor(a['actions']);reset=torch.zeros(len(act),dtype=torch.bool);reset[0]=True
            lp=D.logp(mean,act,torch.cat([mean[:1],mean[:-1]]),torch.cat([act[:1],act[:-1]]),reset).numpy()
        assert np.array_equal(mean.numpy(),a['means']);error=float(np.abs(lp-a['logp']).max());assert error<1e-8;lp_errors.append(error)
    frozen=copy.deepcopy(q).eval().requires_grad_(False);ax,ay,offset=anchors(p,frozen);shadow=copy.deepcopy(q);opt=L.optimizer(q,p);so=L.optimizer(shadow,p);calls=[]
    def count(o,name):
        original=o.step
        def step(*a,**kw):
            assert len(calls)<4;calls.append(name);C.save(OUT/'preflight_optimizer_calls.json',dict(calls=calls,native_steps=0,navigation_updates=0));return original(*a,**kw)
        o.step=step
    count(opt,'supervised_clone');count(so,'supervised_shadow_clone');live=supervised_step(q,opt,items,ax,ay,offset,p,0);again=supervised_step(shadow,so,items,ax,ay,offset,p,0)
    assert live==again and S.digest(q.state_dict())==S.digest(shadow.state_dict()) and S.digest(opt.state_dict())==S.digest(so.state_dict())
    before=S.digest(q.state_dict());bo=S.digest(opt.state_dict());L.guarded_adam_step(q,opt,lambda actor:dict(accept=False),[1.,.25,.0625]);assert S.digest(q.state_dict())==before and S.digest(opt.state_dict())==bo
    def interrupt(actor):raise RuntimeError('T299 disposable interrupt')
    try:L.guarded_adam_step(q,opt,interrupt,[1.,.25,.0625])
    except RuntimeError:pass
    else:raise AssertionError('Expected interruption')
    assert S.digest(q.state_dict())==before and S.digest(opt.state_dict())==bo and len(calls)==4
    raw=io.BytesIO();torch.save(dict(policy=q.state_dict(),optimizer=L.optimizer(q,p).state_dict(),report=live),raw);assert raw.tell()+16384<p['model_bytes']
    assert 2+20+2+275+54+16+78+4==451<=p['max_compact_payloads']
    assert p['payload_bytes']+p['metadata_bytes']<p['final_bytes'] and S.used(ROOT/'stage/runs')-S.used(OUT)+p['final_bytes']<p['global_bytes']
    C.save(OUT/'preflight.json',dict(state='PASS',native_steps=0,native_environments_created=0,disposable_clone_optimizer_calls=4,navigation_updates=0,all_entry_global_names_resolve=True,original_policy_means_bit_exact=True,original_AR_logp_max_error=max(lp_errors),actual_gradient_contract=gradient,extreme_numerical_reference=reference,frozen_gradient_scale=scale,live_shadow_ActorAdam_exact=True,reject_interrupt_ActorAdam_exact=True,model_raw_bytes=raw.tell(),global_if_own_full=S.used(ROOT/'stage/runs')-S.used(OUT)+p['final_bytes'],native_parent_aux_controls_not_yet_run=True))
    print(json.dumps(dict(state='PASS',scale=scale['factor'],original_logp_max=max(lp_errors),clone_calls=4,native_steps=0)))

if __name__=='__main__':main()
