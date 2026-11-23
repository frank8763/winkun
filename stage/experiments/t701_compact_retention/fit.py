"""Two fixed compact pilots, no native; every evaluation/update counted."""
from pathlib import Path
import copy,gzip,io,json,os,resource,sys,time
import t701_actor as A
torch,np,digest=A.torch,A.np,A.digest
ROOT=A.ROOT;HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t701_compact_retention'

def data():
    return torch.load(io.BytesIO(gzip.decompress((ROOT/'stage/runs/t699_matched_value/DATA.pt.gz').read_bytes())),map_location='cpu',weights_only=False)

def event(who):
    def emit(stage,k,kind):
        with (OUT/'fit/fit_calls.jsonl').open('a') as f:
            f.write(json.dumps(dict(who=who,stage=stage,step=k,kind=kind,epoch=time.time()))+'\n')
    return emit

def refine(p,d,guard,emit):
    q=A.ORIGINAL_FROZEN(p,p['small_replay_initial']['path'])
    head=q.replay_head;head.requires_grad_(True)
    # Same ID-balanced, equal old/new objective as T699, now evaluated on the
    # complete frozen finite dataset instead of minibatches.
    lengths=torch.tensor([r['length'] for r in d['records']],dtype=torch.float64)
    group=d['groups'];weight=torch.where(group<403,1/(403*lengths[group]),1/(7*lengths[group]))/4
    optimizer=torch.optim.LBFGS(head.parameters(),lr=1,max_iter=1,max_eval=25,history_size=20,
                               tolerance_grad=1e-12,tolerance_change=1e-15,line_search_fn='strong_wolfe')
    evaluations=0;accepted=0;budget_reached=False;curve=[]
    class EvaluationBudget(Exception):pass
    def closure():
        nonlocal evaluations
        guard()
        if evaluations>=80:raise EvaluationBudget()
        emit('BEFORE',evaluations,'LBFGS_gradient_evaluation')
        optimizer.zero_grad(set_to_none=True);loss=torch.zeros((),dtype=torch.float64)
        # Sum per chunk in fixed order; no trajectory-length or minibatch bias.
        for start in range(0,len(group),2048):
            guard();prediction=head(d['x'][start:start+2048])
            term=((prediction-d['y'][start:start+2048]).square()*weight[start:start+2048,None]).sum()
            term.backward();loss+=term.detach()
        assert torch.isfinite(loss) and all(torch.isfinite(v.grad).all() for v in head.parameters())
        gradient=max(float(v.grad.abs().max()) for v in head.parameters())
        curve.append(dict(evaluation=evaluations,loss=float(loss),gradient_inf=gradient))
        emit('AFTER',evaluations,'LBFGS_gradient_evaluation');evaluations+=1
        return loss
    for step in range(64):
        guard();before=copy.deepcopy(head.state_dict());before_opt=copy.deepcopy(optimizer.state_dict())
        try:optimizer.step(closure)
        except EvaluationBudget:
            head.load_state_dict(before);optimizer.load_state_dict(before_opt);budget_reached=True;break
        accepted+=1
        if curve[-1]['gradient_inf']<=1e-12:break
        if evaluations>=80:budget_reached=True;break
    head.requires_grad_(False);q.requires_grad_(False)
    return q,dict(state_hash=digest(q.state_dict()),optimizer_hash=digest(optimizer.state_dict()),
                  gradient_evaluations=evaluations,accepted_outer_steps=accepted,budget_reached=budget_reached,
                  last_evaluated_gradient_inf=curve[-1]['gradient_inf'],curve=curve,
                  optimizer='fullbatch_LBFGS_strong_wolfe',objective='same T699 equal-old-new ID-balanced residual MSE',
                  finite_optimization_pilot_not_BC_optimum=True)

def metrics(p,q,d,guard,quantized):
    torch.set_num_threads(1);os.sched_setaffinity(0,{15});bad_old=set();bad_new=set();oldcount=newcount=0
    with torch.no_grad():
        for i,(f,mu,target,group) in enumerate(zip(d['x'],d['mu'],d['commands'],d['groups'])):
            if i%128==0:guard()
            if quantized:
                phi=A.P.basis(f[None],q.new_centers.double(),q.new_radii)
                value=mu[None]+phi@q.new_weights if bool(phi.any()) else mu[None]
            else:value=mu[None]+q.replay_head(f[None])
            if i%4096==0:assert torch.equal(value,q.location(f[:64][None],f[64:][None]))
            cmd=value.clamp(-1,1).float()[0];same=A.Old.Existing.same_float32(cmd.numpy(),target.float().numpy())
            if not same:
                if int(group)<403:bad_old.add(d['records'][int(group)]['id']);oldcount+=1
                else:bad_new.add(d['records'][int(group)]['id']);newcount+=1
    if quantized:
        assert not bad_old and not bad_new,'Fixed fp32 reconstruction command gate failed; no rescue'
        # Use full deployment owncarry trajectory path after the factored check.
        A.Old.Existing.verify_commands(p,q,guard)
    state=q.state_dict();parent=A.P.frozen(p,p['initial']['path'])
    parent_bytes=sum(v.numel()*v.element_size() for v in parent.state_dict().values())
    total=sum(v.numel()*v.element_size() for v in state.values())
    row=p['protected403_rows'][0];a=A.Old.Existing.L.T.R.E.N.S.arrays(row['path'])
    x=torch.tensor(a['states'][:64],dtype=torch.float64);q.sample=False;q.rng=None;q.reset_noise();h=None;lat=[]
    with torch.no_grad():
        for i in range(320):
            start=time.perf_counter_ns();_,_,h,_=q.step(x[i%len(x):i%len(x)+1],h)
            if i>=64:lat.append((time.perf_counter_ns()-start)/1e6)
    q.reset_noise()
    return dict(old_command_ids_changed=sorted(bad_old),new_command_ids_changed=sorted(bad_new),
        old_decisions_changed=oldcount,new_decisions_changed=newcount,old_decisions=81956,new_decisions=1737,
        complete_410_owncarry_command_gate=quantized,total_tensor_bytes=total,added_tensor_bytes=total-parent_bytes,
        common_parent_bytes=parent_bytes,batch1_CPU15_median_ms=float(np.median(lat)),batch1_CPU15_p95_ms=float(np.percentile(lat,95)),
        latency_warmup=64,latency_samples=256,peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        peak_includes_cached_training_data_and_Python=True,decoded_center_workspace_counted=True)

def main(mode):
    # `run` may be the historical generic dependency; use the unique controller.
    import importlib.util
    spec=importlib.util.spec_from_file_location('t701_control_for_fit',HERE/'run.py')
    C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
    p=C.read(OUT/'protocol.json');C.verify(p,force=True);d=data();torch.use_deterministic_algorithms(True)
    deadline=min(time.time()+(1200 if mode=='main' else 900),p['absolute_learning_stop'] if mode=='main' else p['absolute_all_stop'])-15
    def guard():
        assert time.time()<deadline
        C.resource_wait(deadline)
    (OUT/'fit').mkdir(exist_ok=True);torch.set_num_threads(8);os.sched_setaffinity(0,set(range(8,16)))
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b')
    import fcntl;fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    who='live' if mode=='main' else 'independent';result={}
    try:
        for name in ['quant32','replay_refined108']:
            torch.set_num_threads(8);os.sched_setaffinity(0,set(range(8,16)))
            construct=(lambda e:A.construct(p,d,guard,e)) if name=='quant32' else (lambda e:refine(p,d,guard,e))
            q,info=construct(event(who))
            if mode=='main':
                other,check=construct(event('shadow'));assert info==check and digest(q.state_dict())==digest(other.state_dict())
                del other
                model=A.export(q,p,OUT/'fit'/(name+'.pt.gz'));loaded=A.frozen(p,model['path'])
                assert digest(loaded.state_dict())==digest(q.state_dict());del loaded
                measured=metrics(p,q,d,guard,name=='quant32')
                result[name]=dict(model=model,construction=info,metrics=measured)
                C.save(OUT/'fit/summary.json',dict(state='COMPLETE' if len(result)==2 else 'ACTIVE',finals=result))
            else:
                saved=C.read(OUT/'fit/summary.json')['finals'][name]
                assert info==saved['construction']
                loaded=A.frozen(p,saved['model']['path']);assert digest(loaded.state_dict())==digest(q.state_dict())
                measured=metrics(p,q,d,guard,name=='quant32')
                for k,v in saved['metrics'].items():
                    if k.startswith(('batch1','peak_')):continue
                    assert measured[k]==v,(name,k)
                result[name]=dict(independent_state_exact=True,independent_construction_exact=True,command_gate=measured)
        C.verify(p,force=True)
        if mode=='audit':C.save(OUT/'fit/audit.json',dict(state='PASS',finals=result,new_native=0,new_Adam=0,new_RL=0))
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main(sys.argv[1])
