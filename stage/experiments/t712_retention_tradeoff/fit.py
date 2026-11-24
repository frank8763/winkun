"""Three equal-compute full-head fits and an optional compact support ablation."""
from pathlib import Path
import copy, fcntl, gzip, importlib.util, io, json, os, resource, sys, time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('t712_fit_controller',HERE/'run.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
A,torch,OUT=C.A,C.A.torch,C.OUT
sys.path.insert(0,str(ROOT/'stage/experiments/t701_compact_retention'))
import t701_actor as Legacy

def event(who,arm):
    def emit(stage,step,kind):
        with (OUT/'fit/fit_calls.jsonl').open('a') as f:f.write(json.dumps(dict(who=who,arm=arm,stage=stage,step=step,kind=kind,epoch=time.time()))+'\n')
    return emit

def plain_radii(q,guard):
    centers=q.projected_centers;nearest=torch.full((len(centers),),float('inf'),dtype=torch.float64)
    for off in range(0,len(centers),128):
        guard();count=min(128,len(centers)-off)
        distance=((centers[:,None]-centers[off:off+128])**2).sum(-1)
        distance[torch.arange(off,off+count),torch.arange(count)]=float('inf')
        nearest=torch.minimum(nearest,distance.min(-1).values)
    stored=(.49*nearest.sqrt()).float()
    assert torch.isfinite(stored).all() and (stored>1e-10).all() and (stored.double()<.5*nearest.sqrt()).all()
    return stored

def construct_plain(p,d,guard,emit):
    torch.set_num_threads(1);q=A.frozen(p,p['compact_reference']['path']);before=copy.deepcopy(q.state_dict())
    radii=plain_radii(q,guard);assert A.digest(radii)==C.read(C.DOC/'GEOMETRY_CHECK.json')['plain_radii_hash']
    emit('BEFORE',0,'radius');q.projected_radii.copy_(radii);emit('AFTER',0,'radius')
    assert all(torch.equal(v,q.state_dict()[k]) for k,v in before.items() if k!='projected_radii')
    for off in range(0,1737,128):
        guard();K=A.P.basis(q.projected_centers[off:off+128],q.projected_centers,radii.double())
        i=torch.arange(len(K));j=torch.arange(off,off+len(K));assert torch.equal(K[i,j],torch.ones(len(K),dtype=torch.float64));K[i,j]=0;assert not K.any()
    # Old equality is intentionally NOT an eligibility condition for plain.
    A.Old.Existing.verify_commands(dict(p,protected403_rows=[]),q,guard)
    return q,dict(state_hash=A.digest(q.state_dict()),radii_hash=A.digest(radii),source_K_exact_identity=True,source7_owncarry_commands_exact=True,
        unchanged_except_radius=True,old_equality_not_an_admission_gate=True,rule='float32(.49*nearest_other_source_distance)',new_Adam=0,new_RL=0)

def replay(p,d,arm,guard,emit):
    torch.set_num_threads(8);os.sched_setaffinity(0,set(range(8,16)))
    q=A.frozen(p,p['repair_initial']['path']);before=copy.deepcopy(q.state_dict());head=q.replay_head;head.requires_grad_(True)
    alpha=p['replay_alphas'][arm];g=d['groups'];lengths=torch.tensor([r['length'] for r in d['records']],dtype=torch.float64)
    weight=torch.where(g<403,alpha/(2*403*lengths[g]),(1-alpha)/(2*7*lengths[g]))
    assert abs(float(weight[g<403].sum())-alpha/2)<1e-12 and abs(float(weight[g>=403].sum())-(1-alpha)/2)<1e-12
    opt=torch.optim.LBFGS(head.parameters(),lr=1,max_iter=1,max_eval=25,history_size=20,tolerance_grad=1e-12,tolerance_change=1e-15,line_search_fn='strong_wolfe')
    calls=attempts=accepted=0;curve=[];budget=False
    class Budget(Exception):pass
    def closure():
        nonlocal calls
        guard()
        if calls>=p['gradient_cap_per_arm']:raise Budget()
        emit('BEFORE',calls,'gradient');calls+=1;opt.zero_grad(set_to_none=True);loss=torch.zeros((),dtype=torch.float64)
        for off in range(0,len(g),2048):
            guard();term=((head(d['x'][off:off+2048])-d['y'][off:off+2048]).square()*weight[off:off+2048,None]).sum();term.backward();loss+=term.detach()
        assert torch.isfinite(loss) and all(torch.isfinite(v.grad).all() for v in head.parameters())
        curve.append(dict(evaluation=calls,loss=float(loss),gradient_inf=max(float(v.grad.abs().max()) for v in head.parameters()),state_hash=A.digest(head.state_dict())))
        emit('AFTER',calls-1,'gradient');return loss
    for step in range(p['optimizer_cap_per_arm']):
        guard();saved=copy.deepcopy(head.state_dict());saved_opt=copy.deepcopy(opt.state_dict())
        emit('BEFORE',step,'optimizer');attempts+=1
        try:opt.step(closure)
        except BaseException as error:
            # Both explicit closure-budget and deadline failures restore the
            # last completed state, never publish a line-search trial point.
            head.load_state_dict(saved);opt.load_state_dict(saved_opt);emit('AFTER',step,'optimizer')
            if isinstance(error,Budget):budget=True;break
            raise
        emit('AFTER',step,'optimizer');accepted+=1
        if calls>=p['gradient_cap_per_arm']:budget=True;break
        if curve[-1]['gradient_inf']<=1e-12:break
    head.requires_grad_(False);q.requires_grad_(False)
    assert all(torch.equal(v,q.state_dict()[k]) for k,v in before.items() if not k.startswith('replay_head.'))
    with torch.no_grad():
        final_data_loss=0.
        for off in range(0,len(g),2048):
            guard();final_data_loss+=float(((head(d['x'][off:off+2048])-d['y'][off:off+2048]).square()*weight[off:off+2048,None]).sum())
    return q,dict(alpha_old=alpha,initial_head_hash=A.digest({k:v for k,v in before.items() if k.startswith('replay_head.')}),state_hash=A.digest(q.state_dict()),optimizer_hash=A.digest(opt.state_dict()),weight_hash=A.digest(weight),gradient_evaluations=calls,optimizer_attempts=attempts,accepted_outer_steps=accepted,budget_reached=budget,curve=curve,final_exported_state_data_loss=final_data_loss,full_head_parameters=20198,no_ridge=True,no_convergence_claim=True)

def metrics(p,q,d,guard,arm):
    torch.set_num_threads(1);os.sched_setaffinity(0,{15})
    if arm!='plain':
        s=importlib.util.spec_from_file_location('t712_existing_metrics',ROOT/'stage/experiments/t701_compact_retention/fit.py');F=importlib.util.module_from_spec(s);s.loader.exec_module(F)
        measured=F.metrics(p,q,d,guard,False);assert measured['added_tensor_bytes']==161584;return measured
    changed_old=set();changed_source=set();old_decisions=source_decisions=0
    with torch.no_grad():
        for off in range(0,len(d['x']),128):
            guard();x=d['x'][off:off+128];z=A.project(x,q.projection);phi=A.P.basis(z,q.projected_centers,q.projected_radii.double())
            correction=torch.cat([row[None]@q.projected_weights for row in phi]);pred=(d['mu'][off:off+len(x)]+correction).clamp(-1,1).float()
            same=(pred==d['commands'][off:off+len(x)].float()).all(-1)
            for i in torch.where(~same)[0].tolist():
                group=int(d['groups'][off+i]);key=d['records'][group]['id']
                if group<403:changed_old.add(key);old_decisions+=1
                else:changed_source.add(key);source_decisions+=1
        assert not changed_source,'Plain source commands changed'
    parent=A.P.frozen(p,p['initial']['path']);base=sum(v.numel()*v.element_size() for v in parent.state_dict().values());total=sum(v.numel()*v.element_size() for v in q.state_dict().values());assert total-base==157684
    arr=A.Old.Existing.L.T.R.E.N.S.arrays(p['protected403_rows'][0]['path']);q.sample=False;q.rng=None;q.reset_noise();h=None;lat=[]
    with torch.no_grad():
        for i in range(320):
            begin=time.perf_counter_ns();_,_,h,_=q.step(torch.tensor(arr['states'][i%64:i%64+1]),h)
            if i>=64:lat.append((time.perf_counter_ns()-begin)/1e6)
    return dict(old_command_ids_changed=sorted(changed_old),new_command_ids_changed=[],old_decisions_changed=old_decisions,new_decisions_changed=source_decisions,old_equality_not_admission=True,total_tensor_bytes=total,added_tensor_bytes=total-base,common_parent_bytes=base,batch1_CPU15_median_ms=float(A.np.median(lat)),batch1_CPU15_p95_ms=float(A.np.percentile(lat,95)),peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)

def main(mode,arm):
    p=C.root_plan();C.verify(p,force=True);assert arm in p['arms'];torch.use_deterministic_algorithms(True);os.sched_setaffinity(0,set(range(8,16)))
    bound=(1200 if arm=='plain' else 1500) if mode=='main' else 900
    end=min(time.time()+bound,p['absolute_learning_stop'] if mode=='main' else p['absolute_all_stop'])-20;last=[0.]
    def guard():
        assert time.time()<end,'Fit action/absolute deadline'
        if time.time()-last[0]>2:C.verify(p);last[0]=time.time()
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        d=torch.load(io.BytesIO(gzip.decompress((ROOT/'stage/runs/t699_matched_value/DATA.pt.gz').read_bytes())),map_location='cpu',weights_only=False)
        def construct(who):return construct_plain(p,d,guard,event(who,arm)) if arm=='plain' else replay(p,d,arm,guard,event(who,arm))
        q,info=construct('live' if mode=='main' else 'independent')
        if mode=='main':
            # Immediate, durable export before spending any shadow work.
            model=A.export(q,p,OUT/'fit'/arm/'candidate.pt.gz') if arm=='plain' else Legacy.Old.export(q,p,OUT/'fit'/arm/'candidate.pt.gz')
            C.save(OUT/'fit'/arm/'PROVISIONAL.json',dict(model=model,construction=info,eligibility='PENDING_SHADOW_AND_INDEPENDENT_AUDIT'))
            shadow,other=construct('shadow');assert info==other and A.digest(shadow.state_dict())==A.digest(q.state_dict());del shadow
        else:
            saved=C.read(OUT/'fit/summary.json')['finals'][arm];assert info==saved['construction'];model=saved['model']
        restored=A.frozen(p,model['path']);assert A.digest(restored.state_dict())==A.digest(q.state_dict());del restored
        measured=metrics(p,q,d,guard,arm)
        if mode=='main':
            f=OUT/'fit/summary.json';summary=C.read(f) if f.exists() else dict(state='COMPLETE_AVAILABLE_FINALS',finals={})
            summary['finals'][arm]=dict(model=model,construction=info,metrics=measured);C.save(f,summary)
        else:
            for key,value in saved['metrics'].items():
                if not key.startswith(('batch1','peak_')):assert measured[key]==value,key
            f=OUT/'fit/audit.json';audit=C.read(f) if f.exists() else dict(state='PASS',finals={})
            audit['finals'][arm]=dict(state='PASS',independent_state_optimizer_curves_and_metrics_exact=True,new_native=0);C.save(f,audit)
        C.verify(p,force=True)
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main(sys.argv[1],sys.argv[2])
