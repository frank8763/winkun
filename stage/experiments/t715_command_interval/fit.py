"""One command-interval supervised fit with exact state and optimizer reconstruction."""
from pathlib import Path
import copy, fcntl, gzip, importlib.util, io, json, os, resource, sys, time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('t715_fit_controller',HERE/'run.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
A,torch,OUT=C.A,C.A.torch,C.OUT
sys.path.insert(0,str(ROOT/'stage/experiments/t701_compact_retention'))
import t701_actor as Legacy

def event(who,arm):
    def emit(stage,step,kind):
        with (OUT/'fit/fit_calls.jsonl').open('a') as f:f.write(json.dumps(dict(who=who,arm=arm,stage=stage,step=step,kind=kind,epoch=time.time()))+'\n')
    return emit

def augmented_data(p,d,arm,guard):
    parent=A.P.frozen(p,p['initial']['path']);base=A.frozen(p,p['repair_initial']['path']);initial=A.digest(parent.state_dict())
    X=[];Y=[];MU=[];CMD=[];GROUP=[];records=list(d['records']);tapes=[];saturation=0;raw_error=0.;mismatches=0;coordinates=0
    for k,row in enumerate(p['source_selection']['rows']):
        guard();r=row['source' if arm=='success' else 'baseline'];assert C.sha(r['path'])==r['sha256']
        arrays=A.Old.Existing.L.T.R.E.N.S.arrays(r['path']);assert not arrays['ownership'].any() and not arrays['stochastic'].any()
        assert {key:A.digest(v) for key,v in arrays.items()}==r['array_hashes']
        C.M.N.N.P.frozen=A.frozen;C.M.N.BASE_INSPECT(p,arrays,r)
        producer=A.frozen(p,r['request']['model']);assert producer.phash()==r['policy_hash']
        assert all(torch.equal(v,producer.state_dict()[key]) for key,v in parent.state_dict().items()),'Source shared-parent identity'
        A.Old.Existing.trajectory(producer,arrays,r['job_id'],arrays['actions'],guard)
        features,mu=A.Old.Existing.trajectory(parent,arrays,r['job_id'],None,guard)
        commands=torch.tensor(arrays['actions'],dtype=torch.float64);target=commands-mu
        assert len(features)==len(commands);index=410+k
        with torch.no_grad():
            predicted=torch.cat([base.replay_head(x[None]) for x in features])
            raw_error+=float((predicted-target).square().sum());coordinates+=commands.numel()
            mismatches+=int(((mu+predicted).clamp(-1,1).float()!=commands.float()).sum())
        X.append(features);Y.append(target);MU.append(mu);CMD.append(commands);GROUP.append(torch.full((len(features),),index,dtype=torch.int64))
        records.append(dict(id=row['job_id'],length=len(features),source_path=r['path'],sha256=r['sha256'],successful_source=arm=='success'))
        saturation+=int((commands.abs()==1).sum());tapes.append(dict(id=row['job_id'],raw_sha256=r['sha256'],features=A.digest(features),targets=A.digest(target),decisions=len(features),producer_policy=r['policy_hash']))
    assert A.digest(parent.state_dict())==initial
    result={key:torch.cat([d[key],*parts]) for key,parts in [('x',X),('y',Y),('mu',MU),('commands',CMD),('groups',GROUP)]};result['records']=records
    return result,dict(side=arm,source_tapes=tapes,source_count=len(tapes),added_decisions=sum(t['decisions'] for t in tapes),initial_raw_residual_MSE=raw_error/coordinates,initial_executed_command_coordinate_mismatches=mismatches,saturated_command_coordinates=saturation,executed_float32_targets_lifted_float64=True,self_is_not_zero_gradient_due_clipping_and_rounding=True,occupancy_and_behavior_bundled=True)

def interval_error(mean,command):
    assert torch.isfinite(mean).all() and torch.isfinite(command).all() and (command.abs()<=1).all()
    error=torch.where(command==1,torch.relu(1-mean),torch.where(command==-1,torch.relu(mean+1),mean-command))
    return error.square()

def replay(p,d,arm,guard,emit):
    torch.set_num_threads(8);C.M.N.N.G._real_affinity(0,set(range(8,16)))
    for task in Path('/proc/self/task').iterdir():C.M.N.N.G._real_affinity(int(task.name),set(range(8,16)))
    q=A.frozen(p,p['repair_initial']['path']);before=copy.deepcopy(q.state_dict());head=q.replay_head;head.requires_grad_(True)
    g=d['groups'];lengths=torch.tensor([r['length'] for r in d['records']],dtype=torch.float64)
    weight=torch.where(g<403,.5/(2*403*lengths[g]),torch.where(g<410,.25/(2*7*lengths[g]),.25/(2*len(p['source_selection']['rows'])*lengths[g])))
    assert abs(float(weight[g<403].sum())-.25)<1e-12 and abs(float(weight[(g>=403)&(g<410)].sum())-.125)<1e-12 and abs(float(weight[g>=410].sum())-.125)<1e-12
    initial_geometry={}
    with torch.no_grad():
        for label,mask in [('old403',g<403),('original7',(g>=403)&(g<410)),('success18',g>=410)]:
            mean=d['mu'][mask]+head(d['x'][mask]);commands=d['commands'][mask];raw=head(d['x'][mask])-d['y'][mask]
            equivalent=((commands==1)&(mean>=1))|((commands==-1)&(mean<=-1))
            initial_geometry[label]=dict(decisions=int(mask.sum()),saturated_coordinates=int((commands.abs()==1).sum()),correct_side_saturated_coordinates=int(equivalent.sum()),raw_MSE=float(raw.square().mean()),interval_MSE=float(interval_error(mean,commands).mean()),raw_parent_to_executed_max=float((d['mu'][mask]-commands).abs().max()) if label=='old403' else None,unsaturated_old_rounding_max=float((d['mu'][mask]-commands)[commands.abs()<1].abs().max()) if label=='old403' else None)
    opt=torch.optim.LBFGS(head.parameters(),lr=1,max_iter=1,max_eval=25,history_size=20,tolerance_grad=1e-12,tolerance_change=1e-15,line_search_fn='strong_wolfe')
    calls=attempts=accepted=0;curve=[];budget=False
    class Budget(Exception):pass
    def closure():
        nonlocal calls
        guard()
        if calls>=p['gradient_cap_per_arm']:raise Budget()
        emit('BEFORE',calls,'gradient');calls+=1;opt.zero_grad(set_to_none=True);loss=torch.zeros((),dtype=torch.float64)
        for off in range(0,len(g),2048):
            guard();term=(interval_error(d['mu'][off:off+2048]+head(d['x'][off:off+2048]),d['commands'][off:off+2048])*weight[off:off+2048,None]).sum();term.backward();loss+=term.detach()
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
            guard();final_data_loss+=float((interval_error(d['mu'][off:off+2048]+head(d['x'][off:off+2048]),d['commands'][off:off+2048])*weight[off:off+2048,None]).sum())
    final_geometry={}
    with torch.no_grad():
        for label,mask in [('old403',g<403),('original7',(g>=403)&(g<410)),('success18',g>=410)]:
            mean=d['mu'][mask]+head(d['x'][mask]);command=d['commands'][mask];raw=head(d['x'][mask])-d['y'][mask]
            final_geometry[label]=dict(raw_MSE=float(raw.square().mean()),interval_MSE=float(interval_error(mean,command).mean()),active_interval_coordinates=int((interval_error(mean,command)>0).sum()),saturated_correct_side_coordinates=int((((command==1)&(mean>=1))|((command==-1)&(mean<=-1))).sum()),batch_diagnostic_command_coordinate_mismatches=int((mean.clamp(-1,1).float()!=command.float()).sum()),raw_absolute_max=float(mean.abs().max()))
    return q,dict(alpha_old=.5,alpha_original7=.25,alpha_added18=.25,initial_head_hash=A.digest({k:v for k,v in before.items() if k.startswith('replay_head.')}),state_hash=A.digest(q.state_dict()),optimizer_hash=A.digest(opt.state_dict()),weight_hash=A.digest(weight),gradient_evaluations=calls,optimizer_attempts=attempts,accepted_outer_steps=accepted,budget_reached=budget,curve=curve,final_exported_state_data_loss=final_data_loss,full_head_parameters=20198,initial_geometry=initial_geometry,final_geometry=final_geometry,objective='all_categories_executed_command_interval',no_ridge=True,no_convergence_claim=True)

def metrics(p,q,d,guard,arm):
    torch.set_num_threads(1);C.M.N.N.G._real_affinity(0,{15})
    # Only original410 data are measured by the existing finite data metric;
    # new18 are separately evaluated in canonical native episodes.
    original={k:v[:83693] for k,v in d.items() if k in ['x','y','mu','commands','groups']};original['records']=d['records'][:410]
    z=importlib.util.spec_from_file_location('t715_metrics',ROOT/'stage/experiments/t701_compact_retention/fit.py');F=importlib.util.module_from_spec(z);z.loader.exec_module(F)
    m=F.metrics(p,q,original,guard,False);assert m['added_tensor_bytes']==161584;return m

def main(mode,arm):
    p=C.root_plan();C.verify(p,force=True);assert arm in p['arms'];torch.use_deterministic_algorithms(True);os.sched_setaffinity(0,set(range(8,16)))
    bound=1200 if mode=='main' else 600
    end=min(time.time()+bound,p['absolute_learning_stop'] if mode=='main' else p['absolute_all_stop'])-20;last=[0.]
    def guard():
        assert time.time()<end,'Fit action/absolute deadline'
        if time.time()-last[0]>2:C.verify(p);last[0]=time.time()
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        d=torch.load(io.BytesIO(gzip.decompress((ROOT/'stage/runs/t699_matched_value/DATA.pt.gz').read_bytes())),map_location='cpu',weights_only=False)
        torch.set_num_threads(1);d,tapes=augmented_data(p,d,'success',guard)
        def construct(who):
            model,info=replay(p,d,arm,guard,event(who,arm));info['data']=tapes;return model,info
        q,info=construct('live' if mode=='main' else 'independent')
        if mode=='main':
            # Immediate, durable export before spending any shadow work.
            model=Legacy.Old.export(q,p,OUT/'fit'/arm/'candidate.pt.gz')
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
