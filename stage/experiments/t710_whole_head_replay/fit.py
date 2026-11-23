"""Matched full-head optimization: nominal versus fixed learner-state replay."""
from pathlib import Path
import copy, fcntl, gzip, importlib.util, io, json, os, sys, time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('t710_fit_controller',HERE/'run.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
A,torch,OUT=C.A,C.A.torch,C.OUT
sys.path.insert(0,str(ROOT/'stage/experiments/t701_compact_retention'))
import t701_actor as Legacy


def datasets(p,data,guard):
    q=A.frozen(p,p['repair_initial']['path']);teacher=A.P.frozen(p,p['initial']['path'])
    producer={name:A.frozen(p,str(ROOT/'stage/runs/t708_agem_replay/fit'/(name+'.pt.gz'))) for name in ['agem_mean','agem_worst']}
    for model in [q,*producer.values()]:
        for key,value in teacher.state_dict().items():assert torch.equal(model.state_dict()[key],value),key
    originals=data['records'];lengths=torch.tensor([r['length'] for r in originals],dtype=torch.float64);group=data['groups']
    weight=torch.where(group<403,1/(2*403*lengths[group]),1/(2*7*lengths[group]))
    selected=set(r['job_id'] for r in p['induced_selection']['rows']);index={r['id']:i for i,r in enumerate(originals[:403])}
    mixture=weight.clone()
    for key in selected:mixture[group==index[key]]*=.5
    X=[];W=[];tapes=[]
    with torch.no_grad():
        h=q.replay_head;z=h[1](h[0](data['x']));z=torch.cat([z,torch.ones(len(z),1,dtype=torch.float64)],1)
        old=group<403;G=z[old].T@(weight[old,None]*z[old]);computed_rho=1e-10*float(torch.trace(G))/109
        rho=C.read(ROOT/'stage/runs/t709_induced_replay/fit/summary.json')['construction']['rho']
        assert abs(rho-computed_rho)<=1e-12*max(abs(rho),1e-30)
        for row in p['induced_selection']['rows']:
            guard();assert C.sha(row['path'])==row['sha256']
            arr=A.Old.Existing.L.T.R.E.N.S.arrays(row['path']);assert {k:A.digest(v) for k,v in arr.items()}==row['array_hashes']
            assert not arr['ownership'].any() and not arr['stochastic'].any()
            f,mu=A.P.trajectory(teacher,arr,row['job_id'],None,guard)
            model=producer[row['arm']];assert model.phash()==row['policy_hash']
            for i,feature in enumerate(f):
                if i%128==0:guard()
                action=model.location(feature[:64][None],feature[64:][None]).clamp(-1,1).float()
                assert A.Old.Existing.same_float32(action.numpy()[0],arr['actions'][i])
            X.append(f);W.append(torch.full((len(f),),.25/(2*403*len(f)),dtype=torch.float64))
            tapes.append(dict(id=row['job_id'],arm=row['arm'],length=len(f),feature_hash=A.digest(f),teacher_location_hash=A.digest(mu),trajectory_sha256=row['sha256']))
    extra=torch.cat(X);ew=torch.cat(W)
    nominal=(data['x'],data['y'],weight)
    induced=(torch.cat([data['x'],extra]),torch.cat([data['y'],torch.zeros((len(extra),2),dtype=torch.float64)]),torch.cat([mixture,ew]))
    # Sum weights per output is1: old half + source half. This equals loss
    # old_mean+source_mean when both output coordinates are summed.
    assert abs(float(weight.sum())-1.)<1e-12 and abs(float(induced[2].sum())-1.)<1e-12
    assert len(selected)==51 and len(tapes)==102 and len(extra)==26188
    return dict(nominal=nominal,induced=induced),dict(rho=rho,original_feature_hash=A.digest(data['x']),induced_feature_hash=A.digest(extra),nominal_weights_hash=A.digest(weight),mixture_weights_hash=A.digest(induced[2]),tapes=tapes,old_ID_total_weights_unchanged=True)


def optimize(q,dataset,rho,guard,emit,arm):
    head=q.replay_head;head.requires_grad_(True);start=copy.deepcopy(head.state_dict())
    W0=torch.cat([head[2].weight.T,head[2].bias[None]],0).detach().clone()
    opt=torch.optim.LBFGS(head.parameters(),lr=1,max_iter=1,max_eval=25,history_size=20,tolerance_grad=1e-12,tolerance_change=1e-15,line_search_fn='strong_wolfe')
    x,y,weight=dataset;calls=0;accepted=0;attempts=0;curve=[];budget=False
    class EvaluationBudget(Exception):pass
    def closure():
        nonlocal calls
        guard()
        if calls>=160:raise EvaluationBudget()
        emit('BEFORE',calls,'gradient_evaluation',arm);calls+=1;opt.zero_grad(set_to_none=True);loss=torch.zeros((),dtype=torch.float64)
        for off in range(0,len(x),2048):
            guard();value=((head(x[off:off+2048])-y[off:off+2048]).square()*weight[off:off+2048,None]).sum();value.backward();loss+=value.detach()
        W=torch.cat([head[2].weight.T,head[2].bias[None]],0);reg=rho*(W-W0).square().sum();reg.backward();loss+=reg.detach()
        assert torch.isfinite(loss) and all(torch.isfinite(v.grad).all() for v in head.parameters())
        inf=max(float(v.grad.abs().max()) for v in head.parameters())
        curve.append(dict(evaluation=calls,loss=float(loss),gradient_inf=inf,head_hash=A.digest(head.state_dict())))
        emit('AFTER',calls-1,'gradient_evaluation',arm);return loss
    for k in range(128):
        guard();before=copy.deepcopy(head.state_dict());before_opt=copy.deepcopy(opt.state_dict())
        emit('BEFORE',k,'optimizer_attempt',arm);attempts+=1
        try:opt.step(closure)
        except EvaluationBudget:
            head.load_state_dict(before);opt.load_state_dict(before_opt);budget=True;emit('AFTER',k,'optimizer_attempt',arm);break
        emit('AFTER',k,'optimizer_attempt',arm);accepted+=1
        if curve[-1]['gradient_inf']<=1e-12:break
        if calls>=160:budget=True;break
    head.requires_grad_(False);q.requires_grad_(False)
    with torch.no_grad():
        final=sum(float(((head(x[off:off+2048])-y[off:off+2048]).square()*weight[off:off+2048,None]).sum()) for off in range(0,len(x),2048))
    assert all(torch.isfinite(v).all() for v in head.state_dict().values())
    return q,dict(initial_head_hash=A.digest(start),final_head_hash=A.digest(head.state_dict()),optimizer_hash=A.digest(opt.state_dict()),gradient_evaluations=calls,optimizer_attempts=attempts,accepted_outer_steps=accepted,budget_reached=budget,final_data_loss=final,curve=curve,global_optimum_not_claimed=True)


def event(who):
    def emit(stage,step,kind,arm):
        with (OUT/'fit/fit_calls.jsonl').open('a') as f:f.write(json.dumps(dict(who=who,stage=stage,step=step,kind=kind,arm=arm,epoch=time.time()))+'\n')
    return emit


def construct(p,data,guard,who):
    torch.set_num_threads(1);ds,info=datasets(p,data,guard);models={};construction={}
    for arm in ['nominal','induced']:
        torch.set_num_threads(8);q=A.frozen(p,p['repair_initial']['path']);initial=copy.deepcopy(q.state_dict())
        model,stats=optimize(q,ds[arm],info['rho'],guard,event(who),arm)
        for key,value in initial.items():
            if not key.startswith('replay_head.'):assert torch.equal(model.state_dict()[key],value),key
        stats['state_hash']=A.digest(model.state_dict());models[arm]=model;construction[arm]=stats
    assert construction['nominal']['initial_head_hash']==construction['induced']['initial_head_hash']
    return models,dict(data=info,arms=construction,whole_head_trainable_parameters=20198,common_parent_frozen=True,new_Adam=0,new_RL=0)


def main(mode):
    p=C.read(OUT/'protocol.json');C.verify(p,force=True);end=min(time.time()+(2400 if mode=='main' else 1200),p['absolute_learning_stop'] if mode=='main' else p['absolute_all_stop'])-15
    def guard():assert time.time()<end;C.verify(p)
    torch.use_deterministic_algorithms(True);os.sched_setaffinity(0,set(range(8,16)))
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        data=torch.load(io.BytesIO(gzip.decompress((ROOT/'stage/runs/t699_matched_value/DATA.pt.gz').read_bytes())),map_location='cpu',weights_only=False)
        models,info=construct(p,data,guard,'live' if mode=='main' else 'independent')
        if mode=='main':
            shadow,check=construct(p,data,guard,'shadow');assert info==check;del shadow
        fs=importlib.util.spec_from_file_location('t710_existing_metrics',ROOT/'stage/experiments/t701_compact_retention/fit.py');F=importlib.util.module_from_spec(fs);fs.loader.exec_module(F);result={}
        for arm,q in models.items():
            metric=F.metrics(p,q,data,guard,False);assert metric['added_tensor_bytes']==161584
            if mode=='main':
                model=Legacy.Old.export(q,p,OUT/'fit'/(arm+'.pt.gz'));assert A.digest(A.frozen(p,model['path']).state_dict())==info['arms'][arm]['state_hash'];result[arm]=dict(model=model,metrics=metric)
            else:
                saved=C.read(OUT/'fit/summary.json');assert saved['construction']==info
                assert A.digest(A.frozen(p,saved['finals'][arm]['model']['path']).state_dict())==info['arms'][arm]['state_hash']
                for key,value in saved['finals'][arm]['metrics'].items():
                    if not key.startswith(('batch1','peak_')):assert metric[key]==value,key
                result[arm]=dict(independent_state_and_optimizer_and_all_closures_exact=True)
        C.verify(p,force=True)
        if mode=='main':C.save(OUT/'fit/summary.json',dict(state='COMPLETE',finals=result,construction=info,new_Adam=0,new_RL=0))
        else:C.save(OUT/'fit/audit.json',dict(state='PASS',finals=result,independent_tapes_teacher_history_and_fullhead_optimization_exact=True,new_native=0))
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main(sys.argv[1])
