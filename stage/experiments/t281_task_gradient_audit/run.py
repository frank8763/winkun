"""Actual per-reset policy gradients; no optimizer or native execution."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t281_task_gradient_audit'
sys.path.insert(0,str(ROOT/'stage/experiments/t280_local_cost'))
from t280_core import *
import copy,json,os,shutil,signal,time,traceback


def norm(g):return float(torch.linalg.vector_norm(g.double()))
def gram(gg):
    return torch.stack(gg).double()@torch.stack(gg).double().T
def projected(gg,seed):
    rng=np.random.default_rng(seed);out=[];records=[]
    for i,gi in enumerate(gg):
        g=gi.clone()
        for j in rng.permutation(len(gg)):
            if i==j:continue
            dot=torch.dot(g,gg[j]);den=gg[j].square().sum().clamp_min(1e-30)
            if float(dot)<0:
                amount=dot/den;g-=amount*gg[j];records.append(dict(task=i,against=int(j),coefficient=float(amount)))
        out.append(g)
    return out,records


def main():
    assert not (OUT/'status.json').exists(),'No repeat or data grid'
    plan=json.loads((HERE/'protocol.json').read_text());start=time.time();result=[];reason='COMPLETE';failure=None;forbidden=[]
    def prohibit(*args,**kwargs):forbidden.append(True);raise RuntimeError('T281 forbids optimizer/native execution')
    torch.optim.Adam.step=prohibit;S.native_setup=prohibit
    def alarm(signum,frame):raise TimeoutError('Registered600s wall deadline')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(plan['wall_seconds'])
    def status(state):C.save(OUT/'status.json',dict(state=state,pid=os.getpid(),completed_batches=len(result),physics_steps=0,optimizer_calls=0,updated_epoch=time.time()))
    try:
        for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
        assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        base=json.loads(Path(plan['source_protocol']).read_text());rows=list(map(json.loads,(ROOT/'stage/runs/t280_local_cost/cases.jsonl').read_text().splitlines()))
        C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[]));status('STARTED')
        for arm,ids in plan['first_batch_ids'].items():
            assert time.time()<start+plan['wall_seconds'] and shutil.disk_usage(ROOT).free>=plan['min_free_bytes'];status(arm+'_DENSITY')
            items=[]
            for name in ids:
                row=next(r for r in rows if r['id']==name)
                with np.load(row['path']) as z:a={k:z[k].copy() for k in z.files}
                assert all(S.digest(v)==row['array_hashes'][k] for k,v in a.items());items.append(dict(row=row,arrays=a))
            q=policy(base);before=S.digest(q.state_dict());params=[v for v in q.parameters() if v.requires_grad];names=[name for name,v in q.named_parameters() if v.requires_grad];dimensions=[v.numel() for v in params]
            data=L.training_batch(items,q);lp,_=L.likelihood(q,data);delta=lp-data['oldlp'];ratio=delta.exp();a=data['advantage'];term=torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a)*data['weights']
            losses=[-term[data['task']==task].sum()/4/base['surrogate_horizon_decisions'] for task in range(4)]
            gg=[torch.cat([v.reshape(-1) for v in torch.autograd.grad(loss,params,retain_graph=True)]).detach() for loss in losses]
            pg,records=projected(gg,base['seed']);direction=torch.stack(pg).mean(0);reference,conflicts=L.project(gg,base['seed'])
            assert torch.equal(direction,reference) and conflicts==len(records)
            gmat=gram(gg);pmat=gram(pg);nn=torch.sqrt(torch.diag(gmat));cos=gmat/(nn[:,None]*nn[None,:]).clamp_min(1e-300)
            mean=torch.stack(gg).mean(0);taskids=sorted({r['row']['source_reset_id'] for r in items})
            allgroups=[];offset=0
            for name,count in zip(names,dimensions):
                sl=slice(offset,offset+count);offset+=count
                allgroups.append(dict(parameter=name,task_norms=[norm(g[sl]) for g in gg],projected_direction_norm=norm(direction[sl])))
            # The real unchanged-parent soft preservation loss has zero residual;
            # an action Jacobian contraction additionally describes first-order
            # sensitivity at the known fragile old task, without changing weights.
            frozen=copy.deepcopy(q).eval().requires_grad_(False);ax,ay,off=anchors(base,frozen)
            rng=np.random.default_rng(base['seed']);ind=np.concatenate([rng.integers(off[t],off[t+1],32) for t in range(8)])
            ret=base['retention_weight']*(q.step(ax[ind])[0]-ay[ind]).square().mean();rg=torch.cat([v.reshape(-1) for v in torch.autograd.grad(ret,params)]).detach();assert norm(rg)==0
            with np.load(base['retention_states']) as z:
                case=int(np.flatnonzero(z['case_ids']=='dev_task6_24')[0]);oldx=z['states'][z['offsets'][case]:z['offsets'][case+1]].copy()
            # Fixed first/middle/last states, not a full trajectory guarantee.
            samples=sorted(set([0,len(oldx)//2,len(oldx)-1]));sensitivity=[]
            unit=direction/max(norm(direction),1e-30)
            for index in samples:
                mode=q.step(torch.tensor(oldx[index:index+1]))[0][0]
                slopes=[]
                for axis in range(2):
                    jac=torch.cat([v.reshape(-1) for v in torch.autograd.grad(mode[axis],params,retain_graph=True)])
                    slopes.append(float(-torch.dot(jac.double(),unit.double())))
                sensitivity.append(dict(row=index,command=mode.detach().numpy().tolist(),command_derivative_along_unit_negative_PCGrad=slopes))
            relative=(nn/nn[0].clamp_min(1e-300)).tolist();assert taskids[0]=='dev6_LEFT_13'
            record=dict(arm=arm,task_order=taskids,losses=[float(x.detach()) for x in losses],density=data['density'],
                gradients_SHA=[S.digest(g) for g in gg],gradient_norms=nn.tolist(),gradient_norm_ratio_to_LEFT=relative,
                Gram_float64=gmat.tolist(),cosine_float64=cos.tolist(),PCGrad_projection_records=records,PCGrad_projected_Gram_float64=pmat.tolist(),
                PCGrad_projected_norms=[norm(g) for g in pg],PCGrad_direction_norm=norm(direction),ordinary_mean_norm=norm(mean),
                PCGrad_norm_to_mean=norm(direction)/max(norm(mean),1e-300),
                task_loss_improvement_first_order_along_negative_PCGrad=[float(torch.dot(g.double(),direction.double())) for g in gg],
                parameter_group_norms=allgroups,soft100_gradient_norm_at_parent=norm(rg),fragile_old_dev_task6_24_three_state_sensitivity=sensitivity,
                weak_tasks_screen={task:relative[k]<plan['decision_weak_ratio'] for k,task in enumerate(taskids) if k>0},
                Actor_hash_unchanged=before==S.digest(q.state_dict())==plan['policy_hash'],
                limitations='Two reused fixed first batches only; first-order parameter sensitivity is not real rollout/Adam effect. No future success or universally good direction is proved.')
            assert record['Actor_hash_unchanged'];result.append(record);C.save(OUT/(arm+'.json'),record);status(arm+'_COMPLETE')
            del losses,lp,term,gg,pg,q,frozen,data,items,params,mode,jac,ret,rg
        assert not forbidden
        for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    except BaseException:
        reason='FAILED';failure=traceback.format_exc();C.save(OUT/'failure.json',dict(traceback=failure,physics_steps=0,optimizer_calls=0))
    signal.alarm(0)
    summary=dict(state=reason,physics_steps=0,optimizer_calls=0,navigation_Actor_updates=0,all_workers_stopped=True,
        completed_batches=len(result),elapsed_s=time.time()-start,prohibited_calls=len(forbidden),
        own_bytes=S.used(OUT),global_bytes=S.used(ROOT/'stage/runs'),raw_results=[dict(arm=r['arm'],task_order=r['task_order'],gradient_norms=r['gradient_norms'],gradient_norm_ratio_to_LEFT=r['gradient_norm_ratio_to_LEFT'],PCGrad_norm_to_mean=r['PCGrad_norm_to_mean'],weak_tasks_screen=r['weak_tasks_screen']) for r in result],
        no_policy_model_exported=True,automatic_next_training=False,remote_backup_not_verified=True)
    assert summary['own_bytes']<plan['final_bytes'] and summary['global_bytes']<plan['global_bytes'];C.save(OUT/'summary.json',summary);C.save(OUT/'status.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
