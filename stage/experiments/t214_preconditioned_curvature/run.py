"""Matrix-free Gauss-Newton audit and fixed-grid offline policy proposals."""
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
from pathlib import Path
import sys,time,json,copy,fcntl,traceback,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t214_preconditioned_curvature'
sys.path.insert(0,str(ROOT/'stage/experiments/t203_native_exploration'))
import runtime as R
sys.path.insert(0,str(ROOT/'stage/experiments'))
from native_update_guard import truncated_gaussian_kl
torch,np,C=R.torch,R.np,R.C
class BudgetStop(Exception):pass
def used(p):return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
def model(state,dtype=torch.float32):
    p=R.Policy();p.load_state_dict(state);p.logstd.requires_grad_(False);return p.to(dtype=dtype).eval()
def mc(rewards,counts):
    result=np.empty(len(rewards));g=0.
    for i in range(len(rewards)-1,-1,-1):g=float(rewards[i])+R.GAMMA**int(counts[i])*g;result[i]=g
    return result
def flatgrad(loss,params,create=False):
    gs=torch.autograd.grad(loss,params,create_graph=create,retain_graph=True,allow_unused=True)
    return torch.cat([(torch.zeros_like(q) if g is None else g).reshape(-1) for q,g in zip(params,gs)])
def cg(A,b,iterations,tolerance,diagonal=None):
    x=torch.zeros_like(b);r=b.clone();z=r.clone() if diagonal is None else r/diagonal
    d=z.clone();rho=torch.dot(r,z);norm=float(b.norm());history=[]
    for iteration in range(iterations):
        ad=A(d);den=torch.dot(d,ad);assert float(den)>0,'Nonpositive CG curvature'
        alpha=rho/den;x=x+alpha*d;r=r-alpha*ad
        relative=float(r.norm())/norm;history.append(relative)
        if relative<=tolerance:break
        z=r.clone() if diagonal is None else r/diagonal
        next_rho=torch.dot(r,z);d=z+(next_rho/rho)*d;rho=next_rho
    residual=float((A(x)-b).norm())/norm
    return x,{'iterations':len(history),'recursive_residuals':history,'actual_relative_residual':residual,'converged':residual<=tolerance}

def main():
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'status.json').exists(),'No duplicate T214'
    plan=json.loads((HERE/'protocol.json').read_text())
    for path,h in plan['input_hashes'].items():assert C.sha(path)==h,path
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(plan['seed'])
    start=time.monotonic();deadline=start+plan['wall_seconds'];trials=[];solves=[];contracts={};export=None;reason='COMPLETE'
    def check():
        if time.monotonic()>deadline-5:raise BudgetStop('WALL_RESERVE')
        if used(OUT)+3*1024**2>=plan['final_bytes'] or used(ROOT/'stage/runs')+3*1024**2>=1024**3 or shutil.disk_usage(ROOT).free<40*1024**3:raise BudgetStop('STORAGE_RESERVE')
    C.save(OUT/'status.json',{'state':'RUNNING','pid':os.getpid(),'physics_steps':0,'optimizer_calls':0})
    try:
        state=torch.load(plan['source_policy'],map_location='cpu',weights_only=False)['policy']
        parent_state=torch.load(plan['parent_policy'],map_location='cpu',weights_only=False)['policy']
        p=model(state,torch.float64);evalp=model(state).requires_grad_(False);parent=model(parent_state).requires_grad_(False)
        names=[n for n,q in p.named_parameters() if q.requires_grad];params=[q for q in p.parameters() if q.requires_grad]
        with np.load(plan['retention_targets']) as z:anchor=torch.tensor(z['states']);target=torch.tensor(z['actions']);offsets=z['offsets'].copy()
        data=[]
        for row in plan['rows']:
            with np.load(row['trace']) as z:data.append((z['states'].copy(),z['actions'].copy(),z['logp'].copy(),mc(z['rewards'],z['native_counts'])))
        x,actions,oldlp,G=[torch.tensor(np.concatenate([d[i] for d in data]),dtype=torch.float32 if i<2 else torch.float64) for i in range(4)]
        adv=(G-G.mean())/G.std(unbiased=False).clamp_min(1e-8)
        with torch.no_grad():
            reference=parent.physical_mean(anchor);behavior=evalp.physical_mean(x)
            current_anchor=evalp.physical_mean(anchor);current_KL=truncated_gaussian_kl(reference,current_anchor)
            current_MSE=(evalp(anchor)-target).square().mean(-1)
        selected=set()
        for i in range(7):selected.update(np.linspace(offsets[i],offsets[i+1]-1,plan['old_uniform_per_task'],dtype=int).tolist())
        selected.update(torch.topk(current_KL,plan['old_worst_KL']).indices.tolist());selected.update(torch.topk(current_MSE,plan['old_worst_MSE']).indices.tolist())
        old_metric=anchor[sorted(selected)].double();new_metric=x[np.linspace(0,len(x)-1,plan['new_metric_states'],dtype=int)].double()
        sigma=torch.tensor(R.SIGMA,dtype=torch.float64)
        old_reference=p.physical_mean(old_metric).detach();new_reference=p.physical_mean(new_metric).detach()
        def objective(q,xx,aa):
            ratio=(q.physical_logp(xx,aa)-oldlp).exp()
            surrogate=torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()
            aa_anchor=anchor.to(dtype=xx.dtype);yy=target.to(dtype=xx.dtype)
            errs=(q(aa_anchor)-yy).square().mean(-1)
            mse=torch.stack([errs[offsets[i]:offsets[i+1]].mean() for i in range(7)]).mean()
            return surrogate-mse,mse
        loss,_=objective(p,x.double(),actions.double());g=flatgrad(loss,params).detach();assert float(g.norm())>0
        def metric(q,beta):
            old=(q.physical_mean(old_metric)-old_reference)/sigma
            new=(q.physical_mean(new_metric)-new_reference)/sigma
            return .5*old.square().sum(-1).mean()+.5*beta*new.square().sum(-1).mean()
        def operator(beta,damping):
            grad=flatgrad(metric(p,beta),params,True)
            assert float(grad.norm())==0
            def A(v):
                check();return flatgrad(torch.dot(grad,v),params).detach()+damping*v
            return A
        A=operator(1.,.1)
        u=torch.randn_like(g);u/=u.norm();v=torch.randn_like(g);v/=v.norm();au,av=A(u),A(v)
        symmetry=abs(float(torch.dot(u,av)-torch.dot(v,au)))/max(abs(float(torch.dot(u,av))),abs(float(torch.dot(v,au))),1e-12)
        positivity=float(torch.dot(v,av));assert symmetry<1e-6 and positivity>0
        epsilon=plan['finite_difference_epsilon'];finite_model=model(state,torch.float64);finite_params=[q for q in finite_model.parameters() if q.requires_grad]
        finite=[]
        for sign in (1.,-1.):
            finite_model.load_state_dict(state);offset=0
            with torch.no_grad():
                for q in finite_params:q.add_(sign*epsilon*v[offset:offset+q.numel()].reshape_as(q));offset+=q.numel()
            finite.append(flatgrad(metric(finite_model,1.),finite_params).detach())
        fd=(finite[0]-finite[1])/(2*epsilon)+.1*v
        finite_error=float((fd-av).norm()/av.norm());assert finite_error<.005
        contracts={'symmetry_relative_error':symmetry,'positive_quadratic':positivity,'finite_difference_relative_error':finite_error,
            'metric_old_states':len(old_metric),'metric_new_states':len(new_metric),'parameters':len(g),'gradient_norm':float(g.norm()),
            'all_checks_pass':True}
        with torch.no_grad():initial_objective,initial_mse=objective(evalp,x,actions)
        initial=float(initial_objective);best=None;directions=[('ordinary_gradient',g/g.norm(),True)]
        def exact_diagonal(states):
            result=torch.zeros_like(g)
            for xx in states:
                check();mean=p.physical_mean(xx[None])/sigma
                for j in range(2):
                    jac=flatgrad(mean[0,j],params).detach();result+=jac.square()/len(states)
            return result
        diag_old=exact_diagonal(old_metric);diag_new=exact_diagonal(new_metric)
        contracts['diagonal_old_minmax']=[float(diag_old.min()),float(diag_old.max())]
        contracts['diagonal_new_minmax']=[float(diag_new.min()),float(diag_new.max())]
        for config in plan['configs']:
            check();precondition=diag_old+config['beta']*diag_new+config['damping'] if config['preconditioned'] else None
            direction,info=cg(operator(config['beta'],config['damping']),g,plan['cg_max_iterations'],plan['cg_relative_tolerance'],precondition)
            label=f"{'PCG' if config['preconditioned'] else 'CG'}_beta{config['beta']}_damping{config['damping']}"
            solves.append({'id':label,**config,**info})
            if info['converged']:directions.append((label,direction/direction.norm(),True))
        for label,direction,_ in directions:
            for size in plan['step_L2']:
                check();candidate={k:q.clone() for k,q in state.items()};offset=0
                for name,q in zip(names,params):
                    candidate[name]=(state[name].double()+size*direction[offset:offset+q.numel()].reshape_as(q)).float();offset+=q.numel()
                evalp.load_state_dict(candidate)
                with torch.no_grad():
                    obj,mse=objective(evalp,x,actions);bkl=truncated_gaussian_kl(behavior,evalp.physical_mean(x)).mean()
                    akl=truncated_gaussian_kl(reference,evalp.physical_mean(anchor)).max();improvement=float(obj)-initial
                    passed=(improvement>=plan['minimum_objective_improvement'] and float(mse)<=plan['max_balanced_anchor_MSE'] and float(akl)<=plan['max_parent_anchor_KL'] and float(bkl)<=plan['max_behavior_mean_KL'])
                row={'id':label,'step_L2':size,'objective_improvement':improvement,'balanced_anchor_MSE':float(mse),
                     'behavior_mean_KL':float(bkl),'parent_anchor_max_KL':float(akl),'all_nonlinear_checks_pass':passed}
                trials.append(row)
                if passed and (best is None or improvement>best[0]['objective_improvement']):best=(row,candidate)
        if best:
            file=OUT/'offline_candidate.pt';torch.save({'policy':best[1],'selection':best[0],'protocol_sha256':C.sha(HERE/'protocol.json'),
                'origin':'Archived-data completeMC-minus-zero gradient / GN offline proposal; not onlineDRL or fullTRPO/CPO','automatic_promotion':False},file)
            assert file.stat().st_size<3*1024**2
            export={'path':str(file),'sha256':C.sha(file),'selection':best[0]}
    except BudgetStop as e:reason=str(e)
    except BaseException:reason='FAILED';C.save(OUT/'failure.json',{'traceback':traceback.format_exc()})
    for source,h in plan['input_hashes'].items():assert C.sha(source)==h,source
    result={'state':'FAILED' if reason=='FAILED' else 'COMPLETE','stop_reason':reason,'physics_steps':0,'optimizer_calls':0,
        'online_training_updates':0,'offline_parameter_candidates':len(trials),'contracts':contracts,'CG_solves':solves,'trials':trials,
        'export':export,'automatic_promotion':False,'elapsed_s':time.monotonic()-start,'own_bytes':used(OUT),
        'scope':plan['contract']}
    assert used(OUT)<plan['final_bytes'] and used(ROOT/'stage/runs')<1024**3
    C.save(OUT/'summary.json',result);C.save(OUT/'status.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
