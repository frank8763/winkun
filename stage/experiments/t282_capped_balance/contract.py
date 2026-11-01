from balanced_learning import *
import json,os,signal,time,traceback
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t282_capped_balance'


def gram_project(G,factors,seed):
    # Coefficient-space representation of every projected full parameter vector.
    coeff=np.diag(np.array(factors,dtype=np.float64));original=coeff.copy();rng=np.random.default_rng(seed);records=[]
    for i in range(len(coeff)):
        for j in rng.permutation(len(coeff)):
            if i==j:continue
            dot=float(coeff[i]@G@original[j]);den=max(float(original[j]@G@original[j]),1e-30)
            if dot<0:
                alpha=dot/den;coeff[i]-=alpha*original[j];records.append(dict(task=i,against=int(j),coefficient=alpha))
    return coeff,coeff.mean(axis=0),records


def main():
    assert not (OUT/'status.json').exists();p=json.loads((HERE/'protocol.json').read_text());start=time.time();reason='COMPLETE';result={};forbidden=[]
    def reject(*args,**kw):forbidden.append(True);raise RuntimeError('No optimizer or native in T282')
    torch.optim.Adam.step=reject;S.native_setup=reject
    def timeout(signum,frame):raise TimeoutError('T282300s deadline')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(p['wall_seconds'])
    C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[]));C.save(OUT/'status.json',dict(state='STARTED',physics_steps=0,optimizer_calls=0))
    try:
        for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
        torch.set_num_threads(1)
        vectors=[torch.tensor(v,dtype=torch.float64) for v in [[3.,1.,0.,0.],[-.02,.03,0.,0.],[0.,0.,.001,0.],[0.,0.,0.,.002]]]
        balance=fixed_balance(vectors,p['seed'],p['balance_cap'],p['relative_noise_floor'])
        original,_=L.project(vectors,p['seed']);weighted,_=L.project([g*f for g,f in zip(vectors,balance['factors'])],p['seed'])
        assert abs(norm(original)-norm(weighted*balance['common_scale']))<1e-12
        G=torch.stack(vectors)@torch.stack(vectors).T;coeff,direction,records=gram_project(G.numpy(),balance['factors'],p['seed'])
        full=torch.tensor(direction)@torch.stack(vectors);assert torch.max((full-weighted).abs())<1e-12
        zeros=[torch.zeros_like(vectors[0]) for _ in range(4)];zero=fixed_balance(zeros,p['seed']);assert zero['factors']==[1.]*4 and zero['common_scale']==1.
        tiny=fixed_balance([vectors[0],vectors[0]*1e-16,zeros[0],vectors[0]*.01],p['seed'])
        assert tiny['factors'][1:3]==[1.,1.] and tiny['factors'][3]<=32.
        # Same round factors multiply both loss and its gradient. They are detached
        # constants, never recomputed inside candidate objective evaluation.
        x=torch.tensor([.2,-.3],dtype=torch.float64,requires_grad=True);losses=[(k+1)*(x.square().sum()+x[0]) for k in range(4)]
        actual=torch.autograd.grad(sum(f*l for f,l in zip(balance['factors'],losses))*balance['common_scale']/4,x)[0]
        expected=sum(f*(k+1)*(2*x.detach()+torch.tensor([1.,0.],dtype=torch.float64)) for k,f in enumerate(balance['factors']))*balance['common_scale']/4
        assert torch.max((actual-expected).abs())<1e-12
        record=json.loads((ROOT/'stage/runs/t281_task_gradient_audit/bounded_endpoint_cost.json').read_text());G=np.array(record['Gram_float64']);norms=np.sqrt(np.diag(G));ref=float(norms.max());floor=max(ref*p['relative_noise_floor'],1e-12)
        factors=[min(p['balance_cap'],ref/n) if n>floor else 1. for n in norms]
        old_coeff,old_dir,_=gram_project(G,[1.]*4,p['seed']);new_coeff,new_dir,new_records=gram_project(G,factors,p['seed'])
        old_norm=math.sqrt(float(old_dir@G@old_dir));new_norm=math.sqrt(float(new_dir@G@new_dir));scale=old_norm/new_norm;d=new_dir*scale
        assert abs(old_norm-record['PCGrad_direction_norm'])/old_norm<5e-6
        assert all(1<=f<=32 for f in factors) and abs(math.sqrt(float(d@G@d))-old_norm)<1e-10
        progress=(G@d).tolist();assert np.isfinite(progress).all()
        result=dict(synthetic_PCGrad_fullvector_and_Gram_direction_maxerror=float(torch.max((full-weighted).abs())),
            zero_and_roundoff_signal_not_amplified=True,synthetic_weighted_objective_gradient_exact=True,
            real_T281_task_order=record['task_order'],real_factors=factors,real_common_scale=scale,
            real_original_PCGrad_norm_recorded=record['PCGrad_direction_norm'],real_original_PCGrad_norm_Gram_reconstructed=old_norm,
            real_balanced_PCGrad_norm_preserved=math.sqrt(float(d@G@d)),real_direction_cosine=float(old_dir@G@d)/(old_norm*old_norm),
            real_first_order_task_loss_improvement_old=(G@old_dir).tolist(),real_first_order_task_loss_improvement_balanced=progress,
            real_projected_task_norms_before_common_scale=np.sqrt(np.diag(new_coeff@G@new_coeff.T)).tolist(),projection_records=new_records,
            full_parameter_actual_candidate_not_evaluated=True,native_old20_acceptance_not_evaluated=True,
            limitation='Gram uses measured FP32gradients contracted inFloat64; current existingPCGrad usesFloat32reductions, so original agreement is tolerance not bit-exact. Newtask direction first-order only, not Adam/native success.')
        assert not forbidden
        for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
        C.save(OUT/'contract.json',result)
    except BaseException:reason='FAILED';C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physics_steps=0,optimizer_calls=0))
    signal.alarm(0)
    summary=dict(state=reason,physics_steps=0,optimizer_calls=0,navigation_Actor_updates=0,all_workers_stopped=True,
        elapsed_s=time.time()-start,prohibited_calls=len(forbidden),no_policy_exported=True,automatic_next_training=False,
        native_acceptance_or_navigation_success_proved=False,result=result,own_bytes=S.used(OUT),global_bytes=S.used(ROOT/'stage/runs'),remote_backup_verified=False)
    assert summary['own_bytes']<p['final_bytes'] and summary['global_bytes']<p['global_bytes'];C.save(OUT/'summary.json',summary);C.save(OUT/'status.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
