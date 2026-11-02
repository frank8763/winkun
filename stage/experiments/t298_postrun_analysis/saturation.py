"""Posthoc output-Jacobian diagnosis only; no optimizer or simulator.

This does not change T298 acceptance criteria or its registered training source.
Saturated regression errors have zero local derivative through hard clipping.
That fact alone is not a causal proof of navigation failure or a trained remedy.
"""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t298_self_goal_transfer'))
from t298_learning import policy,auxiliary,L,S,C,torch,np
sys.path.insert(0,str(ROOT/'stage/experiments/t296_log_tail_support'))
import stable_density as D

def main():
    out=ROOT/'stage/runs/t298_self_goal_transfer';dest=ROOT/'stage/docs/distance_20261102/T298_POSTHOC_SATURATION.json';assert not dest.exists()
    summary=json.loads((out/'summary.json').read_text());assert summary['all_workers_stopped'];p=json.loads((out/'protocol.json').read_text());torch.set_num_threads(1);items=auxiliary(p);reports=[]
    frozen=policy(p).eval().requires_grad_(False);final=policy(p).eval().requires_grad_(False);export=summary['arms']['self_goal_transfer']['export'];assert C.sha(export['path'])==export['sha256'];final.load_state_dict(L.read(export['path'])['policy'])
    for name,q in [('safeT255',frozen),('T298_final',final)]:
        rows=[]
        for source,(x,y) in zip(p['auxiliary_sources'],items):
            with torch.no_grad():mode,mean,_,raw=q.step(x)
            leaf=raw.detach().clone().requires_grad_(True);prediction=q.mode(leaf);loss=(prediction-y).square().sum();grad=torch.autograd.grad(loss,leaf)[0]
            raw_leaf=raw.detach().clone().requires_grad_(True);raw_loss=(raw_leaf-y).square().sum();raw_grad=torch.autograd.grad(raw_loss,raw_leaf)[0]
            likelihood_leaf=raw.detach().clone().requires_grad_(True);mean=L.R.center(likelihood_leaf)
            reset=torch.zeros(len(y),dtype=torch.bool);reset[0]=True
            logp=D.logp(mean,y,torch.cat([mean[:1],mean[:-1]]),torch.cat([y[:1],y[:-1]]),reset)
            nll=-logp.sum();likelihood_grad=torch.autograd.grad(nll,likelihood_leaf)[0];assert torch.isfinite(likelihood_grad).all()
            errors=(mode-y).square();zero=grad==0;clipped=raw.abs()>.999999
            assert bool((grad[clipped]==0).all());assert bool((raw_grad[clipped].abs()>0).all())
            rows.append(dict(id=source['id'],clipped_fraction_by_drive_steer=clipped.float().mean(0).tolist(),MSE_by_drive_steer=errors.mean(0).tolist(),fraction_regression_squared_error_at_zero_output_gradient=float(errors[zero].sum()/errors.sum()),mode_output_gradient_norm=float(grad.norm()),raw_output_gradient_norm=float(raw_grad.norm()),conditional_AR_NLL_output_gradient_norm=float(likelihood_grad.norm()),mode_clipped_coordinates_with_nonzero_conditional_AR_NLL_gradient=int(((likelihood_grad!=0)&clipped).sum()),total_mode_clipped_coordinates=int(clipped.sum()),conditional_AR_NLL_finite=True,raw_drive_min=float(raw[:,0].min()),raw_drive_max=float(raw[:,0].max()),scope='Derivative with respect to network raw outputs, not full parameter gradient or causal navigation attribution. ConditionalAR loglikelihood includes previous-mean derivative, not HPG or an IS estimator. It demonstrates available gradients, not a trained remedy.'))
        reports.append(dict(policy=name,policy_hash=S.digest(q.state_dict()),rows=rows))
    # Interior saturated point: exact dead local derivative and independent FD.
    z=torch.tensor([1.2],dtype=torch.float64,requires_grad=True);target=.4;value=(frozen.mode(z)-target).square().sum();g=float(torch.autograd.grad(value,z)[0]);eps=1e-5;fd=float(((frozen.mode(z.detach()+eps)-target).square()-(frozen.mode(z.detach()-eps)-target).square())/(2*eps));assert g==fd==0
    raw_fd=float(((z.detach()+eps-target).square()-(z.detach()-eps-target).square())/(2*eps));assert abs(raw_fd-1.6)<1e-9
    dest.write_text(json.dumps(dict(scope='Posthoc T298 diagnostic, no preregistered screen changed, no extra native or optimizer, no new model training; clipping is one supervision optimization defect, not sufficient explanation of all6m failures.',native_steps=0,optimizer_calls=0,Actor_updates=0,input_protocol_sha256=C.sha(out/'protocol.json'),input_final_checkpoint_sha256=export['sha256'],source_sha256=C.sha(Path(__file__)),reports=reports,synthetic_mode_autograd=g,synthetic_mode_finite_difference=fd,synthetic_raw_finite_difference=raw_fd),indent=2)+'\n');print('T298 posthoc output-gradient saturation verified,0native0optimizer')

if __name__=='__main__':main()
