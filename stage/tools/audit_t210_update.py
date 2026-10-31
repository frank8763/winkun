"""Posthoc frozen numerical/trust audit. Zero physics and optimizer updates."""
from pathlib import Path
import sys,json,math
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'stage/experiments/t210_full_parameter_progress'))
from support import *

def exact_kl(old,new):
    """KL of independent equal-sigma Gaussians truncated to [-1,1]."""
    old,new=old.double(),new.double()
    sigma=old.new_tensor(R.SIGMA)
    al,be=(-1-old)/sigma,(1-old)/sigma
    z=torch.special.ndtr(be)-torch.special.ndtr(al)
    zn=torch.special.ndtr((1-new)/sigma)-torch.special.ndtr((-1-new)/sigma)
    phi=lambda q:torch.exp(-q.square()/2)/math.sqrt(2*math.pi)
    mean_offset=sigma*(phi(al)-phi(be))/z
    shift=old-new
    kl=torch.log(zn/z)+(shift.square()+2*shift*mean_offset)/(2*sigma.square())
    assert float(kl.min())>-1e-10
    return kl.sum(-1)

def main():
    torch.set_num_threads(1)
    plan=json.loads((OUT/'protocol.json').read_text())
    summary=json.loads((OUT/'summary.json').read_text()); assert summary['all_workers_stopped']
    result_path=OUT/'posthoc_update_audit.json'; assert not result_path.exists()
    parent=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['policy']
    p0=policy(parent).eval().requires_grad_(False)
    with np.load(plan['retention_targets']) as z: anchors=torch.tensor(z['states']);targets=torch.tensor(z['actions'])
    records=[json.loads(s) for s in (OUT/'rollouts.jsonl').read_text().splitlines()]
    density=[]; interpolation=[]; sources={str(HERE/'protocol.json'):C.sha(HERE/'protocol.json')}
    for arm in plan['arms']:
        for gen in (0,1):
            rows=[r for r in records if r['role']=='training' and r['round']==gen and r['arm']==arm]
            path=Path(rows[0]['weights']);sources[str(path)]=C.sha(path)
            p=policy(torch.load(path,map_location='cpu',weights_only=False)['policy']).eval().requires_grad_(False)
            all_states=[]; all_actions=[]; all_logp=[]; all_means=[]
            for row in rows:
                sources[row['trace']]=C.sha(row['trace'])
                with np.load(row['trace']) as z:
                    all_states.append(z['states'].copy());all_actions.append(z['actions'].copy())
                    all_logp.append(z['logp'].copy());all_means.append(z['means'].copy())
            x=torch.tensor(np.concatenate(all_states));a=torch.tensor(np.concatenate(all_actions));lp=torch.tensor(np.concatenate(all_logp));means=torch.tensor(np.concatenate(all_means))
            with torch.no_grad():
                batch_lp=p.physical_logp(x,a);batch_means=p.physical_mean(x)
                scalar_lp=torch.cat([p.physical_logp(s[None],ac[None]) for s,ac in zip(x,a)])
            single_err=float((scalar_lp-lp).abs().max());batch_err=float((batch_lp-lp).abs().max())
            assert single_err==0
            density.append({'arm':arm,'round':gen,'decisions':len(x),'single_forward_logp_max_error':single_err,
                'batch_forward_logp_max_error':batch_err,'batch_native_mean_max_error':float((batch_means-means).abs().max()),
                'max_probability_ratio_relative_error':float(((batch_lp-lp).exp()-1).abs().max())})
            if gen!=0:continue
            proposed_path=OUT/'models'/f'{arm}_round0.pt';sources[str(proposed_path)]=C.sha(proposed_path)
            proposed=torch.load(proposed_path,map_location='cpu',weights_only=False)['policy']
            original_means=p0.physical_mean(x)
            with torch.no_grad():
                for factor in (1.,.5,.25,.125,.0625,0.):
                    state={k:parent[k]+factor*(proposed[k]-parent[k]) for k in parent}
                    q=policy(state).eval().requires_grad_(False);new_means=q.physical_mean(x)
                    kl=exact_kl(original_means,new_means)
                    with_old7=exact_kl(p0.physical_mean(anchors),q.physical_mean(anchors))
                    approx=((q.physical_logp(x,a)-lp).exp()-1-(q.physical_logp(x,a)-lp)).mean()
                    interpolation.append({'arm':arm,'factor':factor,'exact_mean_KL_training':float(kl.mean()),
                        'exact_max_KL_training':float(kl.max()),'sampled_approx_KL_training':float(approx),
                        'exact_mean_KL_old7_anchors':float(with_old7.mean()),'exact_max_KL_old7_anchors':float(with_old7.max()),
                        'old7_action_MSE':float((q(anchors)-targets).square().mean())})
                # Independent numerical integration, not just a second use of the algebra.
                grid=torch.linspace(-1,1,4097,dtype=torch.float64)
                m0=original_means[:4].double();m1=policy(proposed).physical_mean(x[:4]).double()
                sigma=m0.new_tensor(R.SIGMA)
                def coordinate_density(m):
                    z=torch.special.ndtr((1-m)/sigma)-torch.special.ndtr((-1-m)/sigma)
                    return -.5*((grid[:,None,None]-m)/sigma).square()-torch.log(sigma)-.5*math.log(2*math.pi)-torch.log(z)
                l0,l1=coordinate_density(m0),coordinate_density(m1)
                numerical=torch.trapezoid(l0.exp()*(l0-l1),grid,dim=0).sum(-1)
                error=float((numerical-exact_kl(m0,m1)).abs().max());assert error<1e-5
                interpolation.append({'arm':arm,'quadrature_contract_max_error':error,'grid_points':len(grid)})
    vector={a:torch.cat([(torch.load(OUT/'models'/f'{a}_round0.pt',map_location='cpu',weights_only=False)['policy'][k]-parent[k]).flatten() for k in parent]) for a in plan['arms']}
    cosine=float(torch.nn.functional.cosine_similarity(vector['original'],vector['terminal_progress'],dim=0))
    result={'state':'POSTHOC_FROZEN_AUDIT_COMPLETE','physics_steps':0,'optimizer_updates':0,
        'training_run_state':summary['state'],'density_recomputation':density,'first_update_interpolation':interpolation,
        'first_update_parameter_delta_cosine_between_arms':cosine,'input_sha256':sources,
        'interpretation':'Exact single-state collection law reconstructed with zero error. Batch-vs-single float32 forward discrepancy tripped1e-5gate; no stale-policy mismatch observed on checked records. Sampling estimator and exact truncated-Gaussian KL both reported; no interpolated candidate executed or exported. Offline anchor bounds do not guarantee closed-loop retention or success. PPO early stop after first oversized Adam step is not a hard trust-region guarantee. No original evidence overwritten.'}
    C.save(result_path,result); print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','first_update_interpolation')}))
    for row in interpolation:print(json.dumps(row))
if __name__=='__main__':main()
