"""Bounded posthoc gradient-direction probe: zero physics/optimizer, no model export."""
from pathlib import Path
import sys,json,time,hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'stage/experiments/t211_guarded_progress'))
sys.path.insert(0,str(ROOT/'stage/experiments'))
from support import *
from native_update_guard import truncated_gaussian_kl
from functional_constraint_direction import project_two_constraints

def main():
    start=time.monotonic();torch.set_num_threads(1)
    target=OUT/'posthoc_direction.json';assert not target.exists()
    summary=json.loads((OUT/'summary.json').read_text());assert summary['all_workers_stopped']
    plan=json.loads((OUT/'protocol.json').read_text())
    rows=[json.loads(s) for s in (OUT/'rollouts.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['arm']=='original' and r['round']==1 and r['role']=='training']
    assert len(rows)==4
    state=torch.load(rows[0]['weights'],map_location='cpu',weights_only=False)['policy'];p=policy(state)
    params=[q for q in p.parameters() if q.requires_grad]
    parent=policy(torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['policy']).requires_grad_(False)
    with np.load(plan['retention_targets']) as z:anchor_x=torch.tensor(z['states']);anchor_y=torch.tensor(z['actions']);offsets=z['offsets']
    xs=[];actions=[];lps=[];returns=[]
    for row in rows:
        with np.load(row['trace']) as z:
            xs.append(z['states'].copy());actions.append(z['actions'].copy());lps.append(z['logp'].copy())
            returns.append(monte_carlo(z['rewards'],z['native_counts']))
    x=torch.tensor(np.concatenate(xs));actions=torch.tensor(np.concatenate(actions));oldlp=torch.tensor(np.concatenate(lps))
    G=torch.tensor(np.concatenate(returns));adv=(G-G.mean())/G.std(unbiased=False).clamp_min(1e-8)
    with torch.no_grad():reference=parent.physical_mean(anchor_x);behavior=p.physical_mean(x).clone()
    def measures():
        ratio=(p.physical_logp(x,actions)-oldlp).exp()
        surrogate=torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()
        kl=truncated_gaussian_kl(reference,p.physical_mean(anchor_x)).max()
        error=(p(anchor_x)-anchor_y).square().mean(-1)
        mse=torch.stack([error[offsets[i]:offsets[i+1]].mean() for i in range(7)]).mean()
        return surrogate-mse,kl,mse
    def flat_grad(value):
        gradients=torch.autograd.grad(value,params,allow_unused=True,retain_graph=True)
        return torch.cat([(torch.zeros_like(q) if g is None else g).detach().reshape(-1) for q,g in zip(params,gradients)])
    objective,kl,mse=measures();initial={'objective':float(objective.detach()),'anchor_max_KL':float(kl.detach()),'balanced_MSE':float(mse.detach())}
    gradient=flat_grad(objective);normal_kl=flat_grad(kl);normal_mse=flat_grad(mse)
    del objective,kl,mse
    unit=gradient/gradient.norm();normals=torch.stack([normal_kl,normal_mse])
    margins=torch.tensor([.01-initial['anchor_max_KL'],.00025-initial['balanced_MSE']],dtype=torch.float64)
    assert bool((margins>=0).all())
    trials=[]
    for size in (.01,.005,.0025,.00125):
        if time.monotonic()-start>60:raise TimeoutError('Posthoc direction 60s bound')
        proposed=size*unit;projected,certificate=project_two_constraints(proposed,normals,margins)
        for kind,delta in (('plain_gradient',proposed),('linearized_projected',projected)):
            p.load_state_dict(state)
            with torch.no_grad():
                offset=0
                for q in params:
                    q.add_(delta[offset:offset+q.numel()].reshape_as(q));offset+=q.numel()
                objective,kl,mse=measures();behavior_kl=truncated_gaussian_kl(behavior,p.physical_mean(x)).mean()
                improvement=float(objective)-initial['objective']
                passes=improvement>=1e-8 and float(kl)<=.01 and float(mse)<=.00025 and float(behavior_kl)<=.01
                trials.append({'kind':kind,'nominal_step_L2':size,'actual_step_L2':float(delta.norm()),
                    'objective_improvement':improvement,'anchor_max_KL':float(kl),'balanced_MSE':float(mse),
                    'behavior_mean_KL':float(behavior_kl),'all_nonlinear_checks_pass':passes,
                    'projection_certificate':certificate if kind=='linearized_projected' else None})
    p.load_state_dict(state)
    assert all(torch.equal(v,p.state_dict()[k]) for k,v in state.items())
    report={'state':'POSTHOC_DIRECTION_AUDIT_COMPLETE','physics_steps':0,'optimizer_calls':0,'models_exported':False,
        'max_wall_seconds':60,'elapsed_s':time.monotonic()-start,'initial':initial,'gradient_norm':float(gradient.norm()),'trials':trials,
        'scope':'Archived original-arm round1 behavior, completeMC-minus-zero baseline (not reconstruction of stored previous V or Adam moments). Fresh analytical gradient only. Two-constraint Euclidean local projection, not natural gradient/fullCPO/A-GEM replication. Strict nonlinear recheck. No navigation/retention/generalization outcome. No training initialized from candidates.',
        'sources':['https://arxiv.org/html/1705.10528','https://arxiv.org/html/1812.00420'],
        'source_sha256':{str(f):C.sha(f) for f in [Path(__file__),ROOT/'stage/experiments/functional_constraint_direction.py']}}
    assert used(OUT)+100*1024<plan['final_bytes'] and used(ROOT/'stage/runs')+100*1024<1024**3
    C.save(target,report);print(json.dumps(report))
if __name__=='__main__':main()
