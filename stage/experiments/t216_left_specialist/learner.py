"""Full-batch, finite-data update acceptance; independent fresh MC value baseline."""
from support import *
sys.path.insert(0,str(ROOT/'stage/experiments'))
from native_update_guard import truncated_gaussian_kl, guarded_adam_step


def verify_behavior(p,x,actions,recorded_means,oldlogp,plan):
    """Separate exact scalar behavior identity from bounded float32 batch error."""
    with torch.no_grad():
        scalar_mean=torch.cat([p.physical_mean(s[None]) for s in x])
        scalar_error=float((scalar_mean-recorded_means).abs().max())
        recorded_lp=R.log_density(recorded_means,actions)
        density_error=float((recorded_lp-oldlogp).abs().max())
        batch_mean=p.physical_mean(x)
        batch_mean_error=float((batch_mean-recorded_means).abs().max())
        batch_lp_error=float((R.log_density(batch_mean,actions)-oldlogp).abs().max())
        assert scalar_error==0 and density_error==0, 'Behavior/data identity mismatch'
        assert batch_mean_error<=plan['batch_mean_tolerance'] and batch_lp_error<=plan['batch_logp_tolerance'], 'Batch arithmetic outside registered error bounds'
    return {'scalar_native_mean_max_error':scalar_error,'recorded_density_max_error':density_error,
            'batch_native_mean_max_error':batch_mean_error,'batch_logp_max_error':batch_lp_error}


def actor_update(p,optimizer,x,actions,oldlogp,advantage,anchor_x,anchor_y,offsets,parent_anchor_mean,plan,check):
    with torch.no_grad():
        behavior_mean=p.physical_mean(x).clone()
    def measure():
        ratio=(p.physical_logp(x,actions)-oldlogp).exp()
        surrogate=torch.minimum(ratio*advantage,ratio.clamp(1-plan['clip_ratio'],1+plan['clip_ratio'])*advantage).mean()
        errors=(p(anchor_x)-anchor_y).square().mean(-1)
        mse=torch.stack([errors[offsets[i]:offsets[i+1]].mean() for i in range(7)]).mean()
        return surrogate-plan['retention_weight']*mse,surrogate,mse
    attempts=[]; accepted=0; first_grad=None
    for step in range(plan['actor_attempts_per_round']):
        check()
        objective,_,_=measure(); previous=float(objective.detach())
        optimizer.zero_grad();(-objective).backward()
        grad=float(p.body.net[0].weight.grad.norm())
        if first_grad is None:first_grad=grad
        grad_norm=float(torch.nn.utils.clip_grad_norm_([q for q in p.parameters() if q.requires_grad],plan['grad_clip'],error_if_nonfinite=True))
        def evaluate(q):
            objective,surrogate,mse=measure()
            train_kl=truncated_gaussian_kl(behavior_mean,q.physical_mean(x))
            retention_kl=truncated_gaussian_kl(parent_anchor_mean,q.physical_mean(anchor_x))
            obj=float(objective);mean_kl=float(train_kl.mean());max_old_kl=float(retention_kl.max());err=float(mse)
            ok=(math.isfinite(obj) and obj>=previous+plan['minimum_objective_improvement']
                and mean_kl<=plan['max_behavior_mean_KL'] and max_old_kl<=plan['max_parent_anchor_KL']
                and err<=plan['max_balanced_anchor_MSE'])
            return {'accept':ok,'regularized_surrogate':obj,'surrogate':float(surrogate),
                    'objective_improvement':obj-previous,'training_mean_KL':mean_kl,
                    'parent_anchor_max_KL':max_old_kl,'balanced_anchor_MSE':err}
        report=guarded_adam_step(p,optimizer,evaluate,plan['step_factors'])
        report.update(step=step,first_layer_gradient_norm=grad,gradient_norm_before_clip=grad_norm)
        attempts.append(report);accepted+=int(report['accepted'])
        if not report['accepted']:break
    return {'attempts':attempts,'accepted_updates':accepted,'first_layer_gradient_norm':first_grad}
