"""Candidate update-acceptance machinery; no rollout controller or model promotion.

Checks cover only supplied finite states, never imply global safety/retention.
Callers must independently validate improvement, behavior identity and rollouts.
"""
import copy
import math
import torch


def truncated_gaussian_kl(old_mean, new_mean, sigma=(.35, .30)):
    """Exact old||new KL for independent fixed-sigma normals on [-1,1]."""
    old_mean, new_mean = old_mean.double(), new_mean.double()
    sigma = old_mean.new_tensor(sigma)
    if not bool((sigma > 0).all()):
        raise ValueError('Positive standard deviations required')
    alpha, beta = (-1-old_mean)/sigma, (1-old_mean)/sigma
    old_z = torch.special.ndtr(beta)-torch.special.ndtr(alpha)
    new_z = torch.special.ndtr((1-new_mean)/sigma)-torch.special.ndtr((-1-new_mean)/sigma)
    if not bool(((old_z > .001) & (new_z > .001)).all()):
        raise ValueError('Truncated density normalization outside validated domain')
    phi = lambda x: torch.exp(-x.square()/2)/math.sqrt(2*math.pi)
    offset = sigma*(phi(alpha)-phi(beta))/old_z
    shift = old_mean-new_mean
    coordinate_kl = torch.log(new_z/old_z)+(shift.square()+2*shift*offset)/(2*sigma.square())
    if not bool(torch.isfinite(coordinate_kl).all()) or float(coordinate_kl.min()) < -1e-10:
        raise ValueError('Invalid analytic KL')
    return coordinate_kl.sum(-1)


def select_proposal(model, old_state, proposed_state, evaluate, factors=(1., .5, .25, .125, .0625)):
    """Accept first explicitly passing candidate; otherwise restore all parameters.

    `evaluate(model)` returns a JSON-compatible dict including bool `accept`.
    It should check a whole fresh rollout and all retention anchors, surrogate
    improvement and finite values; any empirical bound is local to those data.
    This function itself does no optimizer or simulator step and saves no model.
    """
    if set(old_state) != set(proposed_state) or any(not 0 < f <= 1 for f in factors):
        raise ValueError('Mismatched state dictionaries or invalid step factors')
    attempts = []
    try:
        with torch.no_grad():
            for factor in factors:
                candidate = {k: old_state[k]+factor*(proposed_state[k]-old_state[k]) for k in old_state}
                if any(not bool(torch.isfinite(v).all()) for v in candidate.values()):
                    raise ValueError('Non-finite candidate parameters')
                model.load_state_dict(candidate)
                result = evaluate(model)
                if not isinstance(result.get('accept'), bool):
                    raise ValueError('Evaluator must explicitly return accept: bool')
                attempts.append({'factor': factor, **result})
                if result['accept']:
                    return {'accepted': True, 'factor': factor, 'attempts': attempts}
        model.load_state_dict(old_state)
        return {'accepted': False, 'factor': 0., 'attempts': attempts}
    except BaseException:
        model.load_state_dict(old_state)
        raise


def guarded_adam_step(model, optimizer, evaluate, factors=(1., .5, .25, .125, .0625)):
    """Caller supplies already-checked gradients; rejected steps restore Adam too.

    Accepted parameter interpolation scales that Adam proposal's effective step.
    Adam moments correspond to the gradients at the pre-step model; next nominal
    LR is unchanged. No claim to implement natural-gradient TRPO.
    """
    if not isinstance(optimizer, torch.optim.Adam):
        raise TypeError('Only Adam has been covered by this rollback contract')
    old = {k: v.detach().clone() for k, v in model.state_dict().items()}
    old_optimizer = copy.deepcopy(optimizer.state_dict())
    try:
        optimizer.step()
        proposed = {k: v.detach().clone() for k, v in model.state_dict().items()}
        result = select_proposal(model, old, proposed, evaluate, factors)
        if not result['accepted']:
            optimizer.load_state_dict(old_optimizer)
        return result
    except BaseException:
        model.load_state_dict(old)
        optimizer.load_state_dict(old_optimizer)
        raise
