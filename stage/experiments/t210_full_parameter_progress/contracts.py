"""Offline mathematical and full-backbone gradient contracts, zero optimizer/physics."""
from support import *
def main():
    import fcntl
    lock=(OUT/'contracts.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'contracts.json').exists()
    plan=json.loads((HERE/'protocol.json').read_text())
    for source,h in plan['input_hashes'].items(): assert C.sha(source)==h,source
    torch.set_num_threads(1); torch.manual_seed(plan['seed'])
    parent=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['policy']
    p=policy(parent)
    with np.load(plan['retention_targets']) as z:
        x=torch.tensor(z['states']); targets=torch.tensor(z['actions'])
    initial_error=float((p(x)-targets).abs().max())
    assert initial_error==0
    action=p(x[:64]).detach()*.8
    loss=-p.physical_logp(x[:64],action).mean(); loss.backward()
    grad=float(p.body.net[0].weight.grad.norm())
    assert grad>0 and torch.isfinite(p.body.net[0].weight.grad).all()
    counts=np.array([100,100,34]); rewards=np.array([.1,-.2,-1.])
    original=monte_carlo(rewards,counts)
    modified=monte_carlo(training_rewards(rewards,counts,1.5,.3),counts)
    expected=original[0]-R.GAMMA**(counts.sum()-1)*.3*1.5
    error=abs(modified[0]-expected); assert error<1e-12
    # A terminal rollout ends MC at exactly its last physical event; no fictitious bootstrap.
    assert monte_carlo([2.],[34])[0]==2.
    v=Value(); vv=v(x[:64]); assert bool((vv==0).all())
    assert not set(map(id,p.parameters())) & set(map(id,v.parameters()))
    trainable=sum(q.numel() for q in p.parameters() if q.requires_grad)
    assert p.body.net[0].weight.requires_grad and trainable>570000
    row={'state':'PASS','physics_steps':0,'optimizer_updates':0,'initial_parent_action_max_error':initial_error,
         'actor_trainable_parameters':trainable,'first_layer_density_gradient_norm':grad,
         'terminal_distance_discount_identity_error':error,'new_value_initial_zero':True,
         'independent_value_parameters':True,'protocol_sha256':C.sha(HERE/'protocol.json')}
    C.save(OUT/'contracts.json',row); print(json.dumps(row))
if __name__=='__main__':main()
