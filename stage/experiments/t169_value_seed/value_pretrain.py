"""Fit successful physical-time returns before learning action dependence."""
import copy
import time
import numpy as np
import torch
import torch.nn.functional as F

def prepare(agent,path,seed):
    with np.load(path,allow_pickle=False) as z:
        names=['states','actions','rewards','next_states','dones','discounts','native_counts']
        arrays=[z[n].astype(np.float32).copy() for n in names]
        values=torch.from_numpy(z['mc_returns'].astype(np.float32))[:,None]
        offsets=z['episode_offsets'].tolist()
    states,actions=torch.from_numpy(arrays[0]),torch.from_numpy(arrays[1])
    actor={k:v.clone() for k,v in agent.actor.state_dict().items()}
    for net in [agent.critic.q1,agent.critic.q2]:
        with torch.no_grad():net[0].weight[:,-2:]=0
    rng=np.random.default_rng(seed+2);started=time.monotonic()
    with torch.no_grad():
        q1,q2=agent.critic(states,actions);before=float(((q1-values).square()+(q2-values).square()).mean())
    for update in range(1000):
        ids=np.concatenate([rng.integers(offsets[i],offsets[i+1],32) for i in range(len(offsets)-1)])
        q1,q2=agent.critic(states[ids],actions[ids]);loss=F.mse_loss(q1,values[ids])+F.mse_loss(q2,values[ids])
        assert torch.isfinite(loss)
        agent.critic_optimizer.zero_grad(set_to_none=True);loss.backward()
        for net in [agent.critic.q1,agent.critic.q2]:net[0].weight.grad[:,-2:]=0
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(),10);agent.critic_optimizer.step()
        if update%100==0:assert time.monotonic()-started<240,'VALUE_PREFIT_WALL_LIMIT'
    with torch.no_grad():
        q1,q2=agent.critic(states,actions);after=float(((q1-values).square()+(q2-values).square()).mean())
        alt1,alt2=agent.critic(states[:64],-actions[:64])
        action_invariance=float(torch.max((q1[:64]-alt1).abs()))
    assert action_invariance==0
    assert all(torch.equal(actor[k],v) for k,v in agent.actor.state_dict().items())
    agent.critic_target.load_state_dict(agent.critic.state_dict())
    demos=[tuple(a[i].copy() if isinstance(a[i],np.ndarray) else float(a[i]) for a in arrays) for i in range(len(arrays[0]))]
    report={'state':'PASS','critic_prefit_updates':1000,'demonstration_transitions':len(demos),
        'training_fit_mse_before':before,'training_fit_mse_after':after,'actor_unchanged':True,
        'initial_action_dependence_max_error':action_invariance,'elapsed_s':time.monotonic()-started,
        'scope':'Training-fit calibration only, not independent value accuracy or a safety guarantee; action columns now released for online TD3.'}
    return demos,report
