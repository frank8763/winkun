"""Offline two-sided value-gradient action probes, never a deployed controller."""
from pathlib import Path
import sys
import gzip
import io
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t272_event_value_pilot'))
from event_model import EventModel


def propose(plan):
    payload=torch.load(io.BytesIO(gzip.decompress(Path(plan['event_checkpoint']).read_bytes())),map_location='cpu',weights_only=False)
    q=EventModel();q.load_state_dict(payload['model']);q=q.double().eval().requires_grad_(False)
    with np.load(plan['reference']['path']) as z:context=z['states'][0].copy();action=z['actions'][0].copy()
    mu=torch.tensor(payload['normalization_mean'],dtype=torch.float64);sc=torch.tensor(payload['normalization_scale'],dtype=torch.float64)
    state=torch.tensor(context,dtype=torch.float64)
    def score(a):
        x=torch.cat([state,a]);prob=q(((x-mu)/sc).clamp(-10,10)[None])[0]
        return prob[4,0]-prob[4,1],prob
    a=torch.tensor(action,dtype=torch.float64,requires_grad=True);u,prob=score(a);g=torch.autograd.grad(u,a)[0]
    sigma=torch.tensor([.035,.03],dtype=torch.float64);scaled=g*sigma;norm=float(scaled.norm())
    assert norm>1e-10 and torch.isfinite(g).all(),'No identified local action derivative'
    delta=plan['marginal_sigma_radius']*sigma*scaled/scaled.norm()
    plus=np.clip(action.astype(np.float64)+delta.numpy(),-.999999,.999999).astype(np.float32)
    minus=np.clip(action.astype(np.float64)-delta.numpy(),-.999999,.999999).astype(np.float32)
    values={};pred={}
    for name,v in [('control',action),('up',plus),('down',minus)]:
        with torch.no_grad():value,pp=score(torch.tensor(v,dtype=torch.float64))
        values[name]=float(value);pred[name]=pp.numpy().tolist()
    assert values['up']>values['control']>values['down'],'No strict predicted ordering at the one fixed radius'
    fd=[]
    for k in range(2):
        d=torch.zeros(2,dtype=torch.float64);d[k]=1e-5
        fd.append(float((score(a.detach()+d)[0]-score(a.detach()-d)[0])/2e-5))
    error=float(np.max(np.abs(np.array(fd)-g.numpy())/np.maximum(1e-8,np.abs(g.numpy()))));assert error<1e-6
    return dict(context120=context.tolist(),actions=dict(control=action.tolist(),up=plus.tolist(),down=minus.tolist()),
        gradient=g.numpy().tolist(),delta_before_native_rounding=delta.numpy().tolist(),score16=values,
        probabilities=pred,finite_difference_relative_error=error,scorer_float64_from_locked_float32_weights=True,
        original_goal_and_frozen_continuation_Actor=True,diagnostic_only_not_onpolicy_or_BC_data=True)
