"""Causal time baseline; actual worker-step numerical path for the learner."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t259_closed_loop_recovery'))
from closed_loop_learning import *


class StepActor(Actor):
    def sequence(self,x):
        assert not self.memory_on and x.ndim==3 and x.shape[-1]==120
        shape=x.shape[:2]
        outputs=[self.step(row[None]) for row in x.reshape(-1,120)]
        return tuple(torch.cat([item[k] for item in outputs]).reshape(*shape,2) for k in [0,1,3])


def policy(plan):
    p=StepActor(L.read(plan['initial_base'])['policy'],False,2026103501)
    p.load_state_dict(L.read(plan['checkpoint'])['policy'])
    return p


_original_batch=L.training_batch


def training_batch(items,p):
    # The original exact recorded-density checks still apply, using StepActor.sequence.
    data=_original_batch(items,p)
    arrays=[i['arrays'] for i in items]
    returns=[S.monte_carlo(a['rewards'],a['native_counts']) for a in arrays]
    old_raw=torch.zeros_like(data['mask'],dtype=torch.float64)
    new_raw=torch.zeros_like(old_raw)
    max_error=0.
    for j,a in enumerate(arrays):
        other=[returns[k][0] for k,i in enumerate(items)
               if k!=j and i['row']['source_reset_id']==items[j]['row']['source_reset_id']]
        assert len(other)==3
        baseline=float(np.mean(other));n=len(a['states'])
        weights=R.GAMMA**np.r_[0,np.cumsum(a['native_counts'].astype(np.int64))[:-1]]
        old_raw[:n,j]=torch.tensor(returns[j]-baseline)
        new_raw[:n,j]=torch.tensor((float(returns[j][0])-baseline)/weights)
        max_error=max(max_error,float(np.max(np.abs(weights*new_raw[:n,j].numpy()-(returns[j][0]-baseline)))))
    # Keep the prior estimator's scale on this same batch, isolating baseline coefficients.
    scale=old_raw[data['mask']].square().mean().sqrt().clamp_min(.05)
    data['advantage']=new_raw[data['mask']]/scale
    assert max_error<2e-12
    data['density'].update(causal_time_baseline=True,original_estimator_RMS_scale=float(scale),
                          constant_weight_coefficient_max_error=max_error)
    return data


# Only this process uses the new batch coefficients. Original source files are immutable.
L.training_batch=training_batch
