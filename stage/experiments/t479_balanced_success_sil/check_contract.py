"""Fast zero-native/zero-optimizer mathematical checks; no navigation claim."""
import json
from pathlib import Path
import torch
import numpy as np
import t479_learning as L


class ToyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64))
        self.p = dict(raw_sigma=[1., 1.])

    def sequence(self, x):
        return self.weight * x[..., :1].expand(*x.shape[:-1], 2)


class ToyValue(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))

    def forward(self, x):
        return self.weight.expand(len(x))


def run():
    torch.set_num_threads(1)
    q = ToyActor()
    old = [dict(x=torch.zeros((3+i, 120), dtype=torch.float64),
                target=torch.zeros((3+i, 2), dtype=torch.float64), task=i) for i in range(8)]
    own = []
    for side, count, feature in [('LEFT', 5, 1.), ('RIGHT', 3, 3.)]:
        for i in range(count):
            x = torch.full((2+i, 120), feature, dtype=torch.float64)
            own.append(dict(x=x, target=torch.zeros((len(x), 2), dtype=torch.float64), side=side))
    # Episode lengths and 5:3 multiplicity must not change side totals.
    loss, ko, kn = L.keep_loss(q, dict(old=old, own=own), 0)
    assert float(loss.detach()) == 1.25 and float(ko.detach()) == 0. and float(kn.detach()) == 2.5
    loss.backward()
    torch.testing.assert_close(q.weight.grad, torch.tensor(2.5, dtype=torch.float64), rtol=0., atol=1e-14)
    q.weight.grad = None
    a = dict(rewards=np.array([1., 2.]), native_counts=np.array([4, 100]),
             end_flags=np.array([[0, 0, 0], [0, 1, 0]], dtype=np.uint8))
    assert np.array_equal(L.returns(a, .9), np.array([1.+.9**4*2., 2.]))
    a['end_flags'][-1] = 0
    try:
        L.returns(a, .9)
    except AssertionError:
        pass
    else:
        raise AssertionError('Administrative censor was accepted as complete return')
    v = ToyValue()
    x = torch.zeros((2, 120), dtype=torch.float64)
    memory = [dict(side=s, source_sha256=s, x=x,
                   raw=torch.tensor([[1.2, 0.], [0., 0.]], dtype=torch.float64),
                   atom=torch.tensor([[1, 0], [0, 0]], dtype=torch.int8),
                   G=torch.tensor([2., -1.], dtype=torch.float64)) for s in ('LEFT', 'RIGHT')]
    # Need a nonzero input to differentiate the Actor's tail probability.
    x[:, 0] = .25
    al, vl, positive = L.sil_losses(q, v, memory, dict(raw_sigma=[.2, .2]))
    al.backward()
    assert v.weight.grad is None and float(q.weight.grad) < 0
    assert positive == dict(LEFT=.5, RIGHT=.5)
    vl.backward()
    assert float(v.weight.grad) == -1.
    # A genuine stored closed-loop counterexample, without another simulation:
    # 19/54 alone must not admit a policy that destroys old successes.
    import t479_runtime as T
    previous = Path(__file__).resolve().parents[3] / 'stage/runs/t478_wide8_pooled_credit'
    p = json.loads((previous / 'protocol.json').read_text())
    saved = json.loads((previous / 'learn/rounds.json').read_text())[0]
    real = []
    with (previous / 'learn/cases.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            if r['role'] == 'FULL383_0':
                real.append((None, r))
            if len(real) == 383:
                break
    verdict = T.assess(real, saved['proposal'], p)
    assert not verdict['accepted'] and len(verdict['lost_ids']) == 33
    assert verdict['wide8']['success'] == 19 and verdict['wide6']['success'] == 34
    assert verdict['old275']['success'] == 266
    out = dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER_MATH_CONTRACT',
               native=0, optimizer=0, real_Actor_updates=0,
               synthetic_backward_calls=3, equal_side_weights_despite_5_3_and_unequal_lengths=True,
               finite_MC_native_discounts_and_censor_rejection=True,
               SIL_positive_advantage_boundary_tail_and_Value_detach=True,
               real_19_of_54_but_33_old_successes_lost_candidate_rejected=True,
               actual_full8_capture_and_live_shadow_preflight_NOT_yet_executed=True)
    print(json.dumps(out))
    return out


if __name__ == '__main__':
    run()
