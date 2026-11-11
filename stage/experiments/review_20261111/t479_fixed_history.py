"""Posthoc CPU forward only; immutable captured inputs, no optimizer/native.

This measures function differences on fixed histories, NOT closed-loop causal
attribution or a navigation pass. It deliberately does not use CUDA or backward.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'stage/experiments/t479_balanced_success_sil'))
import t479_learning as L


def main():
    L.torch.set_num_threads(1)
    L.torch.use_deterministic_algorithms(True)
    out = ROOT / 'stage/docs/review_20261111/T479_FIXED_HISTORY_POSTHOC.json'
    assert not out.exists(), 'Preserve prior analysis; no overwrite'
    run = ROOT / 'stage/runs/t479_balanced_success_sil'
    p = L.read(run / 'protocol.json')
    cap = L.read(run / 'capture/summary.json')['result']
    q0 = L.P.frozen(p, p['initial']['path'])
    models = {a: run / f'pair/proposal_{a}.pt.gz' for a in ('A', 'B')}
    candidates = {a: L.P.frozen(p, str(f)) for a, f in models.items()}
    groups = []
    for r in cap['success_rows']:
        assert L.sha(r['path']) == r['sha256']
        a = L.E.N.S.arrays(r['path'])
        assert r['success'] and not r['collision'] and a['end_flags'][-1, 1]
        groups.append(dict(id=r['job_id'], side=L.side(r['job_id']),
                           kind='eight_current_success', x=a['states'], source_sha=r['sha256']))
    # This deterministic schedule is exactly epoch0's old eight-task sample,
    # not selected posthoc for unusually high error.
    with L.np.load(p['old_states']) as z:
        for task in range(8):
            i = int(L.np.flatnonzero(z['task'] == task)[0])
            groups.append(dict(id=str(z['case_ids'][i]), side=None,
                               kind='old_epoch0_retention_context',
                               x=z['states'][z['offsets'][i]:z['offsets'][i+1]].copy(),
                               source_sha=L.sha(p['old_states'])))
    rows = []
    with L.torch.no_grad():
        for g in groups:
            x = L.torch.tensor(g['x'])
            before = q0.sequence(x)
            row = {k: v for k, v in g.items() if k != 'x'}
            row['decisions'] = len(x)
            row['arms'] = {}
            for name, q in candidates.items():
                after = q.sequence(x)
                d = after - before
                action_d = after.clamp(-1, 1).float().double() - before.clamp(-1, 1).float().double()
                sd = before.new_tensor(p['raw_sigma'])
                n = min(25, len(x))
                row['arms'][name] = dict(
                    location_RMSE=float(d.square().mean().sqrt()),
                    max_abs_location=float(d.abs().max()),
                    max_abs_executed_action=float(action_d.abs().max()),
                    final5s_location_RMSE=float(d[-n:].square().mean().sqrt()),
                    preclip_Gaussian_KL_episode_mean=float(.5*((d/sd)**2).sum(-1).mean()))
            rows.append(row)
    result = dict(state='POSTHOC_FIXED_HISTORY_FORWARD_COMPLETE_NOT_CAUSAL',
                  epoch=time.time(), pid=os.getpid(), native=0, optimizer=0,
                  backward=0, CUDA=False, policy_parameters_modified=0,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  protocol_sha256=L.sha(run / 'protocol.json'),
                  initial_model_sha256=L.sha(p['initial']['path']),
                  proposal_sha256={a: L.sha(f) for a, f in models.items()}, rows=rows,
                  caveats=['Preclip KL is a proxy, not executed mixed-measure KL.',
                           'Same input history isolates function difference; no resulting closed-loop rollout was run.',
                           'Descriptive output drift does not establish gradient conflict or a unique cause.',
                           'Eight old contexts do not represent the complete old275.'])
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(state=result['state'], rows=len(rows), native=0, optimizer=0, backward=0)))


if __name__ == '__main__':
    main()
