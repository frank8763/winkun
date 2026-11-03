"""Bounded architecture preparation: no simulator, optimizer or parameter update.

One autograd call checks isolation. No fitted model is saved. This is not a
training preflight for a completed native pipeline, nor a navigation result.
"""
from pathlib import Path
import copy, hashlib, io, json, shutil, sys, time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'stage/experiments/t319_imperfect_demo_head'))
import t319_core as H
from mem_actor import R, Actor
from actor import FrozenBaseResidual

torch, np, K = H.torch, H.np, H.K
TARGET = ROOT/'stage/docs/distance_20261103/FROZEN_RESIDUAL_CONTRACT.json'


def main():
    assert not TARGET.exists(), 'Do not overwrite prior preparation evidence'
    start = time.monotonic()
    p = H.read(ROOT/'stage/experiments/t322_full_network_steering/protocol.json')
    input_files = [p[k] for k in ('checkpoint', 'initial_base', 'train_demo',
                                 'validation_demo', 'retention_states')]
    before = {f: K.C.sha(f) for f in input_files}
    q = K.policy(p).eval().requires_grad_(False)
    parent_hash = K.S.digest(q.state_dict())
    assert parent_hash == p['policy_hash']
    train, _ = H.demo(p['train_demo'])
    valid, _ = H.demo(p['validation_demo'])
    old, _ = H.anchors(p)
    # Validation never supplies normalization or fitted parameters.
    fit_x = np.concatenate([train, old]).astype(np.float64)
    mean = torch.tensor(fit_x.mean(0), dtype=torch.float32)
    scale = torch.tensor(np.maximum(fit_x.std(0), 1e-3), dtype=torch.float32)
    model = FrozenBaseResidual(q, R.center, Actor.mode, mean, scale).eval()
    initial = K.S.digest(model.state_dict())
    x = torch.tensor(np.concatenate([train, valid, old]))
    assert x.shape == (816, 120)
    checks = []
    with torch.no_grad():
        for row in x:
            assert time.monotonic()-start < 250
            a, b = q.step(row[None]), model.step(row[None])
            checks.append(all(torch.equal(i, j) for i, j in zip(a, b)))
        assert all(checks)
        seq = model.sequence(x[:16,None,:])
        rows = [model.step(row[None]) for row in x[:16]]
        assert all(torch.equal(seq[n][:,0], torch.cat([r[k] for r in rows]))
                   for n,k in enumerate((0,1,3)))
        arbitrary_hidden = torch.ones(1, 1, 32)
        assert all(torch.equal(a,b) for a,b in zip(model.step(x[:1]), model.step(x[:1], arbitrary_hidden)))
        # Native-domain bounded Gaussian density identity; not fake expert logp.
        means = torch.cat([model.step(r[None])[1] for r in x[:16]])
        baseline_means = torch.cat([q.step(r[None])[1] for r in x[:16]])
        action = torch.zeros((16,2))
        assert torch.equal(R.log_density(means, action), R.log_density(baseline_means, action))
    # Exactly one gradient call, no optimizer instance or step.
    model.zero_grad(set_to_none=True)
    model.step(x[:16])[3].sum().backward()
    assert all(v.grad is None for v in model.frozen.parameters())
    gradients = {n:float(v.grad.norm()) if v.grad is not None else None
                 for n,v in model.correction.named_parameters()}
    assert gradients['2.weight'] > 0 and gradients['2.bias'] > 0
    # Zero output initialization intentionally gives zero first-layer gradient
    # at the first update; do not report every layer as already learning.
    assert gradients['0.weight'] == gradients['0.bias'] == 0
    assert K.S.digest(model.state_dict()) == initial
    b = io.BytesIO(); torch.save(model.state_dict(), b); b.seek(0)
    recovered = copy.deepcopy(model)
    recovered.load_state_dict(torch.load(b, weights_only=True))
    assert K.S.digest(recovered.state_dict()) == initial
    assert K.S.digest(model.frozen.state_dict()) == parent_hash
    assert {f:K.C.sha(f) for f in input_files} == before
    used = K.S.used(ROOT/'stage/runs'); free = shutil.disk_usage(ROOT).free
    assert used <= 1024**3 and free >= 40*1024**3
    report = dict(state='PASS_ARCHITECTURE_ONLY', native_commands=0, physics_steps=0,
        optimizer_calls=0, Actor_parameter_updates=0, autograd_calls=1,
        navigation_training_started=False, native_worker_integration_verified=False,
        zero_initialization_exact_rows=len(checks), step_sequence_exact=True,
        hidden_carry_ignored=True, initial_density_exact=True, frozen_gradient_absent=True,
        RAM_save_reload_state_exact=True, first_gradient_norms=gradients,
        trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
        frozen_policy_hash=parent_hash, initial_composite_hash=initial,
        normalization_training_rows=len(fit_x), validation_used_for_normalization=False,
        input_sha256=before,
        source_sha256={str(f.relative_to(ROOT)):K.C.sha(f) for f in [Path(__file__),Path(__file__).with_name('actor.py')]},
        elapsed_seconds=time.monotonic()-start, global_bytes=used, free_bytes=free,
        trained_models_saved=0, scope='Engineering preparation; no scientific efficacy or novelty claim')
    TARGET.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__':
    main()
