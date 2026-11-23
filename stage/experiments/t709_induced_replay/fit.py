"""One fixed data-coverage ablation; reuse the exact historical convex control."""
from pathlib import Path
import copy, fcntl, gzip, importlib.util, io, json, os, sys, time
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('t709_fit_controller', HERE/'run.py')
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
A, OUT = C.A, C.OUT
torch, digest = A.torch, A.digest
sys.path.insert(0, str(ROOT/'stage/experiments/t701_compact_retention'))
import t701_actor as Legacy


def affine_features(head, features):
    hidden = head[1](head[0](features))
    return torch.cat([hidden, torch.ones(len(hidden), 1, dtype=torch.float64)], 1)


def ridge_solve(G, B, W0, rho):
    matrix = G + rho*torch.eye(len(W0), dtype=torch.float64)
    rhs = B + rho*W0
    factor = torch.linalg.cholesky(matrix)
    W = torch.cholesky_solve(rhs, factor)
    residual = float((matrix@W-rhs).abs().max())/max(1., float(rhs.abs().max()))
    assert torch.isfinite(W).all() and residual <= 1e-8
    return W, residual


def construct(p, data, guard, emit):
    # Historical T705 control used eight threads for this exact arithmetic.
    torch.set_num_threads(8)
    q = A.frozen(p, p['repair_initial']['path'])
    teacher = A.P.frozen(p, p['initial']['path'])
    head = q.replay_head
    with torch.no_grad():
        Z = torch.cat([affine_features(head, data['x'][off:off+2048])
                       for off in range(0, len(data['x']), 2048)])
        Y, groups, records = data['y'], data['groups'], data['records']
        W0 = torch.cat([head[2].weight.T, head[2].bias[None]], 0).clone()
        assert Z.shape == (83693, 109) and W0.shape == (109, 2)
        assert not Y[groups < 403].any()
        lengths = torch.tensor([r['length'] for r in records], dtype=torch.float64)
        grams, rhs = [], []
        for isnew, count in [(False, 403), (True, 7)]:
            guard(); take = groups >= 403 if isnew else groups < 403
            z, y, g = Z[take], Y[take], groups[take]
            weights = 1/(2*count*lengths[g])
            grams.append(z.T@(weights[:, None]*z))
            rhs.append(z.T@(weights[:, None]*y))
        rho = 1e-10*float(torch.trace(grams[0]))/109
        # Same arithmetic/order/ridge as T705, not a newly selected baseline.
        emit('BEFORE', 0, 'historical_control_reconstruction')
        control_W, control_residual = ridge_solve(grams[0]+grams[1], rhs[0]+rhs[1], W0, rho)
        emit('AFTER', 0, 'historical_control_reconstruction')
        control = A.frozen(p, p['control_reference']['path'])
        expected_W = torch.cat([control.replay_head[2].weight.T, control.replay_head[2].bias[None]], 0)
        assert torch.equal(control_W, expected_W), 'Historical control reconstruction changed'
        for key, value in q.state_dict().items():
            if key not in ['replay_head.2.weight', 'replay_head.2.bias']:
                assert torch.equal(control.state_dict()[key], value)

        models = {name: A.frozen(p, str(ROOT/'stage/runs/t708_agem_replay/fit'/(name+'.pt.gz')))
                  for name in ['agem_mean', 'agem_worst']}
        for model in [q, *models.values()]:
            for key, value in teacher.state_dict().items():
                assert torch.equal(model.state_dict()[key], value), ('Frozen teacher body mismatch', key)
        torch.set_num_threads(1)
        selected = p['induced_selection']['rows']
        assert len(selected) == 102 and len(set(r['job_id'] for r in selected)) == 51
        per_id = {r['id']: i for i, r in enumerate(records[:403])}
        induced, manifest = {}, []
        for row in selected:
            guard(); assert C.sha(row['path']) == row['sha256']
            arrays = A.Old.Existing.L.T.R.E.N.S.arrays(row['path'])
            assert {key: digest(value) for key, value in arrays.items()} == row['array_hashes']
            assert not arrays['ownership'].any() and not arrays['stochastic'].any()
            model = models[row['arm']]
            assert model.phash() == row['policy_hash']
            assert C.sha(ROOT/'stage/runs/t708_agem_replay/fit'/(row['arm']+'.pt.gz')) == row['loaded_model_sha256']
            # This replays the exact externally recorded observation/action history,
            # from zero carry, and checks every saved carry before/after the history.
            features, teacher_mu = A.P.trajectory(teacher, arrays, row['job_id'], None, guard)
            for index, f in enumerate(features):
                if index % 128 == 0: guard()
                learner_mu = model.location(f[:64][None], f[64:][None])
                assert A.Old.Existing.same_float32(learner_mu.clamp(-1, 1).float().numpy()[0], arrays['actions'][index])
            z = affine_features(head, features)
            induced[(row['job_id'], row['arm'])] = z
            manifest.append(dict(id=row['job_id'], arm=row['arm'], decisions=len(z),
                                 feature_hash=digest(features), design_hash=digest(z), teacher_raw_hash=digest(teacher_mu),
                                 target_residual_hash=digest(torch.zeros_like(teacher_mu)),
                                 trajectory_sha256=row['sha256'], shared_owncarry_exact=True,
                                 learner_executed_commands_exact=True))
            print(json.dumps(dict(stage='induced_history', completed=len(manifest), total=102)), flush=True)
        changed_G = grams[0].clone()
        for key in sorted(set(r['job_id'] for r in selected)):
            guard(); i = per_id[key]; nominal = Z[groups == i]
            gm = nominal.T@nominal/(2*len(nominal))
            extra = []
            for arm in ['agem_mean', 'agem_worst']:
                z = induced[(key, arm)]; extra.append(z.T@z/(2*len(z)))
            # Each old ID retains total weight1/403. Only state distribution changes.
            changed_G += (-.5*gm+.25*extra[0]+.25*extra[1])/403
        emit('BEFORE', 1, 'induced_state_readout_solve')
        W, residual = ridge_solve(changed_G+grams[1], rhs[0]+rhs[1], W0, rho)
        emit('AFTER', 1, 'induced_state_readout_solve')
        model = copy.deepcopy(q)
        model.replay_head[2].weight.copy_(W[:-1].T); model.replay_head[2].bias.copy_(W[-1])
        for key, value in q.state_dict().items():
            if key not in ['replay_head.2.weight', 'replay_head.2.bias']:
                assert torch.equal(model.state_dict()[key], value)
        metrics = {}
        for label, weights in [('original', W0), ('control', control_W), ('induced', W)]:
            old_losses = {}; cover_losses = []
            for i, record in enumerate(records[:403]):
                take = groups == i; nominal_loss = float((Z[take]@weights).square().mean())
                old_losses[record['id']] = nominal_loss
                if (record['id'], 'agem_mean') in induced:
                    cover_losses.append(.5*nominal_loss + sum(.25*float((induced[(record['id'], a)]@weights).square().mean()) for a in ['agem_mean','agem_worst']))
                else: cover_losses.append(nominal_loss)
            source_losses = {record['id']: float((Z[groups == i]@weights-Y[groups == i]).square().mean())
                             for i, record in enumerate(records) if i >= 403}
            old_value = sum(old_losses.values())/403; cover_value = sum(cover_losses)/403
            source_value = sum(source_losses.values())/7
            induced_losses = {arm: {key: float((z@weights).square().mean()) for (key, name), z in induced.items() if name == arm} for arm in ['agem_mean', 'agem_worst']}
            quad_old = float((weights*(changed_G@weights)).sum())
            quad_new = float((weights*(grams[1]@weights)).sum()-2*(weights*rhs[1]).sum())
            const = sum(float(Y[groups == i].square().mean()) for i in range(403,410))/7
            assert abs(quad_old-cover_value) < 1e-10 and abs(quad_new+const-source_value) < 1e-10
            metrics[label] = dict(nominal_old_mean=old_value, mixture_old_mean=cover_value,
                                  source_mean=source_value, nominal_old_perID=old_losses, source_perID=source_losses, induced_perID=induced_losses, induced_means={arm:sum(v.values())/len(v) for arm,v in induced_losses.items()})
        assert control.phash() == p['control_reference']['policy_hash']
        return {'induced': model.eval()}, dict(state_hash=digest(model.state_dict()),
            control_state_exact=True, control_native_reused_by_SHA=True, control_stationarity=control_residual,
            stationarity=residual, rho=rho, nominal_gram_hash=digest(grams[0]), induced_mixture_gram_hash=digest(changed_G),
            source_gram_hash=digest(grams[1]), original_weights_hash=digest(W0), control_weights_hash=digest(control_W),
            candidate_weights_hash=digest(W), teacher_state_hash=digest(teacher.state_dict()), manifest=manifest,
            selected_old_IDs=51, histories=102, induced_decisions=sum(r['decisions'] for r in manifest),
            no_failure_label_selection=True, per_ID_weight_preserved=True, fixed_hidden=True,
            trained_readout_scalar_parameters=218, linear_solves=2, assigned_new_candidate=1, metrics=metrics,
            no_T704_trajectories_or_labels_used=True, offline_single_coverage_ablation_not_iterative_DAgger=True)


def event(who):
    def emit(stage, step, kind):
        with (OUT/'fit/fit_calls.jsonl').open('a') as f:
            f.write(json.dumps(dict(who=who, stage=stage, step=step, kind=kind, epoch=time.time()))+'\n')
    return emit


def main(mode):
    p = C.read(OUT/'protocol.json'); C.verify(p, force=True)
    end = min(time.time()+(1200 if mode == 'main' else 900),
              p['absolute_learning_stop'] if mode == 'main' else p['absolute_all_stop'])-15
    def guard():
        assert time.time() < end; C.verify(p)
    torch.use_deterministic_algorithms(True); torch.set_num_threads(1); os.sched_setaffinity(0, set(range(8,16)))
    lock = (ROOT/'stage/runs/.stage_writer.lock').open('a+b'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        data = torch.load(io.BytesIO(gzip.decompress((ROOT/'stage/runs/t699_matched_value/DATA.pt.gz').read_bytes())), map_location='cpu', weights_only=False)
        who = 'live' if mode == 'main' else 'independent'
        models, info = construct(p, data, guard, event(who))
        if mode == 'main':
            shadow, check = construct(p, data, guard, event('shadow'))
            assert info == check and digest(models['induced'].state_dict()) == digest(shadow['induced'].state_dict())
            del shadow
        ms = importlib.util.spec_from_file_location('t709_reused_cost_metrics', ROOT/'stage/experiments/t701_compact_retention/fit.py')
        F = importlib.util.module_from_spec(ms); ms.loader.exec_module(F)
        metric = F.metrics(p, models['induced'], data, guard, False)
        assert metric['added_tensor_bytes'] == 161584
        if mode == 'main':
            model = Legacy.Old.export(models['induced'], p, OUT/'fit/induced.pt.gz')
            assert digest(A.frozen(p, model['path']).state_dict()) == info['state_hash']
            C.save(OUT/'fit/summary.json', dict(state='COMPLETE', finals=dict(induced=dict(model=model, metrics=metric)), construction=info, new_Adam=0, new_RL=0))
        else:
            saved = C.read(OUT/'fit/summary.json'); assert info == saved['construction']
            assert digest(A.frozen(p, saved['finals']['induced']['model']['path']).state_dict()) == info['state_hash']
            for key, value in saved['finals']['induced']['metrics'].items():
                if not key.startswith(('batch1', 'peak_')): assert metric[key] == value, key
            C.save(OUT/'fit/audit.json', dict(state='PASS', independent_full_features_teacher_carry_commands_and_solve_exact=True,
                control_state_exact=True, linear_solves=2, candidate_assignments=1, new_native=0))
        C.verify(p, force=True)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN); lock.close()


if __name__ == '__main__': main(sys.argv[1])
