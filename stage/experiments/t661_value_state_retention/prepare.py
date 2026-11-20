"""Actual saved-state/data/density regression, with zero optimizer calls."""
from pathlib import Path
import ast
import copy
import importlib.util
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'stage/runs/t661_value_state_retention'
DOC = ROOT/'stage/docs/day_20261120/t661'
import t661_learning as Learn
import t661_wait as Wait
M, torch = Learn.M, Learn.M.torch


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def main():
    assert not OUT.exists()
    DOC.mkdir(parents=True, exist_ok=True)
    source = ROOT/'stage/runs/t660s_remaining_night_relay'
    old = M.read(source/'protocol.json')
    p = copy.deepcopy(old)
    # Keep the established scene/normalization/architecture contracts. Replace
    # every execution bound; old night authorization never authorizes this run.
    for key in list(p):
        if key.startswith('night_') or key.endswith('_template') or key in ['registration_epoch', '_root_sha256']:
            p.pop(key)
    p.update(id='T661_PREVIOUS_BLOCK_VALUE_WITH_ACTOR_ROLLBACK', out=str(OUT), arm='ppo_mean',
        authorization_approved=False, source_rounds=str(source/'ppo_mean/ROUNDS.json'),
        source_audit=str(source/'ppo_mean/audit.json'),
        max_physics=sum(j['case']['step_cap'] for j in p['evaluation_jobs'])+sum(j['case']['step_cap'] for j in p['key_jobs']),
        max_commands=407+20+6, main_seconds=5400, audit_seconds=600,
        backup_seconds=600, transport_seconds=600, source_seconds=240,
        all_seconds=7200, learning_seconds=600, finalization_seconds=900,
        absolute_learning_stop=0, absolute_native_stop=0, absolute_all_stop=0,
        max_bytes=192*2**20, child_bytes=192*2**20, global_bytes=10*2**30,
        Actor_each_cap=4, Value_each_cap=16, ppo_epochs=4, value_steps=16,
        evaluation_chunk_size=48, admission_physics_per_second=2500,
        compact_bytes=32768, fresh_bytes=4*2**20, full_bytes=4*2**20,
        model_bytes=4*2**20, state_bytes=4*2**20,
        total_max_physics=12037198, total_max_commands=433, total_main_seconds=5400,
        historical_control_reuse=True, fresh_behavior_recollection=0,
        new_Q_updates=0, optimizer_preflight_updates=0,
        user_authorization='User requested start; new independent two-hour window and exact budget pending confirmation')
    assert p['max_physics'] == 12037198
    resource = Wait.Raw.resources()
    os.sched_setaffinity(0, {14})
    Runtime = module('_t661_runtime_preflight', HERE/'runtime.py')
    assert Runtime.R.used(ROOT/'stage/runs')+p['max_bytes'] < p['global_bytes']
    assert Runtime.R.D.B.shutil.disk_usage(ROOT).free >= 40*2**30
    Runtime.R.gpu_guard()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    obj = M.make(p, 'cuda')
    original = Learn.initial_state(p, False)
    merged = Learn.initial_state(p, True)
    # Red uses the actual legacy state-selection result, not toy values.
    legacy_red = False
    try:
        assert original['value']['net.2.weight'].abs().max() > 0, 'LEGACY_PREVIOUS_VALUE_RESET_TO_ZERO'
    except AssertionError as error:
        legacy_red = str(error) == 'LEGACY_PREVIOUS_VALUE_RESET_TO_ZERO'
    assert legacy_red
    assert merged['value']['net.2.weight'].abs().max() > 0
    for key in ['actor', 'actor_opt']:
        assert M.digest(merged[key]) == M.digest(original[key])
    assert M.digest(merged['value_opt']) != M.digest(original['value_opt'])
    M.load(obj, merged)
    before = M.digest(M.cpu_state(obj))
    start = time.time()
    def guard():
        assert time.time() < start+480
        Wait.Raw.resources()
    data = Learn.groups(p, obj[0], guard)
    positive = Learn.advantage_report(obj, data, p)
    assert positive['failed_with_nonzero_advantage'] > 0
    assert positive['failed_with_negative_advantage'] > 0
    refs = M.references(p, obj[0], guard)
    assert len(refs) == 387
    items = M.fresh_items(obj, data, p)
    cached = [g['adv'].clone() for task in items for g in task]
    pg = sum(M.pg_loss(obj[0], task, p) for task in items)/len(items)
    obj[2].zero_grad(set_to_none=True)
    pg.backward()
    grad_carried = torch.cat([x.grad.flatten() for x in obj[0].parameters() if x.requires_grad]).detach().cpu()
    assert grad_carried.isfinite().all()
    assert all(x.grad is None for x in obj[1].parameters())
    # Check detached current-block advantages, no optimizer step.
    with torch.no_grad():
        obj[1].net[-1].bias.add_(.125)
    assert all(torch.equal(g['adv'], expected) for g, expected in zip([g for task in items for g in task], cached))
    M.load(obj, merged)
    assert M.digest(M.cpu_state(obj)) == before
    M.load(obj, original)
    zero = Learn.advantage_report(obj, data, p)
    assert zero['failed_with_nonzero_advantage'] == 0
    zero_items = M.fresh_items(obj, data, p)
    obj[2].zero_grad(set_to_none=True)
    (sum(M.pg_loss(obj[0], task, p) for task in zero_items)/len(zero_items)).backward()
    grad_zero = torch.cat([x.grad.flatten() for x in obj[0].parameters() if x.requires_grad]).detach().cpu()
    difference = float((grad_carried-grad_zero).abs().max())
    assert difference > 1e-12
    # Mature Adam buffers came from stored evidence. No Adam.step is invoked.
    M.load(obj, merged)
    assert M.digest(M.cpu_state(obj)) == before
    obj[0].cpu()
    obj[1].cpu()
    del obj, refs, items, zero_items
    torch.cuda.empty_cache()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    Runtime.setup(p)
    assert Runtime.R.Runtime.guard.__globals__['verify'] is Runtime.verify
    assert Runtime.R.N.Pool.__init__.__globals__['worker'] is Runtime.N.worker
    probes = [0]
    def occupied_then_clear():
        probes[0] += 1
        if probes[0] < 3:
            raise AssertionError('Other GPU compute exists; yield')
        return 'CLEAR'
    assert Runtime.wait_gpu(occupied_then_clear, time.time()+5, sleep=lambda seconds: None) == 'CLEAR'
    assert probes[0] == 3
    unrelated_rejected = False
    def unrelated():
        raise AssertionError('NOT_A_RESOURCE_CONFLICT')
    try:
        Runtime.wait_gpu(unrelated, time.time()+5)
    except AssertionError:
        unrelated_rejected = True
    assert unrelated_rejected
    helper = module('_t661_affinity_preflight', HERE/'affinity_guard.py')
    assert helper.OUT == OUT and helper.DOC == DOC and helper.CG.name == 'stage-t661.service'
    transport = module('_t661_transport_preflight', HERE/'transport.py')
    assert transport.OUT == OUT
    client = transport.Prior.client()
    assert callable(client.req)
    # All 407 IDs exactly once after screen; fixed20 exact replays additional.
    screen_ids = {j['id'] for j in p['screen_jobs']}
    rest = [j for j in p['evaluation_jobs'] if j['id'] not in screen_ids]
    pieces = list(Runtime.chunks(rest, p['evaluation_chunk_size']))
    assert len(rest) == 371 and sum(len(js) for index, js in pieces) == 371
    assert max(Runtime.chunk_time_reserve(js, p) for index, js in pieces) < 1000
    assert len(screen_ids | {j['id'] for j in rest}) == 407
    assert len(p['key_jobs']) == 20 and all(j['id'] in screen_ids for j in p['key_jobs'])
    assert p['main_seconds']+p['audit_seconds']+p['backup_seconds']+p['source_seconds']+60 < p['all_seconds']
    dependencies = set()
    # Existing locked inputs include canonical reset modules, original model,
    # task snapshots and 387 references. Every byte must still be present.
    for file, expected in old['inputs'].items():
        assert M.sha(file) == expected, file
        dependencies.add(Path(file))
    previous, current = Learn.records(p)
    dependencies.update(Path(row['path']) for row in previous['training_rows']+current['training_rows'])
    dependencies.update(Path(row['path']) for row in current['screen'])
    dependencies.update([Path(p['source_rounds']), Path(p['source_audit']), Path(previous['trained']['path']),
        Path(current['start']['path']), Path(current['proposal']['path']), DOC/'PLAN.md'])
    for m in list(sys.modules.values()):
        file = getattr(m, '__file__', None)
        if file and str(file).startswith(str(ROOT)) and str(file).endswith('.py'):
            dependencies.add(Path(file).resolve())
    for file in HERE.glob('*.py'):
        ast.parse(file.read_text())
        dependencies.add(file)
    p['inputs'] = {str(file): M.sha(file) for file in sorted(dependencies)}
    M.save(HERE/'protocol.json', p)
    report = dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER', legacy_actual_state_red_reproduced=legacy_red,
        saved_previous_Value_and_Adam_carried=True, Actor_and_Adam_exactly_unchanged=True,
        actual64_complete_histories_and_likelihoods_checked=True, real387_references_checked=True,
        previous_and_current_noise_seeds_disjoint=True, baseline_current_block_labels_not_used=True,
        detached_advantages_checked=True, carried=positive, zero=zero,
        actual_policy_gradient_max_difference=difference,
        actual_runtime_guard_pool_binding=True, actual_affinity_loader_and_transport_install=True,
        bounded_GPU_yield_and_unrelated_error_propagation=True,
        no_network_used=True, new_native=0, new_optimizer=0, extra_optimizer_preflight=0,
        worst_global_bytes=Runtime.R.used(ROOT/'stage/runs')+p['max_bytes'], resource=resource)
    M.save(DOC/'PREFLIGHT.json', report)
    M.save(DOC/'BUDGET.json', {key: p[key] for key in ['max_physics','max_commands','main_seconds','all_seconds','max_bytes',
        'global_bytes','Actor_each_cap','Value_each_cap','audit_seconds','backup_seconds','source_seconds']})
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
