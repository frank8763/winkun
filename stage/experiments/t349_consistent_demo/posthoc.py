"""Consumed-data inspection only: no simulator, fitting, or selection of a new policy."""
from t349_core import *
import signal


def main():
    target = DOC / 'T349_POSTHOC.json'
    assert not target.exists()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('180s posthoc limit')))
    signal.alarm(180)
    torch.set_num_threads(1)
    p = read(OUT / 'protocol.json')
    human_path = Path(p['sequence_sources'][0]['path'])
    human = arrays(human_path)
    rows = [json.loads(v) for v in (OUT / 'cases.jsonl').read_text().splitlines()]
    result = dict(scope='Posthoc consumed development data; not heldout evaluation or a causal ablation',
                  native_added=0, optimizer_added=0, arms={}, input_sha256={str(human_path): K.C.sha(human_path)})
    for arm in ['reset', 'memory']:
        model_path = OUT / (arm + '_final.pt.gz')
        snapshot = torch.load(io.BytesIO(gzip.decompress(model_path.read_bytes())), weights_only=False, map_location='cpu')
        q = Actor(p, arm).eval().requires_grad_(False)
        q.load_learned(snapshot['state']['learned'])
        before = q.phash()
        predictions = []
        hidden = None
        with torch.no_grad():
            for x in human['states']:
                action, _, new, _ = q.step(torch.tensor(x[None]), hidden)
                predictions.append(action[0].numpy())
                hidden = new if q.memory_on else None
        pred = np.asarray(predictions)
        error = pred.astype(float) - human['actions'].astype(float)
        final = [r for r in rows if r['arm'] == arm and r['role'] == 'final27']
        matched = next(r for r in final if r['job_id'] == 'wide_dev_1')
        a = arrays(matched['path'])
        distances = np.linalg.norm(a['next_poses'][:, :2] - np.asarray(matched['goal_xy']), axis=1)
        near = int(distances.argmin())
        deviation = []
        for k in [0, 25, 50, 76, 100, 150]:
            if k >= min(len(a['poses']), len(human['poses'])):
                continue
            yaw = (a['poses'][k, 2] - human['poses'][k, 2] + np.pi) % (2*np.pi) - np.pi
            deviation.append(dict(decision=k, time_s=k*.2,
                XY_deviation_m=float(np.linalg.norm(a['poses'][k, :2]-human['poses'][k, :2])),
                yaw_deviation_deg=float(np.degrees(yaw))))
        def short(r):
            return {k: r[k] for k in ['job_id', 'success', 'collision', 'timeout', 'physics_steps',
                                     'remaining_m', 'integration_sha256', 'full_arrays_saved']}
        centers = [r for r in final if r['job_id'].startswith('dev_task') and r['job_id'].endswith('_13')]
        assert len(centers) == 8
        result['arms'][arm] = dict(
            human_training_state_drive_RMSE=float(np.sqrt(np.mean(error[:, 0]**2))),
            human_training_state_steer_RMSE=float(np.sqrt(np.mean(error[:, 1]**2))),
            first_action=pred[0].tolist(), old8_centers_success=sum(r['success'] for r in centers),
            old20_failed_ids=[r['job_id'] for r in final[:20] if not r['success']],
            wide5=[short(r) for r in final[20:25]], original2=[short(r) for r in final[25:]],
            matched_human_start_minimum_recorded_endpoint_distance_m=float(distances[near]),
            matched_minimum_endpoint_time_s=float(a['native_counts'][:near+1].sum()*.002),
            matched_endpoint_minimum_is_not_continuous_native_clearance=True,
            matched_closed_loop_path_deviations=deviation)
        assert q.phash() == before
        result['input_sha256'].update({str(model_path): K.C.sha(model_path), matched['path']: K.C.sha(matched['path'])})
    history = {}
    for folder in ['t340_reverse_demo_rl', 't342_balanced_consolidation', 't348_causal_sequence_warmstart']:
        path = ROOT / 'stage/runs' / folder / 'cases.jsonl'
        if not path.exists():
            continue
        old = [json.loads(v) for v in path.read_text().splitlines()]
        history[folder] = [dict(id=r['id'], success=r['success'], physics_steps=r['physics_steps'])
                           for r in old if r.get('success') and ('wide_dev_2' in r['id'] or 'wide_dev_4' in r['id'])]
        result['input_sha256'][str(path)] = K.C.sha(path)
    result['checked_historical_dev2_dev4_successes'] = history
    result['historical_check_is_not_an_exhaustive_all_experiment_search'] = True
    result['attribution_limits'] = [
        'T347 exposure increased from120 to600 when other four sources were removed.',
        'One paired initialization and five repeatedly consumed development starts; no unseen success rate.',
        'Two successful compact trajectories were full-array compared in RAM on native model readback; full states are not persisted for them.',
        'Closed loop errors and loss do not uniquely identify the cause; original narrow task and old preservation remain unsolved.'
    ]
    save(target, result)
    signal.alarm(0)
    print(json.dumps(dict(state='POSTHOC_COMPLETE', native=0, optimizer=0)))


if __name__ == '__main__':
    main()
