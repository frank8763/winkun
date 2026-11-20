"""Zero-step, zero-optimizer comparison of four saved complete histories."""
from pathlib import Path
import fcntl
import gzip
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'stage/runs/t662_frozen_failure_diagnosis'
DOC = ROOT/'stage/docs/day_20261120/t662'
sys.path.insert(0, str(ROOT/'stage/experiments/t661r_value_state_isolation'))
import t661r_learning as L
import t661r_wait as Resource
M, np, torch = L.M, L.M.np, L.M.torch
IDS = ['dev_task6_00', 'dev_task6_26', 'wide117_8m_dev6_RIGHT_18', 'dev_task6_03']


def read_rows(folder):
    return [json.loads(line) for line in gzip.decompress((folder/'cases.jsonl.gz').read_bytes()).decode().splitlines()]


def first_event(times, values, threshold):
    indices = np.flatnonzero(values >= threshold)
    return None if not len(indices) else dict(decision=int(indices[0]), seconds=float(times[indices[0]]), value=float(values[indices[0]]))


def outcome(a):
    flags = a['end_flags'][-1]
    return dict(flags=flags.tolist(), terminal_steps=int(a['native_counts'].sum()),
        terminal_seconds=float(a['native_counts'].sum()*.002),
        goal_endpoint_distance_m=float(np.linalg.norm(a['next_poses'][-1,:2]-a['task_goal_xy'])))


def compare(p, reference, row, q0, q):
    assert M.sha(reference['source_path']) == reference['sha256']
    assert M.sha(row['path']) == row['sha256'] and row['full'] and row['complete']
    a = M.E.N.S.arrays(reference['source_path'])
    b = M.E.N.S.arrays(row['path'])
    assert all(M.digest(value) == row['array_hashes'][key] for key, value in b.items())
    same = {key: bool(np.array_equal(a[key], b[key])) for key in ['initial_native_integration','task_goal_xy','scene_sha256']}
    same.update(initial120=bool(np.array_equal(a['states'][0], b['states'][0])),
        initial_pose=bool(np.array_equal(a['poses'][0], b['poses'][0])),
        initial_hidden=bool(np.array_equal(a['hidden_before'][0], b['hidden_before'][0])))
    assert all(same.values()), (row['job_id'], same)
    assert row['policy_hash'] == q.phash()
    n = min(len(a['states']), len(b['states']))
    ta = np.r_[0, np.cumsum(a['native_counts'])[:-1]][:n]*.002
    tb = np.r_[0, np.cumsum(b['native_counts'])[:-1]][:n]*.002
    assert np.array_equal(ta, tb), 'Compare only aligned original-start decision histories'
    with torch.no_grad():
        x = torch.as_tensor(a['states'])
        y = torch.as_tensor(b['states'])
        old_ref = q0.sequence(x).clamp(-1,1).cpu().numpy().astype(np.float32)
        new_ref = q.sequence(x).clamp(-1,1).cpu().numpy().astype(np.float32)
        new_own = q.sequence(y).clamp(-1,1).cpu().numpy().astype(np.float32)
    assert float(np.max(np.abs(old_ref-a['actions']))) < 1e-6
    assert float(np.max(np.abs(new_own-b['actions']))) < 1e-6
    direct = new_ref[:n].astype(float)-old_ref[:n]
    feedback = new_own[:n].astype(float)-new_ref[:n]
    total = new_own[:n].astype(float)-old_ref[:n]
    assert np.max(np.abs(direct+feedback-total)) < 1e-12
    xy = np.linalg.norm(a['poses'][:n,:2]-b['poses'][:n,:2], axis=1)
    hidden = np.linalg.norm(a['hidden_before'][:n]-b['hidden_before'][:n], axis=1)
    endpoint = np.linalg.norm(b['next_poses'][:,:2]-b['task_goal_xy'], axis=1)
    closest = int(endpoint.argmin())
    elapsed_end = np.cumsum(b['native_counts'])*.002
    return dict(reference_sha256=reference['sha256'], candidate_sha256=row['sha256'],
        same_initial_contract=same, reference_source_policy=reference['source_policy'],
        source_outcome=outcome(a), candidate_outcome=outcome(b),
        candidate_success=bool(row['success'] and not row['collision']), collision=row['collision'], timeout=row['timeout'],
        same_history_command_RMS=float(np.sqrt(np.mean(direct**2))),
        same_history_command_max=float(np.max(np.abs(direct))),
        own_history_feedback_command_RMS=float(np.sqrt(np.mean(feedback**2))),
        first_command_difference=first_event(ta,np.max(np.abs(total),axis=1),1e-8),
        first_position_1cm=first_event(ta,xy,.01), first_position_10cm=first_event(ta,xy,.1),
        first_hidden_L2_001=first_event(ta,hidden,.01),
        closest_recorded_endpoint_m=float(endpoint[closest]), closest_recorded_endpoint_seconds=float(elapsed_end[closest]),
        final_endpoint_distance_m=float(endpoint[-1]),
        endpoint_distance_increase_after_closest_m=float(endpoint[-1]-endpoint[closest]),
        last10_drive_mean=float(np.mean(b['actions'][-10:,0])),
        last10_steer_mean=float(np.mean(b['actions'][-10:,1])),
        terminal_contacts=row.get('terminal_contacts'),
        algebraic_feedback_decomposition_not_causal_proof=True,
        closest_endpoint_is_not_continuous_physics_minimum=True)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.nice(19)
    Resource.Raw.resources()
    os.sched_setaffinity(0,{14})
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    original = ROOT/'stage/runs/t661r_value_state_isolation'
    p = M.read(original/'protocol.json')
    report = M.read(original/'FINAL_REPORT_FINAL.json')
    sealed = M.read(original/'SEALED.json')
    assert sealed['all_scientific_processes_stopped']
    assert all(not Path('/proc',str(pid)).exists() for pid in sealed['scientific_pids'])
    assert not OUT.exists() and time.time() < p['absolute_all_stop']
    lock = (ROOT/'stage/runs/.stage_writer.lock').open('a+b')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir()
    DOC.mkdir(parents=True,exist_ok=True)
    started = time.time()
    manifest = M.read(p['reference_contract'])
    rows = read_rows(original)
    previous = ROOT/'stage/runs/t660s_remaining_night_relay/ppo_mean'
    zero_record = M.read(previous/'ROUNDS.json')[1]
    zero_rows = read_rows(previous)
    source_files = [HERE/'analyze.py', original/'protocol.json', original/'FINAL_REPORT_FINAL.json',
        Path(p['reference_contract']), original/'cases.jsonl.gz', previous/'cases.jsonl.gz']
    models = dict(T501=p['initial'], zero_Value=zero_record['proposal'], retained_Value=report['summary']['learning']['proposal'])
    loaded = {key:M.L.frozen(p,model['path']) for key,model in models.items()}
    for key,model in models.items():
        assert M.sha(model['path']) == model['sha256'] and loaded[key].phash()==model['policy_hash']
        source_files.append(Path(model['path']))
    result = {}
    for task_id in IDS:
        assert time.time() < min(started+300,p['absolute_all_stop']-60)
        ref = next(x for x in manifest['reference_rows'] if x['task_id']==task_id)
        result[task_id] = {}
        source_files.append(Path(ref['source_path']))
        for label,candidates in [('zero_Value',zero_rows),('retained_Value',rows)]:
            selected = [row for row in candidates if row['job_id']==task_id and row['full'] and row['seed'] is None]
            if not selected:
                result[task_id][label] = None
                continue
            row = selected[-1]
            source_files.append(Path(row['path']))
            result[task_id][label] = compare(p,ref,row,loaded['T501'],loaded[label])
    budget = dict(new_native=0,new_optimizer=0,new_GPU=0,seconds=time.time()-started,max_seconds=300,max_bytes=2**20)
    output = dict(state='COMPLETE_ZERO_NATIVE_ZERO_OPTIMIZER', source_experiment_state=report['state'],
        source_independent_audit=None, source_backup=report['remote'],
        source_optimizer_attempts=report['optimizer_attempts'],
        source_audit_failure=report['error'], results=result, budget=budget,
        models=models, inputs={str(file):M.sha(file) for file in sorted(set(source_files))})
    M.save(OUT/'RESULTS.json',output)
    M.save(DOC/'RESULTS.json',output)
    assert sum(file.stat().st_size for file in OUT.rglob('*') if file.is_file()) < 2**20
    print(json.dumps(dict(state=output['state'],budget=budget,results={key:{label:None if item is None else {k:item[k] for k in ['candidate_success','collision','timeout','same_history_command_RMS','first_position_1cm','closest_recorded_endpoint_m','final_endpoint_distance_m']} for label,item in values.items()} for key,values in result.items()})))


if __name__ == '__main__':
    main()
