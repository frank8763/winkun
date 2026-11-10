"""Bounded CPU-only posthoc label-fit audit; no native, optimizer or backward."""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import os
import signal
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'stage/docs/review_20261110/T466_LABEL_AUDIT.json'
SOURCE = ROOT/'stage/experiments/t465_matched_safe_recovery/run.py'


def read(f):
    return json.loads(Path(f).read_bytes())


def sha(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def stopped(folder):
    sealed = read(folder/'SEALED.json')
    assert sealed['all_scientific_processes_stopped']
    ids = sealed['scientific_pids']+[read(folder/'driver_started.json')['pid']]
    launch = read(folder/'launch.json')
    ids.append(launch['pid'])
    assert all(not Path('/proc', str(pid)).exists() for pid in ids)
    return ids


def main():
    start = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('T466300s posthoc limit')))
    signal.alarm(300)
    assert not OUT.exists(), 'Exclusive posthoc report; never overwrite'
    spec = importlib.util.spec_from_file_location('stage466_pinned465', SOURCE)
    V = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(V)
    torch, np, P, E, digest = V.torch, V.np, V.P, V.E, V.digest
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)
    folder = ROOT/'stage/runs/t465_matched_safe_recovery'
    prior = ROOT/'stage/runs/t464_two_failure_recovery'
    p, s, old = read(folder/'protocol.json'), read(folder/'summary.json'), read(prior/'summary.json')
    audit = read(folder/'audit.json')
    receipt_path = ROOT/'stage/docs/long_distance_20261101/release_receipt_T465.json'
    receipt = read(receipt_path)
    assert s['state'] == 'COMPLETE' and audit['state'] == 'PASS'
    assert receipt['state'] == 'FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    stopped_ids = stopped(folder)
    stopped(prior)
    assert s['completed_paired_BC'] == 128 and s['new_PPO'] == s['new_Value'] == 0
    assert s['fresh16'] is None and s['reload78'] is None and not s['six_gate'] and not s['eight_started']
    assert s['own_rows'] == p['previous_own4'] == old['own_rows'] and len(s['own_rows']) == 4
    for row in s['own_rows']:
        assert sha(row['path']) == row['sha256']
    query00 = s['query_rows'][0]
    query21 = next(row for row in old['query_rows'] if row['job_id'] == 'dev_task6_21')
    assert query00['job_id'] == 'dev_task6_00' and query21 == p['previous_valid21'] == s['reused_valid21']
    rows = [query00, query21]
    sources = [SOURCE, Path(__file__), folder/'protocol.json', folder/'summary.json', folder/'audit.json',
               folder/'repair_trace.json', folder/'coverage_rows.json', prior/'protocol.json', prior/'summary.json',
               prior/'audit.json', receipt_path, folder/'SEALED.json', folder/'driver_started.json', folder/'launch.json']
    sources += [Path(row['path']) for row in rows+s['own_rows']]
    for source, expected in p['inputs'].items():
        assert sha(source) == expected, source
    models = dict(input_PPO327=p['learning_initial'], final_T465325=s['proposal'], teacher_T461_context=p['recovery_teacher_model'])
    actors = {}
    for name, model in models.items():
        assert sha(model['path']) == model['sha256']
        actors[name] = V.frozen(p, model['path']).to('cpu').eval().requires_grad_(False)
        assert all(t.device.type == 'cpu' for t in actors[name].parameters())
        assert actors[name].phash() == model['policy_hash']
        sources.append(Path(model['path']))
    safe = E.safe_initial(E.N.science(p), p).to('cpu').eval().requires_grad_(False)
    assert all(t.device.type == 'cpu' for t in safe.parameters())
    thresholds = (.02, .05)

    def errors(actual, target, offset):
        delta = actual.astype(np.float64)-target.astype(np.float64)
        assert delta.ndim == 2 and delta.shape[1] == 2 and len(delta) and np.isfinite(delta).all()
        result = dict(decisions=len(delta), RMSE_per_axis=np.sqrt(np.mean(delta**2, axis=0)).tolist(),
                      max_abs_per_axis=np.max(np.abs(delta), axis=0).tolist(),
                      mean_abs_per_axis=np.mean(np.abs(delta), axis=0).tolist())
        for axis, label in [(0, 'drive'), (1, 'steering')]:
            for threshold in thresholds:
                hits = np.flatnonzero(np.abs(delta[:, axis]) > threshold)
                tag = str(threshold)
                result[label+'_error_gt_'+tag] = dict(count=int(len(hits)),
                    first_global_decision=int(hits[0]+offset) if len(hits) else None,
                    first_region_decision=int(hits[0]) if len(hits) else None,
                    first_global_time_s=float((hits[0]+offset)*.2) if len(hits) else None)
        return result

    results = []
    with torch.inference_mode():
        for row in rows:
            assert row['success'] and not row['collision'] and not row['timeout'] and row['teacher'] and row['full']
            assert sha(row['path']) == row['sha256'] and row['teacher_first'] == 20
            a = E.N.S.arrays(row['path'])
            assert digest(a) == row['array_digest'] and {k:digest(v) for k,v in a.items()} == row['array_hashes']
            for value in a.values():
                if np.issubdtype(value.dtype, np.number):assert np.isfinite(value).all()
            assert not a['end_flags'][:-1].any() and a['end_flags'][-1].tolist() == [0,1,0]
            assert not a['ownership'][:20].any() and a['ownership'][20:].all()
            assert np.array_equal(a['states'][1:], a['next_states'][:-1])
            x = torch.tensor(a['states'], dtype=torch.float64, device='cpu')
            target = np.asarray(a['teacher_actions'])
            assert target.dtype == np.float32 and len(x) > 21 and target.shape == (len(x),2)
            labels = []
            if row['job_id'] == 'dev_task6_00':
                for state in x:labels.append(safe.base_actions(state[None]).float().numpy()[0])
                label_source = 'safeT255 base_actions reset for every actual120 state; no recurrent teacher carry'
            else:
                hidden = None
                for state in x:
                    action, _, hidden, _ = actors['teacher_T461_context'].step(state[None], hidden)
                    labels.append(action.numpy()[0])
                label_source = 'T461 step complete actual teacher-occupancy history from h0; true recurrent carry'
            assert np.array_equal(np.asarray(labels), target)
            locations = {name:q.sequence(x).cpu().numpy() for name,q in actors.items()}
            actions = {name:np.clip(loc,-1,1).astype(np.float32) for name,loc in locations.items()}
            result = dict(job_id=row['job_id'], source_row_SHA256=row['sha256'], source_array_digest=row['array_digest'],
                          decisions=len(x), labels_recomputed_exact=True, label_source=label_source,
                          regions={}, context_teacher_NOT_actual_label_source_for00=True)
            for region, sl in [('prefix_first21',slice(0,21)), ('suffix21onward',slice(21,None)), ('all',slice(None))]:
                offset = 21 if region == 'suffix21onward' else 0
                values = {}
                for name in actors:
                    loc, action, expert = locations[name][sl], actions[name][sl], target[sl]
                    v = errors(action, expert, offset)
                    v['P_action_loss_on_raw_location_mean'] = float(P.action_loss(torch.tensor(loc),torch.tensor(expert,dtype=torch.float64)).mean())
                    v['P_action_loss_on_clipped_float32_action_mean'] = float(P.action_loss(torch.tensor(action,dtype=torch.float64),torch.tensor(expert,dtype=torch.float64)).mean())
                    v['fraction_location_abs_ge1'] = float(np.mean(np.abs(loc)>=1))
                    v['fraction_location_abs_ge1_per_axis'] = np.mean(np.abs(loc)>=1,axis=0).tolist()
                    values[name] = v
                values['final_minus_input_action'] = errors(actions['final_T465325'][sl],actions['input_PPO327'][sl],offset)
                values['final_minus_input_location'] = errors(locations['final_T465325'][sl],locations['input_PPO327'][sl],offset)
                result['regions'][region] = values
            results.append(result)
    trace = read(folder/'repair_trace.json')
    assert len(trace) == 128
    fit = {}
    for key in ['student21','teacher_suffix','reference218','own10']:
        first, last = float(trace[0][key]), float(trace[-1][key])
        fit[key] = dict(first=first,last=last,percent_change=(100*(last-first)/first if first else None),
                        percent_reduction=(100*(first-last)/first if first else None))
    constructor = {}
    for module in list(sys.modules.values())+[V,V.Y,V.X,V.L,V.J,V.H,V.M,V.P,V.E,V.D,V.F,V.R]:
        name = getattr(module,'__file__',None)
        if name:
            path = Path(name).resolve()
            if path.is_file() and path.is_relative_to(ROOT/'stage/experiments'):
                constructor[str(path.relative_to(ROOT))] = sha(path)
    constructor[str(Path(__file__).resolve().relative_to(ROOT))] = sha(__file__)
    elapsed = time.monotonic()-start
    assert elapsed < 300 and not torch.is_grad_enabled()
    report = dict(state='PASS_POSTHOC_CPU_LABEL_AUDIT',native=0,optimizer=0,backward=0,cuda=0,
                  source_rows=results,original_fit_first_last=fit,
                  source_models=models,constructor_file_SHA256=constructor,
                  input_source_SHA256={str(f.relative_to(ROOT)):sha(f) for f in sorted(set(sources))},
                  registered_inputs_verified_count=len(p['inputs']),
                  registered_inputs_mapping_digest=digest(p['inputs']),
                  own4_reused=True,own4_NOT_new_independent_physical_data=True,
                  own4_source_paths=[row['path'] for row in s['own_rows']],
                  fresh16=None,reload78=None,eight_started=False,
                  T465_independent_audit_state=audit['state'],T465_remote_state=receipt['state'],
                  prior_scientific_and_driver_processes_stopped=True,verified_stopped_pids=stopped_ids,
                  elapsed_s=elapsed,max_seconds=300,
                  axis_order=['drive','steering'],indexing='zero_based global decisions; 5Hz control, decision0 time0',
                  regions='prefix includes decisions0..20 (20student executed plus first teacher decision); suffix starts21',
                  meaning='Posthoc fit on two mixed teacher-occupancy histories, not autonomous final-student states, unseen generalization, navigation effectiveness, or causal proof. T461 outputs are context only for00, whose actual labels are safeT255 reset.')
    encoded=(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    assert len(encoded)<=128*1024 and time.monotonic()-start<300
    with OUT.open('xb') as dest:dest.write(encoded)
    signal.alarm(0)
    for result in results:
        parts=[]
        for region in ['prefix_first21','suffix21onward']:
            v=result['regions'][region]
            parts.append(region+':'+str(v['input_PPO327']['P_action_loss_on_raw_location_mean'])+'->'+str(v['final_T465325']['P_action_loss_on_raw_location_mean']))
        print(result['job_id'],' '.join(parts),flush=True)
    print('T466_CPU_POSTHOC_LABEL_AUDIT_PASS',len(encoded),flush=True)


if __name__ == '__main__':main()
