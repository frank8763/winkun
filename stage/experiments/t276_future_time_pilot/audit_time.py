from time_model import *
import gzip,io,os,time
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t276_future_time_pilot'
DEST=ROOT/'stage/docs/long_distance_20261101/T276_AUDIT.json'


def main():
    assert not DEST.exists();s=json.loads((OUT/'summary.json').read_text());p=json.loads((OUT/'protocol.json').read_text())
    assert s['state']=='COMPLETE' and s['all_workers_stopped'] and s['native_envs_created']==0
    for pid in {json.loads((OUT/'workers.json').read_text())['parent'],json.loads((OUT/'launch.json').read_text())['pid']}:
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise AssertionError(('Process exists or PID reused',pid))
    for f,h in p['input_hashes'].items():assert sha(f)==h,f
    assert (OUT/'protocol.json').read_bytes()==(HERE/'protocol.json').read_bytes()
    torch.set_num_threads(1);d=load_data(p);payload=torch.load(io.BytesIO(gzip.decompress((OUT/'time_checkpoint.pt.gz').read_bytes())),map_location='cpu',weights_only=False)
    assert np.array_equal(payload['mean'],d['mean']) and np.array_equal(payload['scale'],d['scale']) and payload['ymean']==d['ymean']
    assert payload['optimizer_calls']==s['optimizer_calls']==500 and all(int(v['step'])==500 for v in payload['optimizer']['state'].values())
    q=TimeModel().eval();q.load_state_dict(payload['model'])
    with torch.no_grad():pred=(q(d['x'])+float(d['ymean'])).numpy()
    coef=fit_ridge(d,p);assert np.array_equal(coef,payload['ridge_coefficients']);ridge=design(d['raw'])@coef
    assert metric(pred,d,'test')==s['report']['final_test'] and metric(pred,d,'train')==s['report']['final_train']
    assert metric(ridge,d,'test')==s['report']['ridge_test']
    ix=np.flatnonzero(~d['train'])
    with np.load(OUT/'test_predictions.npz') as z:
        assert np.array_equal(z['prediction'],pred[ix]) and np.array_equal(z['label'],d['y'].numpy()[ix])
        assert np.array_equal(z['row_indices'],ix) and np.array_equal(z['ridge'],ridge[ix]) and np.array_equal(z['episode'],d['episode'][ix])
    for r in s['report']['strata']:
        ee=[i for i,record in enumerate(d['records']) if record['split']=='test' and record['side']==r['side'] and record['distance_m']==r['distance_m']]
        select=np.isin(d['episode'],ee);assert metric(pred,d,'test',select)==r['model'] and metric(ridge,d,'test',select)==r['ridge']
    passed=bool(s['report']['final_test']['log_MAE']<=.9*s['report']['ridge_test']['log_MAE'] and all(r['model']['log_MAE']<=r['ridge']['log_MAE'] for r in s['report']['strata'] if r['distance_m']==6.))
    assert passed==s['report']['predictive_screen_passed']
    assert sha(OUT/'time_checkpoint.pt.gz')==s['report']['checkpoint_sha256'] and sha(OUT/'test_predictions.npz')==s['report']['predictions_sha256']
    assert s['physics_steps']==s['Actor_updates']==s['navigation_RL_updates']==0 and s['elapsed_s']<p['wall_seconds']
    assert used(OUT)<p['final_bytes'] and used(ROOT/'stage/runs')<p['global_bytes']
    save(DEST,dict(state='POSTRUN_EVIDENCE_RECONCILED',all_processes_absent=True,physics_added=0,optimizer_added=0,
        optimizer_calls=500,Actor_updates=0,all4559_holdout_predictions_labels_indices_ridge_reloaded_exact=True,
        all_train_test_strata_metrics_and_ridge_solve_exact=True,train_only_normalization_exact=True,
        all_input_and_original_Actor_sha_verified=True,Adam_all_parameters_step500=True,
        predictive_screen_passed=passed,automatic_Actor_coupling=False,six_m_navigation_gate=False,
        summary_sha256=sha(OUT/'summary.json'),own_bytes=used(OUT),global_bytes=used(ROOT/'stage/runs'),audit_epoch=time.time()))
    print(json.dumps(dict(state='PASS',predictive_screen=passed)),flush=True)


if __name__=='__main__':main()
