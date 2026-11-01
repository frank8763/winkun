"""Read back the learned head, Adam, holdout labels/probabilities and metrics."""
from event_model import *
import gzip
import io
import os
import time
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t272_event_value_pilot'
DEST=ROOT/'stage/docs/long_distance_20261101/T272_AUDIT.json'


def main():
    assert not DEST.exists()
    s=json.loads((OUT/'summary.json').read_text());p=json.loads((OUT/'protocol.json').read_text())
    assert s['state']=='COMPLETE' and s['all_workers_stopped'] and s['workers_spawned']==0
    w=json.loads((OUT/'workers.json').read_text())
    for pid in [w['parent'],json.loads((OUT/'launch.json').read_text())['pid']]:
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise AssertionError(('Parent still exists or PID reused',pid))
    for path,h in p['input_hashes'].items():assert sha(path)==h,path
    assert (OUT/'protocol.json').read_bytes()==(HERE/'protocol.json').read_bytes()
    d=dataset(p);payload=torch.load(io.BytesIO(gzip.decompress((OUT/'event_checkpoint.pt.gz').read_bytes())),map_location='cpu',weights_only=False)
    assert np.array_equal(d['mean'],payload['normalization_mean']) and np.array_equal(d['scale'],payload['normalization_scale'])
    q=EventModel();q.load_state_dict(payload['model']);q.eval()
    assert payload['optimizer_calls']==s['optimizer_calls']==p['optimizer_steps']==500
    assert all(int(v['step'])==500 for v in payload['optimizer']['state'].values())
    assert all(torch.isfinite(x).all() for v in payload['optimizer']['state'].values() for x in v.values() if isinstance(x,torch.Tensor))
    with torch.no_grad():pred=q(d['x']).numpy()
    ix=np.flatnonzero(~d['train'])
    with np.load(OUT/'test_predictions.npz') as z:
        assert np.array_equal(z['probabilities'],pred[ix]) and np.array_equal(z['row_indices'],ix)
        assert np.array_equal(z['labels'],d['y'].numpy()[ix]) and np.array_equal(z['valid'],d['mask'].numpy()[ix])
        assert np.array_equal(z['episode_indices'],d['episode'][ix])
    assert metrics(pred,d,'test')==s['report']['final_test'] and metrics(pred,d,'train')==s['report']['final_train']
    baseline=np.broadcast_to(frequency_baseline(d),pred.shape)
    assert metrics(baseline,d,'test')==s['report']['frequency_test']
    torch.manual_seed(p['seed']);initial=EventModel()
    assert any(not torch.equal(initial.state_dict()[k],v) for k,v in payload['model'].items())
    logs=list(map(json.loads,(OUT/'training.jsonl').read_text().splitlines()))
    assert [r['step'] for r in logs]==list(range(50,501,50))
    assert sha(OUT/'event_checkpoint.pt.gz')==s['report']['event_checkpoint_sha256']
    assert sha(OUT/'test_predictions.npz')==s['report']['test_predictions_sha256']
    assert used(OUT)<p['final_bytes'] and used(ROOT/'stage/runs')<p['global_bytes'] and s['elapsed_s']<p['wall_seconds']
    assert s['physics_steps']==s['Actor_updates']==s['navigation_RL_updates']==0
    save(DEST,dict(state='POSTRUN_EVIDENCE_RECONCILED',physics_added=0,optimizer_calls_added=0,
        all_parent_processes_absent=True,event_model_optimizer_calls=500,Actor_updates=0,
        train_only_normalization_exact=True,all_holdout_labels_and_probabilities_reloaded_bit_exact=True,
        train_test_metrics_recomputed_exact=True,Adam_all_parameters_at_step500=True,
        original_Actor_and_all_input_sha_verified=True,predictive_screen_passed=s['report']['predictive_screen_passed'],
        counterfactual_action_ranking_passed=False,six_m_navigation_gate=False,
        own_bytes=used(OUT),global_bytes=used(ROOT/'stage/runs'),summary_sha256=sha(OUT/'summary.json'),
        checkpoint_sha256=sha(OUT/'event_checkpoint.pt.gz'),audit_epoch=time.time()))
    print(json.dumps(dict(state='PASS',screen=s['report']['predictive_screen_passed'])),flush=True)


if __name__=='__main__':main()
