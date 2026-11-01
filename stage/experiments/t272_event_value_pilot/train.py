"""One bounded offline event-model fit. Actor/native navigation remain untouched."""
from event_model import *
import gzip
import io
import os
import shutil
import subprocess
import time
import traceback
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t272_event_value_pilot'


def main():
    assert not (OUT/'status.json').exists(),'Never restart an ended run'
    p=json.loads((HERE/'protocol.json').read_text());start=time.time();calls=0;reason='COMPLETE';report={}
    def guard(training=True):
        assert time.time()-start<(p['train_wall_seconds'] if training else p['wall_seconds']),'WALL'
        assert used(OUT)<p['final_bytes'] and used(ROOT/'stage/runs')<p['global_bytes'],'STORAGE'
        assert shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    def status(state):save(OUT/'status.json',dict(state=state,pid=os.getpid(),updated_epoch=time.time(),optimizer_calls=calls,Actor_updates=0,physics_steps=0))
    def predict(q,d):
        with torch.no_grad():return q(d['x']).numpy()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    q=opt=d=None
    try:
        for path,h in p['input_hashes'].items():assert sha(path)==h,path
        assert json.loads((OUT/'preflight.json').read_text())['state']=='PASS'
        save(OUT/'workers.json',dict(parent=os.getpid(),workers=[],workers_spawned=0,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
        status('BUILDING_EXISTING_DATA');guard();d=dataset(p)
        torch.manual_seed(p['seed']);rng=np.random.default_rng(p['seed']);q=EventModel();opt=torch.optim.Adam(q.parameters(),lr=p['learning_rate'])
        baseline=frequency_baseline(d);base_pred=np.broadcast_to(baseline,(len(d['x']),6,3))
        # Initial predictions are descriptive only; never used for model selection.
        initial=predict(q,d);report['initial_test']=metrics(initial,d,'test')
        report['frequency_train']=metrics(base_pred,d,'train');report['frequency_test']=metrics(base_pred,d,'test')
        for step in range(p['optimizer_steps']):
            guard();selected=rng.integers(len(d['train_ids']),size=p['batch_size'])
            ids=np.array([d['train_ids'][i][rng.integers(len(d['train_ids'][i]))] for i in selected])
            z=q(d['x'][ids]);mask=d['mask'][ids];y=d['y'][ids]
            ll=-(y*z.clamp_min(1e-8).log()).sum(-1)
            loss=((ll*mask).sum(-1)/mask.sum(-1).clamp_min(1)).mean()
            assert torch.isfinite(loss);opt.zero_grad();loss.backward()
            gn=float(torch.nn.utils.clip_grad_norm_(q.parameters(),5.,error_if_nonfinite=True));opt.step();calls+=1
            if calls%50==0:
                row=dict(step=calls,loss=float(loss.detach()),gradient_norm_before_clip=gn,elapsed_s=time.time()-start)
                with (OUT/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                status('FITTING_EVENT_MODEL')
        guard(False);pred=predict(q,d)
        assert np.max(np.abs(pred.sum(-1)-1))<5e-7 and np.min(np.diff(pred[:,:,:2],axis=1))>=-1e-7
        report['final_train']=metrics(pred,d,'train');report['final_test']=metrics(pred,d,'test')
        a=report['final_test'];b=report['frequency_test']
        predictive_gate=bool(a['mean_Brier_total']<=.9*b['mean_Brier_total'] and
            a['mean_Brier_goal']<=b['mean_Brier_goal'] and a['mean_Brier_contact']<=b['mean_Brier_contact'] and
            max(a['five_bin_goal_contact_ECE'])<=.10)
        ix=np.flatnonzero(~d['train']);buf=io.BytesIO()
        np.savez_compressed(buf,probabilities=pred[ix],labels=d['y'].numpy()[ix].astype(np.uint8),
            valid=d['mask'].numpy()[ix].astype(np.uint8),row_indices=ix.astype(np.int32),episode_indices=d['episode'][ix])
        bbytes=buf.getvalue();assert len(bbytes)<=p['evaluation_bytes'];(OUT/'test_predictions.npz').write_bytes(bbytes)
        report.update(predictive_screen_passed=predictive_gate,test_episodes=16,train_episodes=80,
            original_Actor_sha_unchanged=sha(p['Actor_checkpoint'])==p['input_hashes'][p['Actor_checkpoint']],
            test_predictions_sha256=sha(OUT/'test_predictions.npz'),data_rows=len(d['x']),
            no_goal_relabeling_or_future_input=True,original_goal_context_retained=True,
            counterfactual_action_ranking_passed=False,automatic_Actor_coupling=False,six_m_navigation_gate=False)
    except BaseException:
        reason='FAILED';save(OUT/'failure.json',dict(traceback=traceback.format_exc(),optimizer_calls=calls,physics_steps=0,Actor_updates=0))
    if q is not None:
        buf=io.BytesIO();torch.save(dict(model=q.state_dict(),optimizer=opt.state_dict() if opt else None,
            normalization_mean=d['mean'],normalization_scale=d['scale'],optimizer_calls=calls,
            architecture='122-64tanh-64tanh-18 logits -> six competing-event intervals',
            original_Actor_not_in_this_checkpoint=True),buf)
        blob=gzip.compress(buf.getvalue(),compresslevel=6,mtime=0);assert len(blob)<=p['checkpoint_bytes']
        (OUT/'event_checkpoint.pt.gz').write_bytes(blob);report['event_checkpoint_sha256']=sha(OUT/'event_checkpoint.pt.gz')
    result=dict(state=reason,physics_steps=0,Actor_updates=0,navigation_RL_updates=0,
        optimizer_calls=calls,event_model_fit_only=True,all_workers_stopped=True,workers_spawned=0,
        elapsed_s=time.time()-start,own_bytes=used(OUT),global_bytes=used(ROOT/'stage/runs'),
        protocol_sha256=sha(HERE/'protocol.json'),report=report,
        limitation='Logged mixed-behavior original-command event prediction; no calibrated value for a changed policy, causal action ranking, new navigation success or safety guarantee.')
    assert time.time()-start<p['wall_seconds']+5
    save(OUT/'summary.json',result);save(OUT/'status.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
