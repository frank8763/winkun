from time_model import *
import gzip,io,os,shutil,subprocess,time,traceback
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t276_future_time_pilot'


def main():
    assert not (OUT/'status.json').exists(),'No replay/continuation/grid'
    p=json.loads((HERE/'protocol.json').read_text());start=time.time();calls=0;state='COMPLETE';report={};q=opt=d=None
    def guard(training=True):
        assert time.time()-start<(p['train_wall_seconds'] if training else p['wall_seconds'])
        assert used(OUT)<p['final_bytes'] and used(ROOT/'stage/runs')<p['global_bytes']
        assert shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    def status(label):save(OUT/'status.json',dict(state=label,pid=os.getpid(),optimizer_calls=calls,physics_steps=0,Actor_updates=0,updated_epoch=time.time()))
    try:
        for f,h in p['input_hashes'].items():assert sha(f)==h,f
        assert json.loads((OUT/'preflight.json').read_text())['state']=='PASS'
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        save(OUT/'workers.json',dict(parent=os.getpid(),workers=[],native_envs_created=0,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
        status('LOADING_EXISTING_FUTURE_PAIRS');guard();d=load_data(p);coef=fit_ridge(d,p)
        ridge=design(d['raw'])@coef;constant=np.full(len(d['y']),d['ymean'],dtype=np.float64)
        torch.manual_seed(p['seed']);rng=np.random.default_rng(p['seed']);q=TimeModel();opt=torch.optim.Adam(q.parameters(),lr=p['learning_rate'])
        def predict():
            with torch.no_grad():return (q(d['x'])+float(d['ymean'])).numpy()
        report['untrained_test']=metric(predict(),d,'test');report['ridge_test']=metric(ridge,d,'test');report['constant_test']=metric(constant,d,'test')
        for k in range(p['optimizer_steps']):
            guard();chosen=rng.integers(len(d['train_ids']),size=p['batch_size'])
            ix=np.array([d['train_ids'][i][rng.integers(len(d['train_ids'][i]))] for i in chosen])
            pred=q(d['x'][ix])+float(d['ymean']);loss=(pred-d['y'][ix]).square().mean();assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(q.parameters(),5,error_if_nonfinite=True));opt.step();calls+=1
            if calls%50==0:
                with (OUT/'training.jsonl').open('a') as f:f.write(json.dumps(dict(step=calls,loss=float(loss.detach()),gradient_norm=gn,elapsed_s=time.time()-start))+'\n')
                status('FITTING_FUTURE_TIME')
        guard(False);pred=predict();report['final_train']=metric(pred,d,'train');report['final_test']=metric(pred,d,'test');report['ridge_train']=metric(ridge,d,'train')
        strata=[]
        for dist in [5.25,6.]:
            for side in ['LEFT','RIGHT']:
                ee=[i for i,r in enumerate(d['records']) if r['split']=='test' and r['distance_m']==dist and r['side']==side]
                select=np.isin(d['episode'],ee)
                strata.append(dict(distance_m=dist,side=side,episodes=len(ee),model=metric(pred,d,'test',select),ridge=metric(ridge,d,'test',select)))
        passed=bool(report['final_test']['log_MAE']<=.9*report['ridge_test']['log_MAE'] and
            all(r['model']['log_MAE']<=r['ridge']['log_MAE'] for r in strata if r['distance_m']==6.))
        ix=np.flatnonzero(~d['train']);buf=io.BytesIO();np.savez_compressed(buf,row_indices=ix.astype(np.int32),
            prediction=pred[ix],label=d['y'].numpy()[ix],ridge=ridge[ix],episode=d['episode'][ix])
        blob=buf.getvalue();assert len(blob)<=p['evaluation_bytes'];(OUT/'test_predictions.npz').write_bytes(blob)
        report.update(strata=strata,predictive_screen_passed=passed,training_pairs=int(d['train'].sum()),test_pairs=len(ix),
            predictions_sha256=sha(OUT/'test_predictions.npz'),ridge_coefficients=coef.tolist(),
            Actor_unchanged=sha(p['Actor_checkpoint'])==p['input_hashes'][p['Actor_checkpoint']],
            counterfactual_action_ranking_passed=False,automatic_Actor_coupling=False,six_m_navigation_gate=False)
    except BaseException:
        state='FAILED';save(OUT/'failure.json',dict(traceback=traceback.format_exc(),optimizer_calls=calls,physics_steps=0))
    if q is not None:
        buf=io.BytesIO();torch.save(dict(model=q.state_dict(),optimizer=opt.state_dict(),mean=d['mean'],scale=d['scale'],
            ymean=d['ymean'],optimizer_calls=calls,ridge_coefficients=coef,input_width=111,
            label='log1p recorded positive future endpoint time, not first-hit/minimum-time/probability/Q'),buf)
        blob=gzip.compress(buf.getvalue(),mtime=0);assert len(blob)<=p['checkpoint_bytes'];(OUT/'time_checkpoint.pt.gz').write_bytes(blob)
        report['checkpoint_sha256']=sha(OUT/'time_checkpoint.pt.gz')
    result=dict(state=state,physics_steps=0,optimizer_calls=calls,Actor_updates=0,navigation_RL_updates=0,
        all_workers_stopped=True,native_envs_created=0,elapsed_s=time.time()-start,own_bytes=used(OUT),global_bytes=used(ROOT/'stage/runs'),
        report=report,limitation='Positive observed endpoint timing under historical behaviors. No unseen-goal reachability, control ranking, shortest path or six-metre success learned/verified.')
    save(OUT/'summary.json',result);save(OUT/'status.json',result)
    print(json.dumps(dict(state=state,optimizer_calls=calls,report=report)),flush=True)


if __name__=='__main__':main()
