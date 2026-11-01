from time_model import *
import ast,shutil
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t276_future_time_pilot'


def main():
    assert not OUT.exists() and not (HERE/'protocol.json').exists()
    source=ROOT/'stage/runs/t275_future_target_contract'
    assert json.loads((source/'summary.json').read_text())['state']=='COMPLETE'
    assert json.loads((ROOT/'stage/docs/long_distance_20261101/T275_AUDIT.json').read_text())['state']=='POSTRUN_EVIDENCE_RECONCILED'
    inputs=[source/'protocol.json',source/'summary.json',source/'future_pairs.npz',
        ROOT/'stage/experiments/t275_future_target_contract/future_data.py',ROOT/'stage/experiments/t275_future_target_contract/contract.py',
        ROOT/'stage/docs/long_distance_20261101/T275_AUDIT.json',ROOT/'stage/runs/t267_causal_credit_course/final_checkpoint.pt.gz',
        ROOT/'stage/runs/t255_overnight/s2_reset_final_checkpoint.pt.gz',*HERE.glob('*.py')]
    p=dict(experiment='T276',scope='One offline actual future-endpoint-time relation learner, not Actor RL',
        source_protocol=str(source/'protocol.json'),pairs_path=str(source/'future_pairs.npz'),input_hashes={str(f):sha(f) for f in inputs},
        Actor_checkpoint=str(ROOT/'stage/runs/t267_causal_credit_course/final_checkpoint.pt.gz'),
        seed=2026111601,optimizer_steps=500,batch_size=128,learning_rate=.001,ridge_lambda=.001,
        train_wall_seconds=210,wall_seconds=300,checkpoint_bytes=512*1024,evaluation_bytes=512*1024,
        metadata_bytes=1024**2,final_bytes=2*1024**2,global_bytes=1024**3,min_free_bytes=40*1024**3,
        predictive_screen='Test episode-balanced log-time MAE <=.90 of fixed12-term geometry/motion/action/clock ridge, both6m side log-MAE no worse than ridge. No Actor coupling regardless.',
        contract='FixedT27526395 realized noncontact future endpoint pairs from same96 actual trajectories (21836train4559family-heldout), not independent episodes or new6m goals reached.111input=107 source-verified stripped direct-goal state+2currentactualactions+2body-relativefuturequerygoal/10. No actualdelta-time, future sensors/absoluteposition/case/seed/policyID input. Originalcapclock and physical/motionhistory retained, NOT statistical goal independence or reconstructed120 Actor. Label log1p positive recorded physical seconds, NOT first-hit/safereachability/shortesttime/metric theorem or changed-policyQ. MLP111-64tanh-64tanh-1, train-onlyepisode-balanced mean/stdmin.001/clip10 and labelmean, MSE, uniformepisode thenpair sampling, seed2026111601/500Adam lr.001/batch128/clip5. Fixed constantmean and12-termgeometry-motion-action-clock ridge lambda.001 fit onlytrain, no grid or bestcheckpoint. Test initial/final only, stratify16episodes/4side-distance groups. No native/Actor/BC/newnavigation, no planner/waypoint/teacher/router/Phase. Ifscreenfails retainnegative and no repeats; ifpasses still no Actor control/safety claim. StorefullAdam/holdoutpredictions, auditreadback and completed-onlybackup.')
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    assert used(ROOT/'stage/runs')+p['final_bytes']<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    print(json.dumps(dict(state='REGISTERED',reserve_bytes=p['final_bytes'])),flush=True)


if __name__=='__main__':main()
