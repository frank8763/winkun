"""Recover T380 serialization evidence without repeating its scientific fits."""
from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OLD=ROOT/'stage/runs/t380_teacher_model_audit'
OUT=ROOT/'stage/runs/t380_evidence_recovery'
DOC=ROOT/'stage/docs/narrow_20261105'
spec=importlib.util.spec_from_file_location('stage380_immutable_producer',ROOT/'stage/experiments/t380_teacher_model_audit/run.py')
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)


def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['inputs'].items():assert S.sha(f)==h,f
    assert time.time()<p['original_all_work_deadline_epoch']-5
    assert S.used(OLD)+S.used(OUT)<=256*1024
    S.budget(S.read(OLD/'protocol.json'))


def compact(predictions):
    arrays={k:v for k,v in predictions.items() if not k.endswith('_target')}
    references={}
    original=S.read(OLD/'protocol.json')
    for k,v in predictions.items():
        if not k.endswith('_target'):continue
        label,horizon,_=k.split('_');dataset={'A':'human364','B':'human361','C':'learner379'}[label]
        path=original['datasets'][dataset];starts=predictions[label+'_'+horizon+'_starts']
        with np.load(path) as z:restored=z['next_poses'][starts+int(horizon)-1]
        assert np.array_equal(restored,v)
        references[k]=dict(source=path,source_sha256=S.sha(path),array='next_poses',
              indices_key=label+'_'+horizon+'_starts',offset=int(horizon)-1,target_array_sha256=S.array_sha(v),shape=list(v.shape),dtype=str(v.dtype))
    return arrays,references


def prepare():
    assert not OUT.exists()
    assert S.read(OLD/'summary.json')['state']=='FAILED'
    assert "149992" in S.read(OLD/'failure.json')['error'] and (OLD/'models.npz').exists()
    for pid in [S.read(OLD/'summary.json')['pid'],S.read(OLD/'launch.json')['pid']]:S.absent(pid)
    with np.load(OLD/'models.npz') as z:assert len(z.files)==9
    inputs=[OLD/'summary.json',OLD/'failure.json',OLD/'models.npz',OLD/'protocol.json',
            ROOT/'stage/experiments/t380_teacher_model_audit/run.py',DOC/'T380_RECOVERY_PLAN.md',*HERE.glob('*.py')]
    p=dict(id='T380_EVIDENCE_RECOVERY',inputs={str(f):S.sha(f) for f in sorted(inputs)},
           original_all_work_deadline_epoch=S.read(OLD/'launch.json')['all_work_deadline_epoch'],
           original_and_recovery_combined_cap=256*1024,producer_fits_repeated=0,independent_reserved_fits=3,
           native=0,optimizer=0,original_producer_remains_failed=True)
    assert time.time()+420<p['original_all_work_deadline_epoch']
    OUT.mkdir();S.save(HERE/'protocol.json',p,8*1024);S.save(OUT/'protocol.json',p,8*1024)
    print('REGISTERED_ORIGINAL_BUDGET_EVIDENCE_RECOVERY')


def launch():
    p=S.read(HERE/'protocol.json');verify(p);assert not (OUT/'launch.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=S.read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    with (OUT/'main.log').open('xb') as log:
        proc=subprocess.Popen(['timeout','--signal=TERM','--kill-after=1s','300s','taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'recover'],
             cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    S.save(OUT/'launch.json',dict(pid=proc.pid,source_remote_commit=receipt['remote_commit'],root_tree=receipt['root_tree'],
           original_all_work_deadline_epoch=p['original_all_work_deadline_epoch']),4*1024)
    print(json.dumps(dict(state='RECOVERY_LAUNCHED',pid=proc.pid)))


def recover():
    p=S.read(HERE/'protocol.json');verify(p);assert not (OUT/'summary.json').exists()
    start=time.time();S.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[]),4*1024)
    with np.load(OLD/'models.npz') as z:saved={k:z[k] for k in z.files}
    calls=[]
    def frozen_models(data):
        label='ABC'[len(calls)];calls.append(label)
        x=np.concatenate([a['X'][a['valid']] for a in data]);mean=x.mean(0);scale=np.maximum(x.std(0),1e-6);mean[0]=0.;scale[0]=1.
        model={k:saved[label+'_'+k] for k in ['mean','scale','coefficient']}
        assert np.array_equal(model['mean'],mean) and np.array_equal(model['scale'],scale)
        z=(x-mean)/scale
        description=dict(transitions=len(x),feature_singular_values=np.linalg.svd(z,compute_uv=False).tolist(),
             standardized_rank=int(np.linalg.matrix_rank(z)),action_min=x[:,8:10].min(0).tolist(),
             action_max=x[:,8:10].max(0).tolist(),action_std=x[:,8:10].std(0).tolist())
        return model,description
    S.fit=frozen_models
    report,models,predictions=S.compute(S.read(OLD/'protocol.json'));assert calls==list('ABC')
    assert all(np.array_equal(models[k],v) for k,v in saved.items())
    arrays,references=compact(predictions)
    S.persist_arrays(OUT/'predictions.npz',arrays,128*1024);S.save(OUT/'target_references.json',references,8*1024)
    S.save(OUT/'results.json',report)
    summary=dict(state='COMPLETE',pid=os.getpid(),elapsed_s=time.time()-start,native_commands=0,physics_steps=0,optimizer_calls=0,
         producer_fits_repeated=0,saved_original_models_used=3,original_producer_remains_failed=True,
         prediction_filter_passed=report['prediction_filter_passed'],MPPI_executed=False,all_workers_stopped=True,
         results_sha256=S.sha(OUT/'results.json'),predictions_sha256=S.sha(OUT/'predictions.npz'),
         target_references_sha256=S.sha(OUT/'target_references.json'))
    S.save(OUT/'summary.json',summary,4*1024);verify(p);print(json.dumps(summary))


def audit():
    p=S.read(HERE/'protocol.json');verify(p);assert not (OUT/'audit.json').exists()
    summary=S.read(OUT/'summary.json');S.absent(summary['pid']);S.absent(S.read(OUT/'launch.json')['pid'])
    report,models,predictions=S.compute(S.read(OLD/'protocol.json'))
    assert report==S.read(OUT/'results.json') and S.sha(OUT/'results.json')==summary['results_sha256']
    with np.load(OLD/'models.npz') as z:assert set(z.files)==set(models) and all(np.array_equal(z[k],v) for k,v in models.items())
    arrays,references=compact(predictions);assert references==S.read(OUT/'target_references.json')
    with np.load(OUT/'predictions.npz') as z:assert set(z.files)==set(arrays) and all(np.array_equal(z[k],v,equal_nan=True) and S.array_sha(z[k])==S.array_sha(v) for k,v in arrays.items())
    result=dict(state='PASS',native=0,optimizer=0,original_reserved_independent_fits=3,producer_fits_repeated=0,
         original3models_and_all_predictions_metrics_gates_exact=True,all_original_target_bytes_recoverable_from_SHA_bound_inputs=True,
         original_producer_remains_failed=True,combined_logical_bytes=S.used(OLD)+S.used(OUT),combined_cap=256*1024)
    S.save(OUT/'audit.json',result,8*1024);S.save(DOC/'T380_RECOVERY_AUDIT.json',result,8*1024);verify(p);print(json.dumps(result))


if __name__=='__main__':
    assert sys.argv[1] in ['prepare','launch','recover','audit'];globals()[sys.argv[1]]()
