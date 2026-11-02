"""Recover saved T292 metrics only; retain producer failure, no main rerun."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t292_crossfit_value';DOC=ROOT/'stage/docs/distance_20261102'
sys.path.insert(0,str(ROOT/'stage/experiments/t292_crossfit_value'))
import run as original
import core as K
import json,numpy as np,os,signal,time


def clean(x):
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
def write(f,x):f.write_text(json.dumps(x,indent=2,default=clean)+'\n')


def check():
    p=json.loads((HERE/'protocol.json').read_text())
    for f,h in p['input_hashes'].items():assert K.sha(f)==h,f
    for pid in p['ended_pids']:
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise AssertionError(('Live/reused PID',pid))
    original.verify(json.loads((OUT/'protocol.json').read_text()))
    return p


def recover():
    check();assert not (OUT/'summary.json').exists();p=json.loads((OUT/'protocol.json').read_text());items=K.load(p)
    with np.load(OUT/'evidence.npz') as z:
        pred=[z['predictions'][z['offsets'][i]:z['offsets'][i+1]] for i in range(32)]
        loo=[z['time_loo'][z['offsets'][i]:z['offsets'][i+1]] for i in range(32)];metrics=K.measure(p,items,pred,loo,z['gram'])
    raw_status=json.loads((OUT/'status.json').read_text())
    summary=dict(state='FAILED',producer_failure='NumPy bool scalar JSON serialization after all96 gradient and evidence writes; originalrun exit1 retained.',
        scientific_evidence_state='ALL_ARRAYS_PRESENT_METRICS_RECOVERED_WITHOUT_RECOMPUTATION',
        all_workers_stopped=True,physics_steps=0,optimizer_calls=0,Actor_updates=0,analytic_value_fits=8,
        elapsed_lower_bound_s=raw_status['updated_epoch']-raw_status['main_epoch'],original_status_preserved=True,
        recovered_epoch=time.time(),automatic_next_training=False,seven_or_eight_started=False,**metrics)
    write(OUT/'summary.json',summary);write(DOC/'T292_SERIALIZATION_RECOVERY.json',dict(state='METADATA_RECOVERED',physics_added=0,optimizer_calls=0,Actor_updates=0,gradients_recomputed=0,original_run_status_and_error_preserved=True,summary_sha256=K.sha(OUT/'summary.json')))
    print(json.dumps(dict(state='RECOVERED_PRODUCER_FAILURE_PRESERVED',offline_screen_pass=metrics['offline_screen_pass']),default=clean))


def audit():
    check();dest=DOC/'T292_AUDIT.json';assert not dest.exists();p=json.loads((OUT/'protocol.json').read_text());s=json.loads((OUT/'summary.json').read_text());assert s['state']=='FAILED'
    start=time.time()
    def guard(phase):assert time.time()-start<p['audit_wall_seconds']
    def alarm(signum,frame):raise TimeoutError('T292 originally registered independent audit deadline')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(p['audit_wall_seconds'])
    items,pred,fits,loo,gram,hashes,density,metrics=original.compute(p,guard)
    assert hashes==json.loads((OUT/'gradient_hashes.json').read_text()) and density==json.loads((OUT/'density.json').read_text())
    expected=dict(gram=gram,offsets=np.r_[0,np.cumsum([len(v) for v in pred])],predictions=np.concatenate(pred),time_loo=np.concatenate(loo),returns=np.concatenate([i['returns'] for i in items]),fold_seeds=np.array([f['seed'] for f in fits]),mu=np.stack([f['mu'] for f in fits]),sd=np.stack([f['sd'] for f in fits]),coeff=np.stack([f['coeff'] for f in fits]))
    with np.load(OUT/'evidence.npz') as z:assert set(z.files)==set(expected) and all(np.array_equal(z[k],v) for k,v in expected.items())
    assert all(s[k]==v for k,v in metrics.items());signal.alarm(0)
    write(dest,dict(state='PASS_RECOVERED_EVIDENCE_ORIGINAL_PRODUCER_FAILED',physics_added=0,optimizer_calls=0,Actor_updates=0,
        all96_full_parameter_gradient_SHA_recomputed_exact=True,all_Gram_predictions_fits_returns_exact=True,
        all_sources_models_SHA_verified=True,all_original_processes_absent=True,producer_error_and_status_preserved=True,
        original_registered_audit_not_extra_repeat=True,offline_screen_pass=s['offline_screen_pass'],elapsed_s=time.time()-start,summary_sha256=K.sha(OUT/'summary.json')))
    print('T292 recovered evidence independent audit PASS; original serialization failure preserved')


if __name__=='__main__':
    if sys.argv[1:]==['recover']:recover()
    elif sys.argv[1:]==['audit']:audit()
    else:raise SystemExit('recover | audit')
