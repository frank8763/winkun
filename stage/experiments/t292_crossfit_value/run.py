from pathlib import Path
import core as K
import builtins,fcntl,gzip,io,json,os,shutil,signal,symtable,sys,time,traceback
import numpy as np
import torch
ROOT=K.ROOT;HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t292_crossfit_value';DOC=ROOT/'stage/docs/distance_20261102'


def used(path):return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def verify(p):
    for f,h in p['input_hashes'].items():assert K.sha(f)==h,f
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    assert used(ROOT/'stage/runs')<p['global_bytes'] and used(OUT)<p['own_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']


def prohibit(*args,**kw):raise RuntimeError('T292 forbids native simulation and optimizer steps')
def setup():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.optim.Adam.step=prohibit;torch.optim.SGD.step=prohibit;K.B.S.native_setup=prohibit


def compute(p,progress):
    setup();verify(p);items=K.load(p);pred,fits=K.fit_predict(items,p['ridge'],progress);loo=K.time_loo(items)
    gram,hashes,density=K.exact_gradients(p,items,pred,loo,progress);measures=K.measure(p,items,pred,loo,gram)
    return items,pred,fits,loo,gram,hashes,density,measures


def preflight():
    p=json.loads((HERE/'protocol.json').read_text());assert not any((OUT/f).exists() for f in ['status.json','preflight.json','launch.json'])
    verify(p);setup();items=K.load(p);assert all(i['row']['complete'] and not i['row']['error'] and i['row']['aux_eta']==.1 and i['row']['policy_hash']==p['policy_hash'] for i in items)
    for mod in [sys.modules[__name__],K]:
        tree=symtable.symtable(Path(mod.__file__).read_text(),mod.__file__,'exec');missing=set()
        def walk(t):
            for s in t.get_symbols():
                if s.is_referenced() and s.is_global() and s.get_name() not in vars(mod) and not hasattr(builtins,s.get_name()):missing.add(s.get_name())
            for c in t.get_children():walk(c)
        walk(tree);assert not missing,(mod.__name__,missing)
    assert len({(i['row']['seed'],i['row']['source_reset_id']) for i in items})==32
    save(OUT/'preflight.json',dict(state='PASS',physics_steps=0,optimizer_calls=0,Actor_updates=0,whole_family_splits_verified=True,all32_source_arrays_verified=True,synthetic=K.synthetic(),time_credit_identity=K.B.enumeration_contract(),source_sha256=K.sha(HERE/'protocol.json')))
    print('T292 preflight PASS zero native/optimizer/Actor updates')


def main():
    assert not (OUT/'status.json').exists();p=json.loads((HERE/'protocol.json').read_text());start=time.time();lock=(ROOT/'stage/runs/.stage_native_exclusive.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    outside=used(ROOT/'stage/runs')-used(OUT);assert outside+p['own_bytes']<p['global_bytes']
    def progress(phase):
        assert time.time()<start+p['wall_seconds'];verify(p);assert used(ROOT/'stage/runs')-used(OUT)==outside
        save(OUT/'status.json',dict(state=phase,pid=os.getpid(),main_epoch=start,updated_epoch=time.time(),absolute_deadline_epoch=start+p['wall_seconds'],physics_steps=0,optimizer_calls=0,Actor_updates=0))
    def alarm(signum,frame):raise TimeoutError('T292 fixed wall budget')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(p['wall_seconds']);state='COMPLETE';result={}
    try:
        save(OUT/'workers.json',dict(parent=os.getpid(),workers=[]));progress('STARTED')
        items,pred,fits,loo,gram,hashes,density,result=compute(p,progress)
        arrays=dict(gram=gram,offsets=np.r_[0,np.cumsum([len(v) for v in pred])],predictions=np.concatenate(pred),time_loo=np.concatenate(loo),returns=np.concatenate([i['returns'] for i in items]),
                    fold_seeds=np.array([f['seed'] for f in fits]),mu=np.stack([f['mu'] for f in fits]),sd=np.stack([f['sd'] for f in fits]),coeff=np.stack([f['coeff'] for f in fits]))
        buf=io.BytesIO();np.savez_compressed(buf,**arrays);b=buf.getvalue();assert len(b)<p['payload_bytes'];(OUT/'evidence.npz').write_bytes(b)
        save(OUT/'gradient_hashes.json',hashes);save(OUT/'density.json',density);save(OUT/'split_manifest.json',[{k:v for k,v in f.items() if k in ['seed','train_ids','test_ids']} for f in fits]);verify(p)
    except BaseException:
        state='FAILED';save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physics_steps=0,optimizer_calls=0))
    signal.alarm(0);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    summary=dict(state=state,physics_steps=0,optimizer_calls=0,Actor_updates=0,native_environments_created=0,all_workers_stopped=True,analytic_value_fits=8 if state=='COMPLETE' else None,
        elapsed_s=time.time()-start,own_bytes=used(OUT),global_bytes=used(ROOT/'stage/runs'),automatic_next_training=False,seven_or_eight_started=False,
        limitations='32 consumed parent-policy trajectories, eight paired noise families on four fixed reset tasks. Crossfitted nonlinear ridge state baseline only, not a new neural Critic or actor training. Exact finite-sample score-gradient Gram before clipping/PCGrad/Adam; not true population variance, convergence, universal unbiasedness, new navigation success or algorithm novelty.',**result)
    save(OUT/'summary.json',summary);save(OUT/'status.json',summary);print(json.dumps(summary),flush=True)


def audit():
    dest=DOC/'T292_AUDIT.json';assert not dest.exists();p=json.loads((OUT/'protocol.json').read_text());s=json.loads((OUT/'summary.json').read_text());assert s['state']=='COMPLETE'
    for pid in [json.loads((OUT/'workers.json').read_text())['parent'],json.loads((OUT/'launch.json').read_text())['pid']]:
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise AssertionError(('Live/reused PID',pid))
    start=time.time()
    def guard(phase):assert time.time()-start<p['audit_wall_seconds']
    def alarm(signum,frame):raise TimeoutError('T292 independent audit wall budget')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(p['audit_wall_seconds'])
    items,pred,fits,loo,gram,hashes,density,metrics=compute(p,guard)
    assert hashes==json.loads((OUT/'gradient_hashes.json').read_text()) and density==json.loads((OUT/'density.json').read_text())
    with np.load(OUT/'evidence.npz') as z:
        expected=dict(gram=gram,offsets=np.r_[0,np.cumsum([len(v) for v in pred])],predictions=np.concatenate(pred),time_loo=np.concatenate(loo),returns=np.concatenate([i['returns'] for i in items]),fold_seeds=np.array([f['seed'] for f in fits]),mu=np.stack([f['mu'] for f in fits]),sd=np.stack([f['sd'] for f in fits]),coeff=np.stack([f['coeff'] for f in fits]))
        assert set(z.files)==set(expected) and all(np.array_equal(z[k],v) for k,v in expected.items())
    assert all(s[k]==v for k,v in metrics.items());signal.alarm(0)
    save(dest,dict(state='PASS',physics_added=0,optimizer_calls=0,Actor_updates=0,all96_full_parameter_gradients_SHA_recomputed_exact=True,all_Gram_predictions_fits_returns_splits_exact=True,all_sources_models_SHA_verified=True,all_original_processes_absent=True,offline_screen_pass=s['offline_screen_pass'],elapsed_s=time.time()-start,summary_sha256=K.sha(OUT/'summary.json')))
    print('T292 independent audit PASS: all96 full gradients and32 trajectories exact')


if __name__=='__main__':
    if sys.argv[1:]==['--preflight']:preflight()
    elif sys.argv[1:]==['--audit']:audit()
    else:assert not sys.argv[1:];main()
