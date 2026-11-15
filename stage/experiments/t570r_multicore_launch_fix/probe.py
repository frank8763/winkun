"""Frozen real-history CPU concurrency check; no simulator, optimizer or CUDA."""
from pathlib import Path
import os,sys,time,json,hashlib,multiprocessing as mp,traceback,shutil,subprocess
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t570r_multicore_launch_fix';DOC=ROOT/'stage/docs/day_20261115/t570r'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def used(p):return sum(f.stat().st_size for f in Path(p).rglob('*') if f.is_file() and not f.is_symlink())
def imports():
    sys.path.insert(0,str(ROOT/'stage/experiments/t562_full_history_representation'));import t562_actor as L
    return L

def worker(pipe,index,p):
    os.sched_setaffinity(0,{14+index});L=imports();t=L.torch;t.set_num_threads(1);t.set_num_interop_threads(1);t.use_deterministic_algorithms(True)
    cfg=read(p['actor_configuration']);cfg.update(p['model_config']);q=L.frozen(cfg,p['model']['path']);q.encoder.requires_grad_(True);q.core.requires_grad_(True);q.student.requires_grad_(True)
    states=[t.tensor(L.G.E.N.S.arrays(r['path'])['states']) for r in p['rows']];before=L.digest(q.state_dict());assert before==p['actor_state_hash'] and not t.cuda.is_initialized()
    pipe.send(dict(ready=True,pid=os.getpid(),cpu=14+index))
    try:
        while True:
            job=pipe.recv()
            if job is None:break
            k=job;started=time.perf_counter();q.zero_grad(set_to_none=True);pred=q.sequence(states[k]);loss=pred.square().mean();loss.backward();grads={n:v.grad for n,v in q.named_parameters() if v.requires_grad}
            item=dict(index=k,seconds=time.perf_counter()-started,loss=float(loss.detach()),prediction_hash=L.digest(pred.detach()),gradient_hash=L.digest(grads),model_unchanged=L.digest(q.state_dict())==before,new_optimizer=0,new_native=0)
            assert item['model_unchanged'] and not t.cuda.is_initialized();pipe.send(item)
    except BaseException:pipe.send(dict(error=traceback.format_exc()));raise
    finally:pipe.close()
def prepare():
    assert not OUT.exists();assert time.time()+480<1794735000
    b=read(ROOT/'stage/runs/t569_frozen_complete_panel/protocol.json');L=imports();L.torch.set_num_threads(1);cfg=read(b['actor_configuration']);cfg.update(b);q=L.frozen(cfg,b['model']['path']);assert q.phash()==b['model']['policy_hash']
    rows=b['imported_rows'][:2];assert len(rows)==2
    p=dict(id='T570R_MULTICORE_FROZEN_HISTORY_ENGINEERING',actor_configuration=b['actor_configuration'],model_config=b,model=b['model'],rows=rows,actor_state_hash=L.digest(q.state_dict()),source_sha256=sha(__file__),max_seconds=240,all_stop=1794735000,max_bytes=256*1024,new_native=0,new_optimizer=0,new_RL=0,gradient_evaluations_main=24,CPU=[14,15],CPUQuota='200%',GPU=False,scope='Frozen saved real histories serial vs parallel same jobs;CPU forward/backward only;not native throughput/learning benefit or CUDA benchmark.')
    p['inputs']={str(Path(r['path'])):sha(r['path']) for r in rows};p['inputs'][p['model']['path']]=p['model']['sha256'];p['inputs'][p['actor_configuration']]=sha(p['actor_configuration'])
    assert used(ROOT/'stage/runs')+p['max_bytes']+131072<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    OUT.mkdir();save(OUT/'protocol.json',p);save(DOC/'BUDGET.json',dict(new_native=0,new_optimizer=0,new_RL=0,gradient_evaluations=24,main_seconds=240,own_bytes=p['max_bytes'],global_worst=used(ROOT/'stage/runs')-used(OUT)+p['max_bytes']+131072,CPUQuota='200%',GPU=False));print('T570R_REGISTERED_ZERO_NATIVE_ZERO_OPTIMIZER')
def main():
    p=read(OUT/'protocol.json');assert not (OUT/'launch.json').exists();assert sha(__file__)==p['source_sha256'];assert all(sha(f)==h for f,h in p['inputs'].items());assert time.time()+p['max_seconds']+180<p['all_stop'];assert used(ROOT/'stage/runs')-used(OUT)+p['max_bytes']+131072<2**30
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and set(os.sched_getaffinity(0))<={14,15}
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    started=time.time();save(OUT/'launch.json',dict(epoch=started,pid=os.getpid(),source_remote=receipt['remote_commit'],CPUQuota='200%',CPU=[14,15],GPU=False,new_native=0,new_optimizer=0));children=[];pipes=[];result=None;error=None;ctx=mp.get_context('spawn');events=[]
    def receive(i):
        assert pipes[i].poll(max(.01,min(60,started+p['max_seconds']-time.time()))),'worker timeout';x=pipes[i].recv();assert 'error' not in x,x;return x
    try:
        for i in range(2):
            a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p));c.start();b.close();children.append(c);pipes.append(a)
        ready=[receive(i) for i in range(2)];save(OUT/'workers.json',ready)
        # Warm both real jobs on both CPUs; four discarded gradient computations.
        for k in range(2):
            for i in range(2):pipes[i].send(k)
            for i in range(2):events.append(dict(mode='warmup',worker=i,**receive(i)))
        reference={k:next(x for x in events if x['index']==k) for k in range(2)};serial=[];parallel=[]
        for repeat in range(5):
            start=time.perf_counter()
            for k in range(2):pipes[0].send(k);events.append(dict(mode='serial',repeat=repeat,worker=0,**receive(0)))
            serial.append(time.perf_counter()-start);start=time.perf_counter()
            for i in range(2):pipes[i].send(i)
            for i in range(2):events.append(dict(mode='parallel',repeat=repeat,worker=i,**receive(i)))
            parallel.append(time.perf_counter()-start)
        for e in events:
            ref=reference[e['index']];assert all(e[k]==ref[k] for k in ['prediction_hash','gradient_hash','loss'])
        import statistics
        result=dict(serial_pair_seconds=serial,parallel_pair_seconds=parallel,median_speedup=statistics.median(serial)/statistics.median(parallel),prediction_and_gradient_same_jobs_bitwise_equal=True,model_unchanged=True,gradient_evaluations=len(events),new_native=0,new_optimizer=0,new_RL=0,CUDA_initialized=False,not_native_or_GPU_benchmark=True)
        assert len(events)==24
    except BaseException:error=traceback.format_exc()
    finally:
        for a in pipes:
            try:a.send(None)
            except (BrokenPipeError,EOFError):pass
        for c in children:
            c.join(timeout=5)
            if c.is_alive():c.terminate();c.join(timeout=2)
            if c.is_alive():c.kill();c.join()
        save(OUT/'events.json',events);save(OUT/'summary.json',dict(state='COMPLETE' if error is None else 'FAILED',error=error,result=result,elapsed_s=time.time()-started,physics=0,commands=len(children)+2,all_workers_stopped=all(not c.is_alive() for c in children)))
    assert error is None,error
    # Independent persisted-result consistency check; no new gradients.
    s=read(OUT/'summary.json');ev=read(OUT/'events.json');assert len(ev)==24 and s['all_workers_stopped'];assert all(sha(f)==h for f,h in p['inputs'].items());save(OUT/'audit.json',dict(state='PASS',scope='saved hashes/counts/constant checkpoint;not independent gradient recomputation',new_native=0,new_optimizer=0))
    files=[f for f in OUT.rglob('*') if f.is_file() and f.name not in ['main.log','local_manifest.json']];save(OUT/'local_manifest.json',dict(state='SEALED_LOCAL_ALL_MEMBER_SHA',files=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in files]));assert used(OUT)<=p['max_bytes'];print(json.dumps(s,ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
