"""One fixed construction, shadow and independent reconstruction; no physics."""
from pathlib import Path
import fcntl,gzip,importlib.util,io,json,os,resource,sys,time
import t702_actor as A
torch,np,digest=A.torch,A.np,A.digest
ROOT=A.ROOT;HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t702_projected_global'

def main(mode):
    spec=importlib.util.spec_from_file_location('t702_fit_controller',HERE/'run.py')
    C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
    p=C.read(OUT/'protocol.json');C.verify(p,force=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    C.M.N.N.G._real_affinity(0,{15})
    deadline=min(time.time()+(1200 if mode=='main' else 900),p['absolute_learning_stop'] if mode=='main' else p['absolute_all_stop'])-15
    def guard():
        assert time.time()<deadline,'Fit absolute/action deadline';C.verify(p)
    def emit(who):
        def event(stage,step,kind):
            with (OUT/'fit/fit_calls.jsonl').open('a') as f:f.write(json.dumps(dict(who=who,stage=stage,step=step,kind=kind,epoch=time.time()))+'\n')
        return event
    d=torch.load(io.BytesIO(gzip.decompress((ROOT/'stage/runs/t699_matched_value/DATA.pt.gz').read_bytes())),map_location='cpu',weights_only=False)
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        q,info=A.construct(p,d,guard,emit('live' if mode=='main' else 'independent'))
        if mode=='main':
            shadow,other=A.construct(p,d,guard,emit('shadow'));assert info==other and digest(q.state_dict())==digest(shadow.state_dict());del shadow
            model=A.export(q,p,OUT/'fit/proj8.pt.gz')
        else:
            saved=C.read(OUT/'fit/summary.json')['finals']['proj8'];assert info==saved['construction'];model=saved['model']
        restored=A.frozen(p,model['path']);assert digest(restored.state_dict())==digest(q.state_dict());del restored
        # Full owncarry, canonical batch1, clipped-float32 executed commands.
        A.Old.Existing.verify_commands(p,q,guard)
        parent=A.P.frozen(p,p['initial']['path']);oldbytes=sum(v.numel()*v.element_size() for v in parent.state_dict().values())
        total=sum(v.numel()*v.element_size() for v in q.state_dict().values());assert total-oldbytes==150744
        arrays=A.Old.Existing.L.T.R.E.N.S.arrays(p['protected403_rows'][0]['path'])
        x=torch.tensor(arrays['states'][:64],dtype=torch.float64);q.sample=False;q.rng=None;q.reset_noise();h=None;lat=[]
        with torch.no_grad():
            for i in range(320):
                start=time.perf_counter_ns();_,_,h,_=q.step(x[i%len(x):i%len(x)+1],h)
                if i>=64:lat.append((time.perf_counter_ns()-start)/1e6)
        q.reset_noise()
        metrics=dict(total_tensor_bytes=total,added_tensor_bytes=total-oldbytes,common_parent_bytes=oldbytes,
            complete403plus7_owncarry_command_gate=True,old_decisions=81956,source_decisions=1737,
            batch1_CPU15_median_ms=float(np.median(lat)),batch1_CPU15_p95_ms=float(np.percentile(lat,95)),
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_includes_training_cache_and_Python=True,projection_matrix_and_shared_radius_counted=True)
        if mode=='main':C.save(OUT/'fit/summary.json',dict(state='COMPLETE',finals=dict(proj8=dict(model=model,construction=info,metrics=metrics))))
        else:
            for k,v in saved['metrics'].items():
                if k.startswith(('batch1','peak_')):continue
                assert metrics[k]==v,(k,metrics[k],v)
            C.save(OUT/'fit/audit.json',dict(state='PASS',independent_state_and_construction_exact=True,command_gate=metrics,new_native=0,new_Adam=0,new_RL=0))
        C.verify(p,force=True)
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main(sys.argv[1])
