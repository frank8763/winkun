from pathlib import Path
import sys,os,json,time,io,gzip,fcntl,shutil,traceback,multiprocessing as mp
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t242_induced_state_recovery'
sys.path.insert(0,str(ROOT/'stage/experiments/t229_valid_start_acceptance'))
import valid_support as V
S,C,R,np,torch=V.S,V.C,V.R,V.np,V.torch
def worker(pipe,index,deadline):V.OUT=OUT;V.worker(pipe,index,deadline)
def read(path):return torch.load(io.BytesIO(gzip.decompress(Path(path).read_bytes())),map_location='cpu',weights_only=False)['policy']
def main():
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'status.json').exists(),'No duplicateT242'
    plan=json.loads((HERE/'protocol.json').read_text());start=time.monotonic();wall=time.time()+plan['wall_seconds'];physics=commands=0;workers=[];pipes=[];rows=[];states=[];offsets=[0];source=None;reason='COMPLETE';dataset=None
    def receive(p):
        while not p.poll(.5):assert time.time()<wall,'Deadline'
        r=p.recv();assert 'error' not in r,r;return r
    def guard(pending=0):assert time.time()<wall and commands<=plan['max_commands'] and physics<=plan['max_physics'] and S.used(OUT)+pending<plan['final_bytes'] and S.used(ROOT/'stage/runs')+pending<plan['global_bytes'] and shutil.disk_usage(ROOT).free>40*1024**3
    try:
        for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);policies={k:read(p) for k,p in plan['checkpoints'].items()};hashes={k:S.digest(p) for k,p in policies.items()};ps={k:{q:v.numpy().copy() for q,v in p.items()} for k,p in policies.items()}
        ctx=mp.get_context('spawn')
        for i in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(i+1);w=ctx.Process(target=worker,args=(b,i,wall));w.start();b.close();workers.append(w);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';assert all(receive(p)['ready'] for p in pipes);C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[w.pid for w in workers]))
        for off in range(0,212,4):
            batch=plan['jobs'][off:off+4];guard()
            for pipe,j in zip(pipes,batch):pipe.send(('evaluate',dict({k:v for k,v in j.items() if k!='expected'},policy_np=ps[j['arm']],policy_hash=hashes[j['arm']])))
            rr=[receive(p) for p in pipes[:len(batch)]];commands+=len(rr);physics+=sum(i['row']['physics_steps'] for i in rr)
            for j,item in zip(batch,rr):
                r=item['row'];old=j['expected'];a=item['arrays'];assert r['complete'] and r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256'];assert np.array_equal(a['states'][1:],a['next_states'][:-1])
                record=dict(id=j['id'],source_reset_id=j['source_reset_id'],arm=j['arm'],group=j['group'],success=r['success'],collision=r['collision'],states_hash=S.digest(a['states']),original_independent_hash_exact=True,physics_steps=r['physics_steps'],original_trace=old['path'],original_trace_sha256=old['sha256'])
                with (OUT/'reconstruction.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                rows.append(record)
                if j['arm']=='T239':states.append(a['states']);offsets.append(offsets[-1]+len(a['states']))
                else:
                    compact={k:v for k,v in a.items() if k!='next_states'};compact['final_next_state']=a['next_states'][-1];buf=io.BytesIO();np.savez_compressed(buf,**compact);b=buf.getvalue();assert len(b)<192*1024;guard(len(b));path=OUT/'T240_success24_full.npz';path.write_bytes(b);source=dict(path=str(path),sha256=C.sha(path),array_hashes=r['array_hashes'],integration_sha256=r['integration_sha256'])
            C.save(OUT/'status.json',dict(state='EXACT_INDUCED_STATE_RECOVERY',cases=len(rows),physics_steps=physics,commands=commands,optimizer_calls=0,elapsed_s=time.monotonic()-start))
        assert len(states)==211 and source is not None
        buf=io.BytesIO();np.savez_compressed(buf,states=np.concatenate(states),offsets=np.asarray(offsets,dtype=np.int64),success=np.asarray([r['success'] and not r['collision'] for r in rows[:211]]),task=np.asarray([int(r['group'][4:]) for r in rows[:211]],dtype=np.uint8),case_ids=np.asarray([r['source_reset_id'] for r in rows[:211]]));data=buf.getvalue();assert len(data)<plan['max_dataset_bytes'];guard(len(data));path=OUT/'T239_all_induced_states.npz';path.write_bytes(data)
        with np.load(path) as z:assert S.digest(z['states'])==S.digest(np.concatenate(states)) and np.array_equal(z['offsets'],offsets)
        dataset=dict(path=str(path),sha256=C.sha(path),bytes=len(data),states=offsets[-1],cases=211,safe_cases=210,failed_case_ids=[r['source_reset_id'] for r in rows[:211] if r['collision']])
        guard()
    except BaseException:reason='FAILED';C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physics_steps=physics,commands=commands,inflight_physics_upper_bound=4*23334))
    finally:
        for p in pipes:
            try:p.send(None)
            except OSError:pass
        for w in workers:
            w.join(3)
            if w.is_alive():w.kill();w.join(3)
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    result=dict(state=reason,cases=len(rows),original_all10arrays_and_integration_exact=reason=='COMPLETE',dataset=dataset,success_source=source,physics_steps=physics,commands=commands,optimizer_calls=0,all_workers_stopped=all(not w.is_alive() for w in workers),new_independent_navigation_tests=0,elapsed_s=time.monotonic()-start,own_bytes=S.used(OUT),runs_MiB=S.used(ROOT/'stage/runs')/1024**2);C.save(OUT/'summary.json',result);C.save(OUT/'status.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
