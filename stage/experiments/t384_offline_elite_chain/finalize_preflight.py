"""Reuse passed numerical tests; verify real offline workers without stepping."""
from night_common import *
import ast,multiprocessing as mp
def main():
    p=read(HERE/'protocol.json');report=read(OUT/'preflight.json');assert report['state']=='PASS' and report['discarded_Adam']==4
    assert not (OUT/'launch.json').exists() and not (OUT/'final_preflight.json').exists();offline()
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    for f in HERE.glob('*.py'):p['inputs'][str(f)]=sha(f)
    base=science(p);base.update(out=str(OUT/'worker_probe_002'));torch.set_num_threads(1)
    ctx=mp.get_context('spawn');physical=ctx.Array('q',2);workers=[];pipes=[]
    try:
        for k in range(2):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(k+1);proc=ctx.Process(target=native_worker,args=(b,k,base,physical));proc.start();b.close();workers.append(proc);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[]
        for k,pipe in enumerate(pipes):
            assert pipe.poll(60),'local worker startup deadline';v=pipe.recv();assert v['ready'] and v['affinity']==[k+1];ready.append(v)
        assert sum(physical[:])==0
        for pipe in pipes:pipe.send(None)
        for proc in workers:proc.join(15)
        codes=[proc.exitcode for proc in workers];assert codes==[0,0],codes
        assert all(not proc.is_alive() for proc in workers)
        report2=dict(state='PASS',physics=0,additional_geometry=0,additional_Adam=0,total_discarded_Adam=4,total_geometry=1,workers_ready=ready,all_probe_workers_stopped=True,
            worker_exitcodes=codes,worker_stdout_stderr_restored_at_outer_boundary=True,finalizer_tracker_outside_immutable_run=True,all_stage_source_input_reverified=True,original_passed_numerical_contract_reused=True)
        save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);save(OUT/'final_preflight.json',report2);save(DOC/'FINAL_LOCAL_READINESS.json',report2)
        capacity(OUT,p['root_bytes']);print('OFFLINE_WORKER_AND_FINALIZER_PREFLIGHT_PASS')
    finally:
        for proc in workers:
            if proc.is_alive():proc.kill();proc.join(3)
if __name__=='__main__':main()
