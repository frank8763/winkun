from pathlib import Path
import importlib.util
import json
import multiprocessing as mp
import os
import time
import numpy as np
CPU_POOL=[int(x) for x in os.environ.get('STAGE_CPU_POOL',','.join(map(str,sorted(os.sched_getaffinity(0))))).split(',')]
os.environ['STAGE_CPU_POOL']=','.join(map(str,CPU_POOL))
os.sched_setaffinity(0,{int(os.environ.get('STAGE_PROCESS_CPU',str(CPU_POOL[0])))})
import torch
from worker import run

REPO=Path(__file__).resolve().parents[3];OUT=REPO/'stage/runs/t171_reverse_curriculum'
def main():
    assert not (OUT/'preflight.json').exists()
    torch.set_num_threads(1);assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(.35);torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    spec=importlib.util.spec_from_file_location('vector_core',REPO/'stage/current/outputs/stage_t148/td3_core_goal_contrast.py')
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    actor=core.Actor();parent=torch.load(REPO/'stage/current/outputs/stage_t159/consolidated_actor.pt',map_location='cpu',weights_only=False)
    actor.load_state_dict(parent['actor']);actor.eval();gpu=core.Actor().cuda();gpu.load_state_dict(parent['actor']);gpu.eval()
    ctx=mp.get_context('spawn');pipes=[];workers=[]
    try:
        for i in range(2):
            a,b=ctx.Pipe();p=ctx.Process(target=run,args=(b,i,str(OUT/'preflight'),2026103070));os.environ['STAGE_PROCESS_CPU']=str(CPU_POOL[(i+1)%len(CPU_POOL)]);p.start();os.environ['STAGE_PROCESS_CPU']=str(CPU_POOL[0]);b.close();pipes.append(a);workers.append(p)
        for pipe in pipes:pipe.send(('reset',0))
        obs=[];signatures=[]
        for pipe in pipes:
            assert pipe.poll(60);row=pipe.recv();assert 'error' not in row,row;obs.append(row['obs']);signatures.append(row['task']['signature'])
        max_error=0.;frames=0
        for _ in range(8):
            x=torch.from_numpy(np.asarray(obs,dtype=np.float32))
            with torch.no_grad():cpu_actions=actor(x).numpy();actions=gpu(x.cuda()).cpu().numpy()
            max_error=max(max_error,float(np.abs(actions-cpu_actions).max()));assert max_error<1e-5
            for pipe,action in zip(pipes,actions):pipe.send(('step',action,25))
            obs=[]
            for pipe in pipes:
                assert pipe.poll(30);row=pipe.recv();assert 'error' not in row,row;assert row['n']==25 and row['episode'] is None
                assert abs(row['discount']-np.exp(-.05/60))<1e-10;obs.append(row['obs']);frames+=row['n']
        assert len(set(signatures))==2 and frames==400
        report={'state':'PASS','native_physics_steps':frames,'training_updates':0,'workers':2,
            'distinct_initial_tasks':2,'obs_shape':[2,112],'actions_shape':[2,2],'physics_dt_s':.002,
            'control_interval_s':.05,'gpu_device':torch.cuda.get_device_name(),
            'cpu_gpu_actor_max_abs_difference':max_error,'stage_only':True,'phase_inputs':False}
        (OUT/'preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    finally:
        for pipe in pipes:
            try:pipe.send(('close',))
            except OSError:pass
        for p in workers:
            p.join(5)
            if p.is_alive():p.terminate();p.join(5)
        for pipe in pipes:pipe.close()
if __name__=='__main__':main()
