"""Fixed-cap response characterization of ended candidates; zero optim/native."""
from pathlib import Path
import os,sys,time,json
import t648_learning as M
sys.path.insert(0,str(M.ROOT/'stage/experiments/t646_offline_night'))
import t646_runtime as T
ROOT=M.ROOT;DOC=ROOT/'stage/docs/day_20261119/t648'

def main():
    p=T.read(ROOT/'stage/experiments/t646_offline_night/protocol.json')
    p.update(response_peak=.01,response_rms=.002,response_batch=16)
    # Lock engineering tolerances BEFORE evaluating old proposals. They are
    # fractions of normalized command half-range, NOT closed-loop guarantees.
    limits=dict(response_peak=.01,response_rms=.002,response_batch=16,selection='fixed 1 percent peak and .2 percent RMS normalized command half-range, not tuned to old candidate outcomes',new_native=0,new_optimizer=0)
    f=DOC/'RESPONSE_LIMITS.json'
    if f.exists():assert T.read(f)==limits
    else:T.save(f,limits)
    started=time.time();T.R.B.setup(p)
    data=T.load(p,lambda:None);assert len(data)==384
    T.R.gpu_guard();os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    results=[]
    models=[('original',p['initial']['path'])]+[(f'full_command_{i}',str(ROOT/f'stage/runs/t646_offline_night/full_command/proposal_{i:02d}.pt.gz')) for i in range(4)]
    for name,path in models:
        q=M.L.frozen(p,path).cuda();begin=time.time()
        with M.torch.no_grad():metrics=M.responses(q,data,p,lambda:None)
        result=dict(name=name,path=path,model_sha256=T.sha(path),full384=metrics,feasible=M.response_gate(metrics,p),seconds=time.time()-begin)
        results.append(result);q.cpu();del q;M.torch.cuda.empty_cache();print(name,result['feasible'],max(r['peak_command_error'] for r in metrics),result['seconds'],flush=True)
    assert results[0]['feasible'] and max(r['peak_command_error'] for r in results[0]['full384'])<1e-10
    assert not any(r['feasible'] for r in results[1:]),'Fixed caps failed to discriminate these known regressing candidates; reconsider method rather than tune silently'
    T.save(DOC/'RESPONSE_CALIBRATION.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',limits=limits,models=results,seconds=time.time()-started,
        meaning='Known regressing candidates violate fixed response caps; does not show caps ensure closed-loop retention'))

if __name__=='__main__':main()
