"""Check exact deployed per-decision inference on immutable failed-batch data."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t266_step_density_contract'
sys.path.insert(0,str(ROOT/'stage/experiments/t259_closed_loop_recovery'))
from closed_loop_learning import L,S,np,torch,policy


def canonical_mean(p, x):
    """Actual worker step(batch=1) for each observation, preserving autograd."""
    assert not p.memory_on and x.ndim==2 and x.shape[1]==120
    return torch.stack([p.step(row[None])[1][0] for row in x])


def main():
    start=time.monotonic()
    plan=json.loads((HERE/'protocol.json').read_text())
    assert not OUT.exists(), 'No repeat, extra states, or model search'
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('T266 wall limit')))
    signal.alarm(plan['wall_seconds'])
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    def forbid(*args,**kwargs):raise AssertionError('No simulator/worker/optimizer allowed')
    S.native_setup=forbid;L.generate=forbid;L.optimizer=forbid
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for path,h in plan['input_hashes'].items():assert sha(path)==h,path
    rows=list(map(json.loads,Path(plan['training_log']).read_text().splitlines()))
    assert len(rows)==64
    tplan=json.loads(Path(plan['t263_protocol']).read_text())
    p=policy(tplan);payload=L.read(plan['checkpoint']);p.load_state_dict(payload['policy'])
    p.eval();expected=S.digest(p.state_dict());assert expected==plan['policy_hash']
    selected=rows[48:64];assert all(r['policy_hash']==expected for r in selected)
    items=[]
    for r in selected:
        assert sha(r['path'])==r['sha256']
        with np.load(r['path']) as z:arrays={k:z[k].copy() for k in z.files}
        assert all(S.digest(arrays[k])==h for k,h in r['array_hashes'].items())
        items.append(dict(row=r,arrays=arrays))
    data=L.pad([i['arrays'] for i in items]);mask=data['mask']
    means=torch.zeros(*mask.shape,2);old=torch.zeros(*mask.shape,dtype=torch.float64)
    for j,i in enumerate(items):
        n=len(i['arrays']['states']);means[:n,j]=torch.tensor(i['arrays']['means']);old[:n,j]=torch.tensor(i['arrays']['logp'])
    with torch.no_grad():
        _,batchmean,_=p.sequence(data['x'])
        batchlp,_=L.likelihood(p,data)
        direct=canonical_mean(p,data['x'][mask])
        full=torch.zeros_like(means);full[mask]=direct
        directlp=L.AR.logp(direct,data['actions'][mask],full.reshape(-1,2)[data['prev_full']],
            data['actions'].reshape(-1,2)[data['prev_full']],data['reset'])
    batcherror=float((batchmean[mask]-means[mask]).abs().max())
    batchlperror=float((batchlp-old[mask]).abs().max())
    directerror=float((direct-means[mask]).abs().max())
    directlperror=float((directlp-old[mask]).abs().max())
    assert batcherror>=3e-6, 'Must reproduce the original failure, not choose a convenient subset'
    assert directerror<3e-6 and directlperror<.005
    # A real backward pass on a fixed 32-row prefix; no parameter update.
    x=data['x'][mask][:32].clone()
    output=canonical_mean(p,x)
    weight=torch.linspace(-.5,.5,len(x))[:,None].expand_as(output)
    params=[q for q in p.parameters() if q.requires_grad]
    grads=torch.autograd.grad((output*weight).sum(),params,allow_unused=True)
    norm=sum(float(g.double().square().sum()) for g in grads if g is not None)**.5
    assert np.isfinite(norm) and norm>0 and S.digest(p.state_dict())==expected
    result=dict(state='COMPLETE',physics_steps=0,optimizer_calls=0,new_navigation_RL_updates=0,
        failing_round=3,episodes=16,decision_rows=int(mask.sum()),checkpoint_sha256=sha(plan['checkpoint']),
        policy_hash=expected,original_batch_mean_error=batcherror,original_batch_logp_error=batchlperror,
        canonical_step_mean_error=directerror,canonical_step_logp_error=directlperror,
        original_thresholds_preserved=dict(mean=3e-6,conditional_logp=.005),
        canonical_backward_rows=32,canonical_backward_gradient_norm=norm,
        actor_unchanged=True,new_model_exported=False,elapsed_s=time.monotonic()-start,
        scope='Forward and backward contract only; no new learning or full navigation validation.',
        limitations=['Canonical per-sample operations can be slower and need an explicit future learner compute budget.',
            'This fixes a numerical inference-path consistency issue, not the left collision or right timeout mechanism.',
            'No threshold relaxation, distribution centering correction, or recorded-logp substitution was used.'])
    data=(json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode();assert len(data)<plan['max_output_bytes']
    assert sum(q.stat().st_size for q in (ROOT/'stage/runs').rglob('*') if q.is_file())+len(data)<1024**3
    assert shutil.disk_usage(ROOT).free>=40*1024**3
    OUT.mkdir();(OUT/'summary.json').write_bytes(data);signal.alarm(0)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
