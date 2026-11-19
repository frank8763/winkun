"""Actual data/sampling/gradient zero-step check for adapters-only next plan."""
from pathlib import Path
import copy,json,os,sys
import t652_learning as M
ROOT=M.ROOT;DOC=ROOT/'stage/docs/day_20261119/t652'
sys.path.insert(0,str(ROOT/'stage/experiments/t651_canonical_outcome_pg'))
import t651_runtime as T

def main():
    p=T.read(ROOT/'stage/runs/t651_canonical_outcome_pg/protocol.json')
    assert T.read(ROOT/'stage/runs/t651_canonical_outcome_pg/pg/audit.json')['state']=='PASS'
    T.R.B.setup(p);old,groups=T.data(p,lambda:None)
    assert len(old)==384 and len(groups)==19 and all(len(v)==4 for k,v in groups)
    T.R.gpu_guard();os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    obj=M.make(p);q,opt=obj;before=M.digest(M.state(obj));body=M.L.fixed_hash(q)
    fixture=groups[:2]+[g for g in groups[2:] if len({M.Prior.outcome(row) for a,row in g[1]})>1][:2]
    items,diag=M.Prior.Prior.build(q,fixture,p)
    loss=M.Prior.Prior.pg_objective(q,items,p);loss.backward()
    grads={key:float(v.grad.norm()) for key,v in q.named_parameters() if v.grad is not None}
    assert grads and all(k.startswith(('drive_adapter.','steer_adapter.')) for k in grads)
    assert any(v>0 for v in grads.values()) and all(M.np.isfinite(v) for v in grads.values())
    assert body==M.L.fixed_hash(q) and before==M.digest(M.state(obj)) and not opt.state
    # Mature optimizer fixture tests exactly this scope's moment topology;
    # values are synthetic engineering data, no optimizer.step call.
    original=M.state(obj)
    with M.torch.no_grad():
        for v in q.parameters():
            if v.requires_grad:opt.state[v]=dict(step=M.torch.tensor(3.),exp_avg=M.torch.full_like(v,.25),exp_avg_sq=M.torch.full_like(v,.5))
        mature=M.state(obj);mhash=M.digest(mature)
        q.steer_adapter[-1].bias.add_(.125)
        for v in opt.state.values():v['step'].add_(1);v['exp_avg'].add_(.01)
        M.load(obj,mature);assert M.digest(M.state(obj))==mhash
        M.load(obj,original);assert M.digest(M.state(obj))==before and not opt.state
    q.cpu();del obj,q;M.torch.cuda.empty_cache()
    T.save(DOC/'SCOPE_PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',source='actual T651 correct76 +old384; no fresh independent collection',
       genuine_complete120_history_and_clipped_density_rebuilt=True,gradient_norms=grads,trainable_parameters=11906,
       encoder_GRU_head_no_gradient_and_state_exact=True,initial_actor_and_Adam_unchanged=True,
       mature_synthetic_Adam_fixture_full_rollback_exact=True,optimizer_steps=0,native_steps=0,not_a_launched_protocol=True))
    print('T652_SCOPE_ZERO_NATIVE_ZERO_OPTIMIZER_PASS',grads)

if __name__=='__main__':main()
