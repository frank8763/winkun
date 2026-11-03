"""One finite imperfect-demonstration residual-head initialization pilot.

The two human trajectories timed out. They are action examples, not successful
expert rollouts or current-policy samples. No reward/Q relabeling or RL update.
"""
from pathlib import Path
import sys,os,json,io,gzip,copy,time,shutil,traceback,signal,contextlib,importlib.util
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t316_frozen_native_validation'))
import t316_core as U
np,torch,K=U.np,U.torch,U.K
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t319_imperfect_demo_head';DOC=ROOT/'stage/docs/distance_20261103'
read=U.read

def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    assert K.S.used(OUT)<=p['final_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']

def demo(path):
    with np.load(path) as z:return z['states'].copy(),z['actions'].copy()

@torch.no_grad()
def features(q,x):
    rows=[];raw=[];modes=[]
    for v in x:
        a=torch.tensor(v[None],dtype=torch.float32)
        h,_=q.core(q.encoder(a)[None],a.new_zeros((1,1,32)))
        mode,_,_,r=q.step(a)
        rows.append(torch.cat([h[0,0],a.new_ones(1)]).double().numpy())
        raw.append(r[0].double().numpy());modes.append(mode[0].double().numpy())
    return np.stack(rows),np.stack(raw),np.stack(modes)

def anchors(p):
    with np.load(p['retention_states']) as z:
        xx=z['states'];off=z['offsets'];task=z['task']
        groups=[np.concatenate([xx[off[k]:off[k+1]] for k in range(211) if task[k]==t]) for t in range(8)]
    ix=[np.linspace(0,len(g)-1,p['anchor_rows_each'],dtype=np.int64) for g in groups]
    return np.concatenate([g[k] for g,k in zip(groups,ix)]),[k.tolist() for k in ix]

def science(p,guard):
    q=K.policy(p).eval().requires_grad_(False);before=K.S.digest(q.state_dict())
    assert before==p['policy_hash'] and not q.memory_on
    frozen={k:K.S.digest(v) for k,v in q.state_dict().items() if not k.startswith('residual.')}
    x,y=demo(p['train_demo']);vx,vy=demo(p['validation_demo']);old,indices=anchors(p)
    guard();phi,base,mode=features(q,x);vphi,vbase,vmode=features(q,vx);ophi,obase,omode=features(q,old)
    assert phi.shape==(280,33) and vphi.shape==(280,33) and ophi.shape==(256,33)
    # Exact quadratic optimum in float64, no advantage/return/Q estimation.
    matrix=phi.T@phi/len(phi)+p['retention_weight']*ophi.T@ophi/len(ophi)+p['ridge']*np.eye(33)
    rhs=phi.T@(y.astype(np.float64)-base)/len(phi)
    guard();delta=np.linalg.solve(matrix,rhs);residual=float(np.linalg.norm(matrix@delta-rhs)/max(1.,np.linalg.norm(rhs)))
    assert np.isfinite(delta).all() and residual<=1e-10
    # Update only the existing 66 scalar output-head parameters, no graph change.
    with torch.no_grad():
        q.residual.weight.add_(torch.tensor(delta[:-1].T,dtype=torch.float32))
        q.residual.bias.add_(torch.tensor(delta[-1],dtype=torch.float32))
    assert frozen=={k:K.S.digest(v) for k,v in q.state_dict().items() if not k.startswith('residual.')}
    metrics={}
    for role,xx,yy,p0,b0,m0 in [('train',x,y,phi,base,mode),('validation',vx,vy,vphi,vbase,vmode),('old',old,omode,ophi,obase,omode)]:
        guard();_,b1,m1=features(q,xx)
        cast_error=float(np.max(np.abs(b1-(b0+p0@delta))))
        assert cast_error<=3e-6
        metrics[role]=dict(rows=len(xx),mode_MSE_before=float(np.mean((m0-yy)**2)),mode_MSE_after=float(np.mean((m1-yy)**2)),
            raw_MSE_before=float(np.mean((b0-yy)**2)),raw_MSE_after=float(np.mean((b1-yy)**2)),
            max_mode_change=float(np.max(np.abs(m1-m0))),float32_head_vs_float64_prediction_max_error=cast_error)
    report=dict(train_demo_success=False,validation_demo_success=False,train_rows=280,validation_rows=280,
        train_demo_sha256=K.C.sha(p['train_demo']),validation_demo_sha256=K.C.sha(p['validation_demo']),
        feature_sha256=K.S.digest(phi),old_feature_sha256=K.S.digest(ophi),matrix_sha256=K.S.digest(matrix),rhs_sha256=K.S.digest(rhs),
        delta_sha256=K.S.digest(delta),solve_relative_residual=residual,condition_number=float(np.linalg.cond(matrix)),
        fixed_old_anchor_indices=indices,changed_parameters=66,closed_form_parameter_proposals=1,Adam_calls=0,RL_updates=0,
        parent_policy_hash=before,policy_hash=K.S.digest(q.state_dict()),frozen_state_hashes=frozen,metrics=metrics,
        validation_is_second_same_scene_human_attempt_not_unseen_navigation=True)
    return q,report

def patch(q,p):
    payload=dict(residual=q.residual.state_dict(),policy_hash=K.S.digest(q.state_dict()),parent_policy_hash=p['policy_hash'],
        parent_checkpoint_sha256=p['checkpoint_sha256'],optimizer=None,closed_form_initialization=True,successful_demo=False)
    b=io.BytesIO();torch.save(payload,b);blob=gzip.compress(b.getvalue(),mtime=0);assert len(blob)<=p['patch_bytes'];return blob

def worker(pipe,index,p,physical):
    sys.path.insert(0,str(ROOT/'stage/experiments/t312_online_goal_ppo'))
    import ppo_worker as original
    original.OUT=OUT;original.B.OUT=OUT;torch.optim.Adam.step=U.forbid_step
    return original.worker(pipe,index,p,physical)

def outcome(r):return U.outcome(r)

def counts(rows):
    return {role:dict(executed=sum(r['role']==role for r in rows),outcomes={kind:sum(r['role']==role and outcome(r)==kind for r in rows) for kind in ['GOAL','CONTACT','TIMEOUT','INVALID','CENSORED']}) for role in ['old_guard','six_centers']}
