"""Read-only admission check for a strict coordinate input subspace.

No SVD tolerance, rank inference, new gradient, network modification or optimizer.
This tests only normalized raw-input coordinates identically zero on every old
history and whether the six fixed new success histories activate any of them.
It cannot reject a general nonlinear or other exact-subspace solution.
"""
from pathlib import Path
import hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[3]
DOC=ROOT/'stage/docs/weekend_20261120_23'
sys.path.insert(0,str(ROOT/'stage/experiments/t690_score_weighted_baseline'))
import t690_learning as L
sys.path.insert(0,str(ROOT/'stage/experiments/t693_measure_valued_actor'))
import t693_wait as Wait


def main():
    started=time.time();os.sched_setaffinity(0,{14});Wait.Raw.resources()
    out=DOC/'EXACT_ZERO_COORDINATE_PREPARATION.json'
    assert not out.exists(),'Do not repeat this fixed preparation'
    def forbidden(*args,**kwargs):raise AssertionError('No new optimizer/native stepping')
    L.torch.optim.Adam.step=forbidden
    if 'mujoco' in sys.modules:sys.modules['mujoco'].mj_step=forbidden
    p=L.read(ROOT/'stage/runs/t691_induced_state_rehearsal/protocol.json')
    q=L.M.L.frozen(p,p['initial']['path']);before=L.digest(q.state_dict())
    rows=[r for r in L.read(p['reference_contract'])['reference_rows'] if r['origin']=='T501_original_success']
    assert len(rows)==len({r['task_id'] for r in rows})==384
    assert {r['task_id'] for r in rows}==set(p['before_success_ids'])
    zeros=L.np.ones(120,dtype=bool);old=[];new=[]
    with L.torch.no_grad():
        for row in rows:
            assert time.time()<started+120
            assert L.sha(row['source_path'])==row['sha256']
            a=L.T.R.E.N.S.arrays(row['source_path'])
            assert not a['ownership'].any() and not a['stochastic'].any()
            assert a['end_flags'][-1,1] and not a['end_flags'][:-1].any()
            x=q.normalized(L.torch.tensor(a['states'])).numpy()
            zeros&=(x==0).all(axis=0)
            old.append(dict(id=row['task_id'],sha256=row['sha256'],decisions=len(x)))
        indices=L.np.flatnonzero(zeros);active=set()
        for row in p['novel_rows']:
            assert time.time()<started+120 and L.sha(row['path'])==row['sha256']
            a=L.T.R.E.N.S.arrays(row['path']);assert row['safe'] and not a['stochastic'].any()
            x=q.normalized(L.torch.tensor(a['states'])).numpy()
            present=[int(i) for i in indices if (x[:,i]!=0).any()]
            active.update(present);new.append(dict(id=row['id'],sha256=row['sha256'],old_zero_coordinates_nonzero=present))
    assert L.digest(q.state_dict())==before
    result=dict(state='READ_ONLY_COORDINATE_SUPPORT_PRESENT' if active else 'NO_SUPPORT_FOR_THIS_EXACT_ZERO_COORDINATE_OPERATOR',
                old384=old,new_fixed6=new,normalized_raw_zero_indices=indices.tolist(),
                augmented_feature_indices=[64+int(i) for i in indices],new_activated_indices=sorted(active),
                no_hidden64_or_general_nullspace_rank_claim=True,no_global_capacity_or_impossibility_claim=True,
                new_native=0,new_optimizer=0,new_Actor=0,new_Value=0,new_gradient=0,no_model_modified=True,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),elapsed_seconds=time.time()-started,
                charged_upper_seconds=20,finite_read_only_preparation_not_registered_science=True)
    assert result['elapsed_seconds']<20
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['old384','new_fixed6']}))


if __name__=='__main__':main()
