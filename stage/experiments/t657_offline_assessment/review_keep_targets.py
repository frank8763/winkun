"""Read-only per-task weighting and identical-input target consistency audit."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t656_qualified_recovery_pg'))
import t656_runtime as T
R,np=T.R,T.np

def main():
    p=T.read(ROOT/'stage/runs/t656_qualified_recovery_pg/protocol.json');original=T.read(ROOT/'stage/runs/t646_offline_night/OFFLINE_INPUTS.json');by={r['id']:r for r in original['members']};v=[]
    for row in p['field_keep_rows']:
        key=row['job_id']
        if key not in by:continue
        old=by[key];assert T.sha(row['path'])==row['sha256'] and T.sha(old['offline_path'])==old['sha256']
        a=R.E.N.S.arrays(row['path']);b=R.E.N.S.arrays(old['offline_path']);same=np.array_equal(a['states'][0],b['states'][0])
        assert np.array_equal(a['initial_native_integration'],b['initial_native_integration']) and np.array_equal(a['task_goal_xy'],b['task_goal_xy'])
        ca=np.clip(a['locations'][0],-1,1);cb=np.clip(b['locations'][0],-1,1)
        v.append(dict(id=key,identical_initial120=bool(same),both_initial_hidden_zero=bool(not a['hidden_before'][0].any() and not b['hidden_before'][0].any()),
            old_initial_command_target=cb.tolist(),new_initial_command_target=ca.tolist(),absolute_difference=np.abs(ca-cb).tolist(),
            inconsistent_exact_initial_command_targets=bool(same and np.any(ca!=cb)),old_sha256=old['sha256'],new_sha256=row['sha256']))
    d=dict(state='READ_ONLY_ZERO_NATIVE_ZERO_OPTIMIZER',old_histories=384,field_histories=8,old_weight=p['keep_weight'],field_weight=p['new_keep_weight'],
        old_per_history_weight=p['keep_weight']/384,field_per_history_weight=p['new_keep_weight']/8,
        relative_field_to_old_single_history_weight=(p['new_keep_weight']/8)/(p['keep_weight']/384),
        overlapping_task_count=len(v),initial_command_target_conflicts=sum(x['inconsistent_exact_initial_command_targets'] for x in v),overlaps=v,
        limits='Action-target inconsistency is not physical task incompatibility; weights alone do not establish causal source of old02 regression; both references can succeed on one task with different commands.',new_native=0,new_optimizer=0)
    T.save(ROOT/'stage/docs/day_20261119/t656/KEEP_TARGET_REVIEW.json',d);print(json.dumps(d,ensure_ascii=False))
if __name__=='__main__':main()
