"""Fast red-capable regression from immutable original native traces."""
from pathlib import Path
import gzip,json,hashlib,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3]

def source_rows():
    old=[json.loads(v) for v in (ROOT/'stage/runs/t636_joint_best_reproduction/cases.jsonl').read_text().splitlines()]
    new=[json.loads(v) for v in gzip.decompress((ROOT/'stage/runs/t649_fresh_outcome_pg/bc/cases.jsonl.gz').read_bytes()).decode().splitlines()]
    key='wide117_80_dev6_LEFT_18'
    return next(r for r in old if r['job_id']==key and r['role']=='fresh'),next(r for r in new if r['job_id']==key and r['full'])

def check(expected,actual):
    for key in ['scene_sha256','initial_native_integration','task_goal_xy']:
        assert np.array_equal(expected[key],actual[key]),key
    for key in ['states','poses','history_poses','history_goals','history_times']:
        assert np.array_equal(expected[key][0],actual[key][0]),'initial '+key

def main():
    old,new=source_rows()
    assert old['request']['job']==new['request']['job'] and old['policy_hash']==new['policy_hash']
    a=np.load(old['path']);b=np.load(new['path'])
    if len(sys.argv)>1 and sys.argv[1]=='self_check':
        check(a,a)
        try:check(a,b)
        except AssertionError as e:
            print('RED_REPRODUCED_SAME_JOB_POLICY_DIFFERENT_INITIAL',str(e));return
        raise AssertionError('Regression lost its red signal')
    check(a,b)

if __name__=='__main__':main()
