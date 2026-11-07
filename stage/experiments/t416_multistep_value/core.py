"""Fixed20-hold actual-behavior continuation target; not unbiased optimal-Q.

The one-step IQL pilot is immutable. Only the registered return operator differs;
same split, initialization, minibatches, head counts, and frozen causal features.
"""
from pathlib import Path
import sys,importlib.util
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t416_multistep_value';DOC=ROOT/'stage/docs/day_20261107'
spec=importlib.util.spec_from_file_location('stage415_locked_value_operators',ROOT/'stage/experiments/t415_iql_value/core.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
torch,np,read,save,sha,digest,E=C.torch,C.np,C.read,C.save,C.sha,C.digest,C.E
ORIGINAL_DATA=C.data;C.HERE=HERE;C.OUT=OUT

def window(rewards,discount,nx,n):
    # Each discount already contains the true terminal mask, so no subsequent
    # reward can cross a GOAL/CONTACT/finite80s terminal boundary.
    size=len(rewards);R=torch.zeros_like(rewards);D=torch.ones_like(discount)
    for i in range(size):
        for j in range(i,min(size,i+n)):
            R[i]+=D[i]*rewards[j];D[i]*=discount[j]
        if i+n>=size:assert D[i]==0
    target=torch.stack([nx[min(size-1,i+n-1)] for i in range(size)])
    return R,D,target

def data(p):
    result,checks=ORIGINAL_DATA(p)
    for split in result:
        records=result[split]['episodes']
        for row in records:
            R,D,nx=window(row['r'],row['discount'],row['nx'],p['n_step'])
            row['r']=R;row['discount']=D;row['nx']=nx
        for key in ['x','nx','a','r','discount','returns']:result[split][key]=torch.cat([v[key] for v in records])
    checks.append(dict(operator='20hold_actual_behavior_continuation',n_step=p['n_step'],not_unbiased_offpolicy_optimal_Q=True,changed_feature_or_episode_split=False))
    return result,checks
C.data=data
Values,fit,verify=C.Values,C.fit,C.verify
run,audit=C.run,C.audit
if __name__=='__main__':globals()[sys.argv[1]]()
