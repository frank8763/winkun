"""Register zero-step architecture contracts; this does not register training."""
from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t209_temporal_actor'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not OUT.exists(),'No duplicate architecture preflight'
 plan={'experiment':'T209_ARCHITECTURE_CONTRACT_ONLY','parent':str(ROOT/'stage/runs/t194_drive_sensitivity/selected_policy.pt'),'causal_source':str(ROOT/'stage/runs/t205_temporal_intervention/traces/source.npz'),'retention_targets':str(ROOT/'stage/runs/t196_left_exploration/retention_targets.npz'),'max_wall_seconds':90,'max_bytes':1024**2,'physics_steps':0,'optimizer_updates':0,'new_online_training_authorized_by_this_protocol':False,
 'contract':'Model and mathematical gradient preflight only. Zero-initialized neural residual preserves parent at initialization. Existing113obs+previousnativeaction2+currentlymeasuredsteer2+previousdecision_dt1 =>118causalpacket. Native2msphysics and100stepdecision .2s. StreamGRU atdecisionrate, episode-start reset only; memoryoff control identical parametercount butresetseachdecision. CNNhandlesordered48rays. AllActorparameters trainableexceptunusedfixedlogstd; separatefresh recurrentVbaseline, noQgradient. Auxiliaryprogress/contactheads definedbutnotconnectedtoActor ortraining. No runtimeplanner/waypoints/map/teacher/filter. No onlinepolicyupdate ornavigation rollout in thisprotocol.',
 'gates':'Initial parentactionequality on actualretentionstates; nonzero native-density gradients reach parentfeaturelayers; zero-residualhead intentionallyblocks newcoregradient initially, nonzero synthetichead copyproves future endtoendpath without optimizerstep; streaming/sequence agreement; future-observation perturbation cannot change previousoutputs; explicit episode reset removes earlierhistory; memoryon olderprefix can affectaction andmemoryoff cannot, withsamecurrentpacket. These prove onlyimplementationproperties, notGRUbenefit/GOAL/learningconvergence.',
 'evidence':['https://arxiv.org/html/2210.04839','https://proceedings.mlr.press/v162/ni22a.html','https://arxiv.org/html/2012.04406','https://github.com/twni2016/pomdp-baselines']}
 files=[Path(plan[k]) for k in ['parent','causal_source','retention_targets']]+list(HERE.glob('*.py'))+[ROOT/'stage/experiments/t203_native_exploration/runtime.py',ROOT/'stage/experiments/t200_residual_placement/common.py'];plan['input_hashes']={str(p):sha(p) for p in files}
 assert sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file())+plan['max_bytes']<1024**3 and shutil.disk_usage(ROOT).free>40*1024**3
 OUT.mkdir()
 for p in [HERE/'protocol.json',OUT/'protocol.json']:p.write_text(json.dumps(plan,indent=2)+'\n')
 print(json.dumps({'state':'ARCHITECTURE_PREFLIGHT_REGISTERED','protocol_sha256':sha(HERE/'protocol.json'),'physics_steps':0,'optimizer_updates':0}))
if __name__=='__main__':main()
