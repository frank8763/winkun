"""One bounded Stage curvature proposal audit, independent of ended T211."""
from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t212_curvature_direction'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not OUT.exists(),'No duplicate T212'
    s=json.loads((ROOT/'stage/runs/t211_guarded_progress/summary.json').read_text());assert s['all_workers_stopped']
    old=json.loads((ROOT/'stage/experiments/t211_guarded_progress/protocol.json').read_text())
    rows=[json.loads(x) for x in (ROOT/'stage/runs/t211_guarded_progress/rollouts.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['arm']=='original' and r['round']==1 and r['role']=='training'];assert len(rows)==4
    plan={'experiment':'T212','seed':2026103212,'wall_seconds':90,'final_bytes':4*1024**2,'peak_bytes':5*1024**2,
          'source_policy':rows[0]['weights'],'parent_policy':old['checkpoint'],'retention_targets':old['retention_targets'],
          'rows':rows,'old_uniform_per_task':16,'old_worst_KL':8,'old_worst_MSE':8,'new_metric_states':96,
          'configs':[{'beta':.1,'damping':.1},{'beta':1.,'damping':.1}],
          'cg_max_iterations':20,'cg_relative_tolerance':.15,'step_L2':[.02,.01,.005,.0025,.00125,.000625],
          'max_behavior_mean_KL':.01,'max_parent_anchor_KL':.01,'max_balanced_anchor_MSE':.00025,
          'minimum_objective_improvement':1e-8,'finite_difference_epsilon':1e-5}
    plan['contract']=('Zero simulator/optimizer calls; archived originalT211round1 MC-minus-zero standardizedbaseline, '
       'not reconstruction of previousV/Adam. All574210effectiveActorparameters in curvaturemodel float64. '
       'Centered squared native-mean changes normalized by fixedsigma ->positive-semidefiniteGaussNewton, NOT exacttruncatedGaussianFisher. '
       'Metric uses balanced16statesperoldtask+8worstcurrentparentKL+8worstcurrentMSE,unique;new96uniformstates. '
       'M=GNold+beta*GNnew+.1I, twofixedbeta. Matrixfree Hessianvectorproducts ofcenteredsquare, CG20,residual<=.15required. '
       'Check symmetry relative<=1e-6,positivequadratic,centralfinitegradientrelative<=.005 beforeproposals. '
       'Compareordinarygradient and2GNdirections on6fixedL2norms,allactualfloat32Actor completebatch/3023anchors exactnativeTruncKL/MSE/surrogatechecks. '
       'Largestregularizedsurrogateimprovementamongpassingcandidates, tiesfixedcandidateorder, mayexportONEfullofflinecandidate<=3MiB. '
       'No originalsceneGOAL/old7safety/trainingbenefitclaim;finiteoldanchorboundnotclosedloopguarantee. '
       'No planner,waypoint,map,teacherswitch,Phase research,sceneedit, onlinelearning or automaticevaluation. '
       '90s/4MiBfinal5MiBpeak/globalnewruns1GiB/free40GiB,no duplicate/resume/extension. Failurepartialpreserved;T159/T194unchanged.')
    plan['sources']=['https://arxiv.org/html/1705.10528','https://proceedings.mlr.press/v37/schulman15.html',
       'https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/trpo/trpo.py']
    files=list(HERE.glob('*.py'))+[Path(plan[k]) for k in ('source_policy','parent_policy','retention_targets')]+[Path(r['trace']) for r in rows]+[
        ROOT/'stage/experiments/t203_native_exploration/runtime.py',ROOT/'stage/experiments/t200_residual_placement/common.py',ROOT/'stage/experiments/native_update_guard.py',
        ROOT/'stage/runs/t211_guarded_progress/summary.json']
    plan['input_hashes']={str(p):sha(p) for p in files}
    used=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file())
    assert used+plan['peak_bytes']<1024**3 and shutil.disk_usage(ROOT).free>40*1024**3
    OUT.mkdir()
    for p in (HERE/'protocol.json',OUT/'protocol.json'):p.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'state':'REGISTERED','physics_steps':0,'wall_seconds':90,'protocol_sha256':sha(HERE/'protocol.json')}))
if __name__=='__main__':main()
