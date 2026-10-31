"""Independent bounded solver test after T212 CG residual gate rejected both GN directions."""
from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t214_preconditioned_curvature'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not OUT.exists(),'No duplicate T214'
    prior=json.loads((ROOT/'stage/runs/t212_curvature_direction/summary.json').read_text())
    assert prior['state']=='COMPLETE' and all(not x['converged'] for x in prior['CG_solves'])
    plan=json.loads((ROOT/'stage/experiments/t212_curvature_direction/protocol.json').read_text())
    plan.update(experiment='T214',seed=2026103214,wall_seconds=90,cg_max_iterations=80,
        configs=[{'beta':1.,'damping':.1,'preconditioned':False},{'beta':1.,'damping':.1,'preconditioned':True},{'beta':1.,'damping':1.,'preconditioned':True}])
    plan['contract']=('IndependentpostT212solveraudit,notrestart:zero simulator/optimizer/onlinelearning. Samearchivedbatch/basepolicy/121old+96newmetricstatesandfull574210parameters. '
        'ComputeexactdiagonalofcenteredmeanGaussNewtonvia2scalaroutputgradientspermetricstate;JacobiPCGpositive diagonal+damping. '
        'Fixedthreeconfigs:CG80damping.1,PCG80damping.1,PCG80damping1,beta1all. Sameoperatorcontractsymmetry/positive/finite-differencegate. '
        'Residual<=.15required fornewdirections; CG/PCGsameoperator is solvercomparison, dampingcomparisonseparate. '
        'Ordinarygradient andvalidnewdirections fixed6L2sizes,actualfloat32fullbatch/3023anchors strictsameKL/MSE/improvementchecks;max24candidates,oneofflinecandidateexportonly. '
        'Not exacttruncatedGaussianFisher/fullCPO/TRPO/onlineDRL/navigationorstatisticalbenefitproof. '
        '90s/4MiBfinal5MiBpeak/globalnewruns1GiB/free40GiB,no duplicate,resume,infiniteCGorautomatictraining. '
        'No maps/planner/waypoints/runtimeTeacher/sceneedit/Phase research;T159/T194unchanged.')
    plan['sources']=['https://arxiv.org/html/1705.10528','https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/trpo/trpo.py',
        'https://web.stanford.edu/class/ee364b/lectures/conj_grad_slides.pdf']
    files=list(HERE.glob('*.py'))+[Path(plan[k]) for k in ('source_policy','parent_policy','retention_targets')]+[Path(r['trace']) for r in plan['rows']]+[
        ROOT/'stage/experiments/t203_native_exploration/runtime.py',ROOT/'stage/experiments/t200_residual_placement/common.py',ROOT/'stage/experiments/native_update_guard.py',ROOT/'stage/runs/t212_curvature_direction/summary.json']
    plan['input_hashes']={str(p):sha(p) for p in files}
    used=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file())
    assert used+plan['peak_bytes']<1024**3 and shutil.disk_usage(ROOT).free>40*1024**3
    OUT.mkdir()
    for p in (HERE/'protocol.json',OUT/'protocol.json'):p.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'state':'REGISTERED','wall_seconds':90,'protocol_sha256':sha(HERE/'protocol.json')}))
if __name__=='__main__':main()
