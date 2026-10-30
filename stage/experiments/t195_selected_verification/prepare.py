"""Freeze artifact reproduction and a small reset perturbation diagnostic."""
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import itertools
from common import *

def main():
    assert not OUT.exists()
    src=REPO/'stage/runs/t194_drive_sensitivity';summary=json.loads((src/'summary.json').read_text());assert summary['state']=='COMPLETE' and summary['selected'] is not None
    previous=json.loads((REPO/'stage/experiments/t194_drive_sensitivity/protocol.json').read_text())
    checkpoint=Path(summary['selected']['path']);assert sha(checkpoint)==summary['selected']['sha256']
    torch.set_num_threads(1)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)['policy'];parent=torch.load(previous['checkpoint'],map_location='cpu',weights_only=False)['policy']
    expected={k:v.clone() for k,v in parent.items()};expected['native_head.bias'][0]+=summary['selected']['delta']
    assert all(torch.equal(state[k],v) for k,v in expected.items())
    primary=[{'id':c['id'],'case':c,'perturbation':[0.,0.,0.],'group':'primary'} for c in previous['primary_cases']]
    right=previous['primary_cases'][-1];assert right['side']=='RIGHT'
    perturbations=[{'id':f'right_{i}','case':right,'perturbation':[dx,dy,yaw],'group':'right_perturbation'} for i,(dx,dy,yaw) in enumerate(itertools.product([0.,-.03,.03],[0.,-.03,.03],[0.,-2.,2.])) if (dx,dy,yaw)!=(0.,0.,0.)]
    plan={'experiment':'T195','checkpoint':str(checkpoint),'selection_summary':str(src/'summary.json'),'selection_cases':str(src/'evaluation.jsonl'),'jobs':primary+perturbations,'wall_seconds':300,'max_physics':34*23334,'max_bytes':16*1024**2,
          'contract':'Actor weights frozen and equal to selected T194 artifact. Reproduce all8original tasks exactly, then26nonzero RIGHTreset perturbations x/y±.03m,yaw±2deg. Zero grid point reuses originalRIGHT reproduction; no duplicate rollout forzero. All pose changes occur before first physics step; current sensors refreshed but original cached factory history/prev_action kept. Same finalgoal/scene/radius/caps. No planner/waypoint/runtime teacher.',
          'scope':'Artifact reproducibility plus consumed fixed-map development reset grid. Repeated canonical outcomes are not statistically independent successes. Perturbed tasks use declared legacy-history contract, not clean-history robot deployment validation. No model selection/promotion or new optimizer updates. LEFT5m still failed at entry. Stageonly,newruns1GiB/free40GiB.'}
    files=[checkpoint,src/'summary.json',src/'evaluation.jsonl',Path(previous['checkpoint'])]+list(HERE.glob('*.py'));plan['input_hashes']={str(p):sha(p) for p in files}
    OUT.mkdir();save(HERE/'protocol.json',plan);save(OUT/'protocol.json',plan);print(json.dumps({'state':'PREPARED','protocol_sha256':sha(HERE/'protocol.json')}))

if __name__=='__main__':main()
