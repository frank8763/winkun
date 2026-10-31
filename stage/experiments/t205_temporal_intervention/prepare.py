"""Register a fixed, nonadaptive causal intervention diagnostic, not deployment."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t205_temporal_intervention'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not OUT.exists(),'No duplicate T205'
 prior=json.loads((ROOT/'stage/runs/t204_native_course/summary.json').read_text());assert prior['state']=='COMPLETE' and prior['all_workers_stopped']
 old=json.loads((ROOT/'stage/experiments/t203_native_exploration/protocol.json').read_text());diag=json.loads((ROOT/'stage/runs/t196_left_exploration/diagnostic.json').read_text())
 probes=[]
 for start in [10000,12500]:
  probes.append({'id':f's{start}_baseline','start':start,'hold_steps':0,'throttle_cap':None,'steer_delta':0.,'kind':'baseline'})
  for hold in [100,2000]:
   for cap in [None,0.]:
    for steer in [-.6,-.3,0.,.3,.6]:
     if cap is None and steer==0:continue
     probes.append({'id':f's{start}_h{hold}_c{cap}_s{steer}','start':start,'hold_steps':hold,'throttle_cap':cap,'steer_delta':steer,'kind':'intervention'})
 assert len(probes)==38
 plan={'experiment':'T205','checkpoint':old['checkpoint'],'case':old['case'],'baseline_reference':str(ROOT/'stage/runs/t196_left_exploration/left_baseline_trace.npz'),'baseline_expected_steps':diag['baseline_steps'],'baseline_expected_remaining_m':diag['remaining_m'],'start_steps':[10000,12500],'probes':probes,'max_physics':650000,'wall_seconds':360,'batch_reserve_seconds':35,'workers':4,'final_bytes':16*1024**2,'peak_bytes':20*1024**2,'pending_bytes':2*1024**2,'optimizer_updates':0,'training_physics_steps':0,
 'contract':'Frozen T194 original LEFT scene/goal/radius. Reproduce original full episode exactly and capture true full integration/environment/history at20s,25s. Two unchanged suffix baselines must match original suffix states/actions/poses/final integration exactly. Thirtysix preregistered paired branches compare1decision(.2s) vs20decisions(4s) native steering residuals[-.6,-.3,0,.3,.6], with native throttle cap0 or unchanged. Zeroeffect duplicates excluded. After the window frozen neural policy continues. Same absolute remaining-time input and original remaining horizon; no resettime budget extension. No map/geometry consulted to choose an action, no sequence optimization or adaptive probe selection, no planner/waypoints. Intervention episodes are causal actuator diagnostics, never end-to-end neural navigation success or onpolicy training samples.',
 'outcomes':'Primary safeGOAL by original deadline. Secondary collision object/time, nearest/final goal distance, return-to-frozen-policy outcome, avoidance displacement/yaw, actual actuator steering response. CONTACT/TIMEOUT never treated asGOAL. Report all36 cases and2 baseline suffixes; deterministic cases not independent noise seeds. Entire-episode original path checked once, subsequent suffix branches do not count as full5m deployment. Noautomatic new training or promotion.',
 'gate':'Source must exactly match frozen T196 baseline and both suffix controls must exactly match states/actions/poses/final integration; otherwise abort before interventions. Effects interpreted as tested interventions only. If safeGOAL exists, separately plan neuralpolicy learning/reproduction; do not deploy timed control. If none, no claim of impossible geometry and no budget expansion.',
 'references':['https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf','https://onno-eberhard.appspot.com/pink','https://arxiv.org/abs/1706.01905']}
 files=[Path(plan['checkpoint']),Path(plan['baseline_reference']),ROOT/'stage/runs/t196_left_exploration/diagnostic.json',ROOT/'stage/runs/t204_native_course/summary.json',ROOT/'stage/experiments/t200_residual_placement/common.py']+list(HERE.glob('*.py'))
 plan['input_hashes']={str(p):sha(p) for p in files}
 used=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file());assert used+plan['peak_bytes']<1024**3 and shutil.disk_usage(ROOT).free>40*1024**3
 OUT.mkdir()
 for p in [HERE/'protocol.json',OUT/'protocol.json']:p.write_text(json.dumps(plan,indent=2)+'\n')
 print(json.dumps({'state':'REGISTERED','protocol_sha256':sha(HERE/'protocol.json'),'cases':len(probes),'wall_seconds':360,'max_physics':650000,'new_runs_MiB':used/1024**2}))
if __name__=='__main__':main()
