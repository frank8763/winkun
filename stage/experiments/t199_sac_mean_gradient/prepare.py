import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
from common import *

def main():
    assert not OUT.exists()
    models={'T194':REPO/'stage/runs/t194_drive_sensitivity/selected_policy.pt','T196_action':REPO/'stage/runs/t196_left_exploration/action_noise/checkpoint_latest.pt','T196_parameter':REPO/'stage/runs/t196_left_exploration/parameter_noise/checkpoint_latest.pt','T197_control':REPO/'stage/runs/t197_timeout_progress/no_timeout_distance/checkpoint_latest.pt','T197_penalty':REPO/'stage/runs/t197_timeout_progress/timeout_distance/checkpoint_latest.pt'}
    states=[REPO/'stage/runs/t196_left_exploration/retention_targets.npz']+[REPO/f'stage/runs/t196_left_exploration/episodes/action_noise_b{i}_w0.npz' for i in range(6)]
    plan={'experiment':'T199','seed':2026103099,'checkpoints':{k:str(v) for k,v in models.items()},'common_states':str(REPO/'stage/runs/t196_left_exploration/left_baseline_trace.npz'),'verification_states':[str(p) for p in states],'physics_steps':0,'optimizer_updates':0,'scope':'Offline inverse-squash gradient audit and matched-size residual candidates, no navigation or learned improvement claim.'}
    files=list(models.values())+states+[Path(plan['common_states'])]+list(HERE.glob('*.py'));plan['input_hashes']={str(p):sha(p) for p in files}
    OUT.mkdir();save(HERE/'protocol.json',plan);save(OUT/'protocol.json',plan);print(json.dumps({'state':'PREPARED','protocol_sha256':sha(HERE/'protocol.json')}))

if __name__=='__main__':main()
