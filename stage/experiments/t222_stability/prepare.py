from pathlib import Path
import json,hashlib,shutil,itertools
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t222_stability'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not OUT.exists(),'No duplicate T222'
    previous=ROOT/'stage/runs/t221_left_success';s=json.loads((previous/'summary.json').read_text())
    assert s['state']=='COMPLETE' and s['reconstruction_verified'] and s['old8_safeGOAL']==s['old8_measured']==8 and s['all_workers_stopped']
    old=json.loads((ROOT/'stage/experiments/t221_left_success/protocol.json').read_text());jobs=[]
    for i in range(6):jobs.append(dict(id=f'old6_{i}',case=old['primary_cases'][i],exact_snapshot=old['task_snapshots'][i],group='old6',delta_x_m=0.,delta_y_m=0.,delta_yaw_deg=0.))
    for i,side in [(6,'LEFT'),(7,'RIGHT')]:
        for j,(dx,dy,yaw) in enumerate(itertools.product([-.03,0.,.03],[-.03,0.,.03],[-2.,0.,2.])):
            jobs.append(dict(id=f'{side}_{j:02}',case=old['primary_cases'][i],exact_snapshot=old['task_snapshots'][i],group=side,delta_x_m=dx,delta_y_m=dy,delta_yaw_deg=yaw))
    plan=dict(experiment='T222',jobs=jobs,workers=4,max_commands=60,max_physics=1500000,wall_seconds=360,final_bytes=256*1024,peak_bytes=256*1024,global_bytes=1024**3,
        final_policy_sha256=json.loads((previous/'recipe.json').read_text())['final_policy_sha256'],
        contract='Independent fresh-process reconstruction of T221 fullMLP fromhashedparent+verifiedsuccessdata+3000deterministicAdamrecipe; hashmustmatchfinalpolicybeforeeval. Rebuildoptimizer3000callsdiagnosticonly, navigationtraining0/learningupdates0. Frozen deterministic singleActor originalnative observation/actions/goal/radius/timestep/horizon. Six oldtasks + LEFT/RIGHT each27fixedresetgrid dx/dy±.03m,yaw±2deg; no maps/planner/waypoints/runtimeTeacher/Phase. Perturbinitialvehicleposeonlybeforeepisode, fixedoriginalfinalgoal/scene; rebuildcausalobservationhistoryasT195convention. Full originalzerooffsetsnapshot centers preserved. Rejectinvalidstart, no retry/replacement. All cases/GOAL/contact/timeout/censored saved with SHA of fullstate/actionsequence/finalintegration, code/input/modelhashes/resetcontract permit deterministic reproduction, no percasefulltracefiles. Max60commands/1.5Mallphysics/360seconds/256KiBown/global1GiB/free40GiB. Localstable5m engineeringgate requires all6old and both27grids safeGOAL + centers exactnativecounts and finalintegration matchT221; not54independentheldoutmaps or statisticalgeneralization, noautomaticpromotion/longdistance/nexttraining.',
        references=['https://arxiv.org/abs/2108.13264'])
    paths=list(HERE.glob('*.py'))+[previous/'summary.json',previous/'recipe.json',previous/'verified_success.npz',ROOT/'stage/experiments/t221_left_success/left_support.py',ROOT/'stage/experiments/t221_left_success/protocol.json',ROOT/'stage/experiments/t219_rnd_runtime/support.py',ROOT/'stage/experiments/t219_rnd_runtime/native_compat/g5b.py',ROOT/'stage/experiments/t216_left_specialist/support.py',ROOT/'stage/experiments/t203_native_exploration/runtime.py',ROOT/'stage/experiments/t200_residual_placement/common.py']+list(map(Path,old['task_snapshots']))
    plan['input_hashes']={str(f):sha(f) for f in paths};size=sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file())
    assert size+plan['peak_bytes']<plan['global_bytes'] and shutil.disk_usage(ROOT).free>40*1024**3
    OUT.mkdir()
    for f in (HERE/'protocol.json',OUT/'protocol.json'):f.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'state':'REGISTERED','cases':len(jobs),'runs_MiB':size/1024**2}))
if __name__=='__main__':main()
