"""Register fresh MuJoCo video only after the full 3m–5m local stability gate."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t244_verified_mujoco'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not OUT.exists(),'No duplicate T244'
    previous=ROOT/'stage/runs/t243_induced_retention_repair';s=json.loads((previous/'summary.json').read_text());assert s['state']=='COMPLETE' and s['stable_valid_local_3m_4m_5m_gate'] and s['all_workers_stopped']
    p=json.loads((ROOT/'stage/experiments/t243_induced_retention_repair/protocol.json').read_text());centers=[]
    for j in p['development_jobs']:
        if j['delta_x_m']==j['delta_y_m']==j['delta_yaw_deg']==0.:centers.append(dict(id='fresh_video_'+j['group'],case=j['case'],exact_snapshot=j['exact_snapshot']))
    assert len(centers)==8
    plan=dict(experiment='T244',checkpoint=s['export']['path'],checkpoint_sha256=s['export']['sha256'],policy_hash=json.loads((previous/'recipe.json').read_text())['final_policy_sha256'],jobs=centers,max_physics=186672,max_cases=8,wall_seconds=360,final_bytes=int(2.5*1024**2),global_bytes=1024**3,
        contract='OnlyafterT243211validdevelopment+ALLvalidnewheldoutoffsetssafeGOALand8originalcentersgate. Original5invaliddevelopmentstarts separatelyreported. T243 must pass ALL211validdevelopment and64heldout navigation; notallpossible5m/multimapgeneralization. Userreviewsfreshvideo BEFOREanylongdistanceexperiment. Fresh8originalcenterclosedloopdeterministicMuJoCoruns, samehasheddeserializedsingleActor/113causalinput/nativeactionhold100/.002physics/originalGOAL/radius/horizon. Zero training. Newvideo8panels1280x480/5fps/nominal1x, cam+greenfinalgoalbeaconvisualonly notphysicsgeom orpolicyinput. StopnativeepisodeonGOAL/contact/timeout;freezecompletedframe withlabel, no learnedparkingclaim. Everyepisodeexactnativecounts/finalintegrationmustmatchT243center; no savedactionplayback. Keepallverification/failures, no prettyvideo=successclaim. Encodingrate300kbps bound2MiBmovie, own2.5MiB/global1GiB/free40GiB. 8freshcases/186672allphysics/360s, no duplicate/extension/nextlaunch. Video only samefixed8tasks, notnewmaps orallpossibleunder5m. Phase isolated.')
    files=list(HERE.glob('*.py'))+[Path(plan['checkpoint']),previous/'summary.json',previous/'development.jsonl',ROOT/'stage/experiments/t219_rnd_runtime/support.py',ROOT/'stage/experiments/t219_rnd_runtime/native_compat/g5b.py',ROOT/'stage/experiments/t200_residual_placement/common.py',ROOT/'stage/experiments/t203_native_exploration/runtime.py']+[Path(j['exact_snapshot']) for j in centers]
    plan['input_hashes']={str(f):sha(f) for f in files};used=sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file());assert used+plan['final_bytes']<plan['global_bytes'] and shutil.disk_usage(ROOT).free>40*1024**3
    OUT.mkdir()
    for f in (HERE/'protocol.json',OUT/'protocol.json'):f.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(state='REGISTERED',runs_MiB=used/1024**2)))
if __name__=='__main__':main()
