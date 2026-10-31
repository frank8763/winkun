"""One bounded archived-observation RND diagnostic, not navigation training."""
from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t217_rnd_preflight'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not OUT.exists(),'No duplicate T217'
    s=json.loads((ROOT/'stage/runs/t216_left_specialist/summary.json').read_text());assert s['all_workers_stopped']
    old=json.loads((ROOT/'stage/experiments/t216_left_specialist/protocol.json').read_text())
    rows=[json.loads(x) for x in (ROOT/'stage/runs/t216_left_specialist/rollouts.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['role']=='training']
    plan={'experiment':'T217_RND_OFFLINE_PREFLIGHT','seed':2026103217,'wall_seconds':120,
          'final_bytes':1024**2,'peak_bytes':2*1024**2,'physics_steps':0,'navigation_actor_updates':0,
          'retention_targets':old['retention_targets'],'parent_checkpoint':old['checkpoint'],'left_failure_rows':rows,
          'train_tasks':[0,1,4],'heldout_tasks':[2,3,5,6],'familiar_left_round':0,'predictor_updates':500,'batch_size':128,'lr':.001,
          'normalization_std_floor':.05,'noise_sigma_normalized':.01/6,'far_rms_range_threshold':.05,
          'seen_probe_count':256,'minimum_far_samples':32,'max_seen_loss_fraction':.1,
          'minimum_far_seen_ratio':2.,'maximum_noise_increment_fraction_of_far_gap':.25,
          'intrinsic_episode_cap':.05,'physics_cap':23334}
    plan['contract']=('No simulator/Actor/Value optimizer or navigation rollout. TrainONLYseparateRNDpredictor500Adamupdates onarchivedoldtasks0,1,4PLUS4familiarT216round0LEFTfailures,rays0:48normalized[0,1]. '
       'Frozenrandom48-128-128-64target andindependentpredictor;trainstatsmean/std floor.05,clipnormalizedinputs+/-5. '
       'Holdoutentiretaskfamilies2,3,5,6;samephysicalscene/possiblyoverlappingviews,NOTindependentmapgeneralization orknownnoveltylabels. '
       'Compute nearesttrainRMSscan distance,heldoutfar>=.05normalized(min32);seenprobesuniform256. '
       '1cmrayGaussianperturbation normalized1/600 asnuisancesensitivityprobe,notrealsensornoisevalidation. '
       'Gate seenloss<=.1initial,far/seen>=2,positivefargap,andq95positive noiseincrement<=.25fargap. '
       'Evaluate8archivedLEFTfailedcorridorepisodes separately;neverlabelthem successfulornewroutecoverage. '
       'Boundedpoststepintrinsicreward <=.05perfulloriginal23334physicssteps, collisionmasked0;no globalpose/map/goal/obstacleID/future information input, '
       'no runtimeactionfilter/planner/waypoint/teacher/Phase research. Fullmodeltargetfrozen/hashchecked;navigationparenthashunchanged. '
       'Target/predictor/normalizercheckpointmayexport<=.5MiB,nooptimizerstates. Thispreflightdoesnotauthorizeonlinenavigationtraining. '
       '120s/1MiBfinal2MiBpeak/globalnewruns1GiB/free40GiB;stoponerror/time/storage,no duplicate/resume/extension.')
    plan['sources']=['https://arxiv.org/abs/1810.12894','https://github.com/openai/random-network-distillation/blob/master/policies/cnn_gru_policy_dynamics.py',
        'https://arxiv.org/html/2404.12594','https://pmc.ncbi.nlm.nih.gov/articles/PMC13667403/']
    files=list(HERE.glob('*.py'))+[Path(plan[k]) for k in ('retention_targets','parent_checkpoint')]+[Path(r['trace']) for r in rows]+[ROOT/'stage/runs/t216_left_specialist/summary.json']
    plan['input_hashes']={str(p):sha(p) for p in files}
    used=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file())
    assert used+plan['peak_bytes']<1024**3 and shutil.disk_usage(ROOT).free>40*1024**3
    OUT.mkdir()
    for p in (HERE/'protocol.json',OUT/'protocol.json'):p.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'state':'REGISTERED','runs_MiB':used/1024**2,'protocol_sha256':sha(HERE/'protocol.json')}))
if __name__=='__main__':main()
