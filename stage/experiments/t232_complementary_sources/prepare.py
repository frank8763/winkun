from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t232_complementary_sources'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not OUT.exists(),'No duplicateT232'
    previous=ROOT/'stage/runs/t230_remaining_sources';s=json.loads((previous/'summary.json').read_text());assert s['state']=='COMPLETE' and s['all_workers_stopped'] and not s['success_source_gate']
    failed=json.loads((ROOT/'stage/experiments/t230_remaining_sources/protocol.json').read_text())['failed_reset_ids'];old=json.loads((ROOT/'stage/experiments/t229_valid_start_acceptance/protocol.json').read_text());jobs=[]
    for j in old['jobs']:
        if j['group']=='task6' or (j['group']=='task4' and j['id'] in failed):jobs.append(dict(j,id=j['id']+'_T221det',source_reset_id=j['id'],arm='T221det',round=-1,role='complementary_existing_neural_policy',seed=0,mode='deterministic'))
    assert len(jobs)==30
    pre=[r for r in map(json.loads,(previous/'episodes.jsonl').read_text().splitlines()) if r['complete'] and r['success'] and not r['collision']]
    t225=json.loads((ROOT/'stage/runs/t225_paired_coverage/summary.json').read_text())
    plan=dict(experiment='T232',checkpoint=t225['export']['path'],candidate_hash='2d075594fd336e4097bdad09fd7e60c73bb6037b67de34ffa92be97a06003f01',jobs=jobs,failed_reset_ids=[j['source_reset_id'] for j in jobs],required_source_ids=failed,previous_verified_source_ids=sorted(set(r['source_reset_id'] for r in pre)),sigma=None,rho=None,max_commands=60,max_physics=1400040,wall_seconds=420,final_bytes=6*1024**2,peak_bytes=6*1024**2,global_bytes=1024**3,
        contract='FrozenretainedT221 deterministiccomplementarity/dataaudit; notT230rerun/newseedsearch. T221 previouslyLEFT25/27 vsT22823/27, failuresetsdiffer, implyingownalreadyverifiedneuralbehaviors maycoverunion. Fixedfull27LEFT5mgrid +3current4mLEFTfailedresets=30, everycaseindependentlytwice all10arrays/finalintegrationexact, keepallfailures/fullarrayslossless. Noactionnoise, nooptimizer, no manualactionteacher/planner/maps/waypoint/runtimeTeacher/Phase. CombineverifiedT230safesourceidswithT221detsourceids onlyforexplicituniongateevery7currentfailedreset; nochangeT230failedgate/summary. Full30casescomplete, nofirstsuccessstop orbestmodelsearch. SameoriginalnativeGOAL/radius/horizon/reset/scene. 60commands/1400040allphysics/420s/6MiBown/global1GiB/free40GiB, noautomatictraining/promotion/nextlaunch/longdistance.',
        references=['https://arxiv.org/abs/1511.06295','https://arxiv.org/abs/2306.02437'])
    files=list(HERE.glob('*.py'))+[Path(plan['checkpoint']),previous/'summary.json',previous/'episodes.jsonl',ROOT/'stage/experiments/t229_valid_start_acceptance/protocol.json',ROOT/'stage/experiments/t223_coverage_repair/repair_support.py',ROOT/'stage/experiments/t219_rnd_runtime/support.py',ROOT/'stage/experiments/t219_rnd_runtime/native_compat/g5b.py']+[Path(j['exact_snapshot']) for j in jobs]
    plan['input_hashes']={str(p):sha(p) for p in files};size=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file());assert size+plan['peak_bytes']<plan['global_bytes'] and shutil.disk_usage(ROOT).free>40*1024**3
    OUT.mkdir()
    for p in (HERE/'protocol.json',OUT/'protocol.json'):p.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(state='REGISTERED',cases=len(jobs),runs_MiB=size/1024**2)))
if __name__=='__main__':main()
