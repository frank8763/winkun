from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t229_valid_start_acceptance'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not OUT.exists(),'No duplicate T229'
    prior=ROOT/'stage/runs/t228_balanced_stability';s=json.loads((prior/'summary.json').read_text());assert s['all_workers_stopped'] and s['stop_reason']=='INVALID_START_CASE_THEN_MISSING_DELTA_FIELDS_IN_SUMMARY'
    old=json.loads((ROOT/'stage/experiments/t228_balanced_stability/protocol.json').read_text())
    plan=dict(experiment='T229',jobs=old['development_jobs'],checkpoint=s['export']['path'],checkpoint_sha256=s['export']['sha256'],policy_hash=json.loads((prior/'recipe.json').read_text())['final_policy_sha256'],prior_cases=str(prior/'development.jsonl'),max_commands=300,max_rollouts=84,max_physics=2000000,wall_seconds=600,final_bytes=16*1024**2,peak_bytes=16*1024**2,global_bytes=1024**3,
        contract='Newfrozencompletion/audit afterT228originalacceptancestoppedinvalidstart andpostprocessingfailed; do not restart/resumeoriginalbudget orchangeweights. Zero-stepnativephysicalvaliditypreflightall216originalfixedresetpositions, no replacement/retry/displacement. Reportallinvalidstartids/poses separately; invalidstates cannotprovepolicyfailure. Sameoriginalgoals/scenes/radius/horizon. ReuseallT228alreadymeasuredvalidcompletecases withsameexactmodelhash/case/reset; dopostprocessingjoinbyid notnewphysics. Four3moriginalcentercontrolsindependentlyfreshreplay mustexactnativecounts/observations/actions/finalintegrationmatchpriorAacceptancehashes. Onlypreviouslyunmeasuredphysicalvalidfixedresetcases are newlyrolled out; keepallactualfailures and hashverifiedfullnewarrays<=192KiBcompactlossless nextstateencoding. Thusnotquietlydropfailingvalidcases orrerollsuccessfulcasesfortuning. Validdomain counts explicitlynot216/216; originalT228strictgate remainsfailed. Prior3valid4mLEFTfailuresretain, so heldout64notrun/nopromotion/preview. FrozenT228/zerooptimizer, no planner/waypoint/teacher/Phase. Max216validitycommands+84rolloutcommands=300/2Mallphysics/600s/16MiBown/global1GiB/free40GiB. Localvalid-domaingate onlyifall216validatedandallvalidmeasuredsafe withcontrolsverified; notoriginalstrictgate/newheldoutclaim. No automatictraining/nextlaunch/longdistance.',
        references=['https://arxiv.org/abs/2108.13264','https://arxiv.org/abs/1703.09327'])
    files=list(HERE.glob('*.py'))+[Path(plan['checkpoint']),prior/'summary.json',prior/'recipe.json',Path(plan['prior_cases']),ROOT/'stage/experiments/t228_balanced_stability/protocol.json',ROOT/'stage/experiments/t223_coverage_repair/repair_support.py',ROOT/'stage/experiments/t219_rnd_runtime/support.py',ROOT/'stage/experiments/t219_rnd_runtime/native_compat/g5b.py']+[Path(j['exact_snapshot']) for j in plan['jobs']]
    plan['input_hashes']={str(p):sha(p) for p in files};size=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file());assert size+plan['peak_bytes']<plan['global_bytes'] and shutil.disk_usage(ROOT).free>40*1024**3
    OUT.mkdir()
    for p in [HERE/'protocol.json',OUT/'protocol.json']:p.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(state='REGISTERED',runs_MiB=size/1024**2)))
if __name__=='__main__':main()
