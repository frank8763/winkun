from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t225_paired_coverage'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not OUT.exists(),'No duplicate T225'
    archive=json.loads((ROOT/'stage/docs/results_20261031_archive/release_receipt.json').read_text());assert archive['full_remote_binary_and_all_member_hashes_verified']
    cold=json.loads((ROOT/'stage/docs/results_20261031_archive/cold_storage_receipt.json').read_text());assert cold['original_payloads_preserved_locally_and_remotely']
    base=json.loads((ROOT/'stage/experiments/t223_coverage_repair/protocol.json').read_text());ep=json.loads(Path(base['evaluation_protocol']).read_text());jobs=[]
    for reset in ['LEFT_06','LEFT_21']:
        for seed in range(2026103221,2026103227):
            for arm in ['parent','candidate']:
                j=dict(next(j for j in ep['jobs'] if j['id']==reset));j.update(id=f'{reset}_{seed}_{arm}',arm=arm,round=-1,role='paired_frozen_coverage',seed=seed,mode='correlated_diagnostic');jobs.append(j)
    for reset in ['RIGHT_00','RIGHT_08','RIGHT_18','RIGHT_26']:
        j=dict(next(j for j in ep['jobs'] if j['id']==reset));j.update(id=reset+'_parent',arm='parent',round=-1,role='right_retention_source',seed=0,mode='deterministic');jobs.append(j)
    plan=dict(experiment='T225',checkpoint=base['checkpoint'],candidate_recipe=str(ROOT/'stage/experiments/t221_left_success/protocol.json'),candidate_hash='2d075594fd336e4097bdad09fd7e60c73bb6037b67de34ffa92be97a06003f01',jobs=jobs,workers=4,max_commands=56,max_physics=1306704,wall_seconds=420,final_bytes=8*1024**2,peak_bytes=8*1024**2,global_bytes=1024**3,
        contract='New bounded paired frozenT194 versus reconstructedT221 originalLEFT failedreset exploration diagnostic; not restartT223 or newtraining. TwoLEFTresets x sixpredeclaredseeds x2policies=24pairedsources +4T194RIGHTcornerretentionsources; all28 independently regenerated twice and10arrays/finalintegrationexact. AR.98nativecopula withsigma(.35,.30), sameinnovationsperpair, initialstatefixed. No bestseedselection/retry/extension; completeallfixedjobs includingfailures. Storeall10arraysfullbothsuccess/failureonce npz <=256KiBperfile, secondindependenthashes. Zero RL/supervisedupdates except3000T221recipe reconstruction counteddiagnostic. PreservecandidatecompressedcheckpointinRAMgzip<2.2MiB onlyifbudgetallows, noActorchange. Endpoint/scenario/radius/horizonunchanged, no planner/waypoints/runtimeTeacher/Phase. Needatleast1safeGOALforeachLEFTfailedreset andall4RIGHTsources beforeseparatefuturelearningproposal. Noautomatictraining/promotion/nextlaunch.',
        references=['https://arxiv.org/html/2312.11091v2','https://arxiv.org/abs/1703.09327','https://arxiv.org/abs/2306.02437'])
    paths=list(HERE.glob('*.py'))+[ROOT/'stage/experiments/t223_coverage_repair/repair_support.py',ROOT/'stage/experiments/t221_left_success/left_support.py',Path(plan['candidate_recipe']),ROOT/'stage/runs/t221_left_success/verified_success.npz',ROOT/'stage/runs/t221_left_success/recipe.json',ROOT/'stage/experiments/t219_rnd_runtime/support.py',ROOT/'stage/experiments/t219_rnd_runtime/native_compat/g5b.py',Path(plan['checkpoint'])]+[Path(j['exact_snapshot']) for j in jobs]
    plan['input_hashes']={str(p):sha(p) for p in paths};size=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file())
    assert size+plan['peak_bytes']<plan['global_bytes'] and shutil.disk_usage(ROOT).free>40*1024**3
    OUT.mkdir()
    for p in [HERE/'protocol.json',OUT/'protocol.json']:p.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(state='REGISTERED',jobs=len(jobs),runs_MiB=size/1024**2)))
if __name__=='__main__':main()
