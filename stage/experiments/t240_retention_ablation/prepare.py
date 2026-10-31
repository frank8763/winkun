"""Single-factor q99 ablation; reuse ended frozen controls transparently."""
from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t240_retention_ablation'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not OUT.exists(),'No duplicateT240'
    prev=ROOT/'stage/runs/t239_ar_policy_gradient';s=json.loads((prev/'summary.json').read_text());assert s['all_workers_stopped'] and s['development_safeGOAL']==210 and s['RL_accepted_updates']==8
    original=json.loads((ROOT/'stage/experiments/t239_ar_policy_gradient/protocol.json').read_text());plan=dict(original);rows=list(map(json.loads,(prev/'training.jsonl').read_text().splitlines()));controls=[r for r in rows if r['round']==0 or r['arm']=='frozen'];shared=[r for r in rows if r['round']==0];assert len(controls)==32 and len(shared)==8
    plan.update(experiment='T240',max_anchor_q99=original['max_anchor_error'],frozen_controls=controls,shared_initial_episodes=shared,max_commands=384,max_physics=8960256,wall_seconds=1500,final_bytes=8*1024**2,peak_bytes=8*1024**2,
      contract='Single-factor ablation against completedT239: sameT231 initialization/4rounds/seeds/8learnerrolloutsperround/ARsigma,rho/reward/fullActorAdam/learningrate/proposalrollback/surrogate/retentionweight/MSE4e-6/maxerror.02/k3guards/old8centerstop/211then64gate. Only q99extra guard .003 removed: setbound equal existingmaxerror.02, making it redundant under same maxguard, stillrecordq99. Not an alpha/lambda/seed grid or restart ofT239; initialT231 held identical to isolate guard effect. T2398acceptedupdates and210/211 result preserved. Frozen32controls fully reuse ended sameparent/samefixedseedsT239 verified evidence, not new independent experiments. T240first8learnerdata independentlyregeneratedtwice and must exactmatchT239initial8, countphysicalreconstruction and explicitlydoNOTcount8asnewindependentdata. Remaining24learnerrollouts are new policy-induced data; all32independently regenerated again,64commandsplus baseline2+4x10center/hardchecks+211dev+64possibleheldout<=381(max384),8.960256Mallphysics/1500s/reserve650sfinal/8MiBown/global1GiB/free40GiB. FullMC/conditionaldensity/syntheticcontracts/shadow exactchecks unchanged. Finite-state constraintsNOTCPO/TRPO/safetyproof. No externalplanner/map/waypoint/runtimeTeacher/modelrouting/Phase. Finalfixedbudgetmodel notbestintermediate; preserveT159/T194/T221/T231/T239/allfailures; noautomaticlongdistance/promotion.',
      references=original['references']+['https://proceedings.mlr.press/v70/achiam17a.html'])
    for k in original:
        if k not in {'experiment','max_anchor_q99','max_commands','max_physics','wall_seconds','final_bytes','peak_bytes','contract','references','input_hashes'}:assert plan[k]==original[k],k
    paths=list(HERE.glob('*.py'))+[prev/'summary.json',prev/'training.jsonl',prev/'updates.jsonl']+[Path(r['path']) for r in controls];plan['input_hashes']={**original['input_hashes'],**{str(p):sha(p) for p in paths}}
    used=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file());assert used+plan['peak_bytes']<plan['global_bytes'] and shutil.disk_usage(ROOT).free>40*1024**3
    OUT.mkdir()
    for p in [HERE/'protocol.json',OUT/'protocol.json']:p.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(state='REGISTERED',only_algorithmic_change='remove_extraq99constraint',runs_MiB=used/1024**2)))
if __name__=='__main__':main()
