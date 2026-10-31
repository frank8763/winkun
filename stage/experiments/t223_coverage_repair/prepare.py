from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t223_coverage_repair'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not OUT.exists(),'No duplicate T223'
    prior=ROOT/'stage/runs/t222_stability';s=json.loads((prior/'summary.json').read_text());assert s['state']=='COMPLETE' and s['all_workers_stopped'] and not s['stable_5m_local_grid_gate_passed']
    base=json.loads((ROOT/'stage/experiments/t221_left_success/protocol.json').read_text());evalplan=json.loads((ROOT/'stage/experiments/t222_stability/protocol.json').read_text())
    jobs=[]
    for side,ids in [('RIGHT',[0,8,18,26]),('LEFT',[6,21])]:
        for i in ids:
            job=dict(next(j for j in evalplan['jobs'] if j['id']==f'{side}_{i:02}'))
            job.update(id='source_'+job['id'],arm=side,round=-1,role='coverage_source',seed=2026103221,mode='deterministic' if side=='RIGHT' else 'correlated_diagnostic');jobs.append(job)
    plan={k:base[k] for k in ('checkpoint','retention_targets','primary_cases','task_snapshots','seed','updates','batch_each_task','lr')}
    plan.update(experiment='T223',jobs=jobs,evaluation_protocol=str(ROOT/'stage/experiments/t222_stability/protocol.json'),max_commands=72,max_physics=1800000,wall_seconds=480,
        final_bytes=128*1024,peak_bytes=128*1024,global_bytes=1024**3,
        contract='One bounded self-imitation coverage repair after T222 LEFT25/27 RIGHT9/27. SameT194fullMLP/old7softtargets+verifiedT221LEFTdata/3000Adam/lr3e-5/batch16per8groups. Change ONLY extra state/action training coverage in this engineering candidate, not RLeffectclaim. Four predeclared RIGHT corner resets use frozen originalT194 deterministic oldsuccessbehavior; two knownLEFTfailedresets use frozenT194 AR.98 seed2026103221. No externalteacher/planner/runtimecontroller/Phase. Every fresh source repeats independently beforelearning withall10arrays/finalintegrationexact; 12diagnosticcommands. Use only complete safeGOAL sources; keep failures in report; require>=1additionalLEFTsourceandall4RIGHTsafeGOAL or do not train. Add RIGHT teacherprecliptargets to oldrightgroup6; add LEFTactualsuccessfulcommands to group7, both groups remain equallyweighted with sixold. All data RAM+exactfrozenparent/reset/innovationrecipe andhashes, no unique filesdeleted. Live3000+independentshadow3000supervisedsteps, RLupdates0; each500 parameter+Adamhashmustmatch. Future reconstruction regenerates6freshsourceepisodes (countsasdiagnosticphysics), checksrecordedhashes, thenrunsfixedrecipe, verifiesfinalhash; no binarycheckpointexport. Same previouslyuseddevelopmentgrid60frozencases no model selection, notindependentheldoutevaluation. Save all outcomes/hashesthinly. 72commands/1.8Mallphysics/480s/128KiBown/global1GiB/free40GiB. No duplicate/resume/extension/automaticnexttrial/promotion/longdistance.',
        references=['https://arxiv.org/abs/1806.05635','https://arxiv.org/abs/1511.06295','https://arxiv.org/abs/1703.09327'])
    paths=list(HERE.glob('*.py'))+[ROOT/'stage/experiments/t221_left_success/left_support.py',ROOT/'stage/experiments/t221_left_success/protocol.json',ROOT/'stage/experiments/t222_stability/acceptance_support.py',Path(plan['evaluation_protocol']),prior/'summary.json',ROOT/'stage/runs/t221_left_success/verified_success.npz',ROOT/'stage/experiments/t219_rnd_runtime/support.py',ROOT/'stage/experiments/t219_rnd_runtime/native_compat/g5b.py',ROOT/'stage/experiments/t220_temporal_noise/temporal_support.py',ROOT/'stage/experiments/t203_native_exploration/runtime.py',ROOT/'stage/experiments/t200_residual_placement/common.py']+[Path(plan[k]) for k in ('checkpoint','retention_targets')]+list(map(Path,plan['task_snapshots']))
    plan['input_hashes']={str(f):sha(f) for f in paths};size=sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file())
    assert size+plan['peak_bytes']<plan['global_bytes'] and shutil.disk_usage(ROOT).free>40*1024**3
    OUT.mkdir()
    for f in (HERE/'protocol.json',OUT/'protocol.json'):f.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'state':'REGISTERED','runs_MiB':size/1024**2}))
if __name__=='__main__':main()
