from pathlib import Path
import json,hashlib,shutil,itertools
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t228_balanced_stability'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not OUT.exists(),'No duplicate T228'
    prev=ROOT/'stage/runs/t227_small_noise_recovery';s=json.loads((prev/'summary.json').read_text());assert s['state']=='COMPLETE' and s['success_source_gate'] and s['all_workers_stopped']
    old=json.loads((ROOT/'stage/experiments/t221_left_success/protocol.json').read_text());sources=[]
    for folder,side in [('t225_paired_coverage','RIGHT'),('t227_small_noise_recovery','LEFT')]:
        for row in map(json.loads,(ROOT/'stage/runs'/folder/'episodes.jsonl').read_text().splitlines()):
            if row['complete'] and row['success'] and not row['collision'] and row['id'].startswith(side):sources.append(dict(side=side,path=row['path'],sha256=row['sha256'],id=row['id'],row_sha256=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()))
    assert sum(r['side']=='RIGHT' for r in sources)==4 and all(any(r['id'].startswith(reset) for r in sources) for reset in ['LEFT_06','LEFT_21'])
    jobs=[];holdout=[]
    for i,case in enumerate(old['primary_cases']):
        for j,(dx,dy,yaw) in enumerate(itertools.product([-.03,0.,.03],[-.03,0.,.03],[-2.,0.,2.])):
            jobs.append(dict(id=f'dev_task{i}_{j:02}',group=f'task{i}',case=case,exact_snapshot=old['task_snapshots'][i],delta_x_m=dx,delta_y_m=dy,delta_yaw_deg=yaw))
        for j,(dx,dy,yaw) in enumerate(itertools.product([-.015,.015],[-.015,.015],[-1.,1.])):
            holdout.append(dict(id=f'hold_task{i}_{j:02}',group=f'task{i}',case=case,exact_snapshot=old['task_snapshots'][i],delta_x_m=dx,delta_y_m=dy,delta_yaw_deg=yaw))
    plan={k:old[k] for k in ('checkpoint','retention_targets','primary_cases','task_snapshots','seed','updates','batch_each_task','lr')}
    plan.update(experiment='T228',additional_sources=sources,development_jobs=jobs,heldout_jobs=holdout,max_commands=280,max_physics=6533520,wall_seconds=1000,final_reserve_seconds=600,final_bytes=4*1024**2,peak_bytes=4*1024**2,global_bytes=1024**3,
        contract='One coverage-only engineeringconsolidation afterT226successgate. SameT221 T194initialMLP/3000Adam/lr3e-5/seed2026103221/batch16each8groups/nativepreclipMSE/norm1. Old6softtargets unchanged; RIGHTgroup retainsoriginalcenter plusall4predeclaredT194rightcorner nativeprecliptargets; LEFTgroup retainsoriginalverifiedT220success plusALLsafeGOALnewT226lowsigmaactualactions, no bestseed/intermediate selection. Sources10arrays/finalintegrationalreadyindependentlyverified, NPZhashes checked. No new RLupdates orruntimeTeacher, maps/planners/waypoints/Phase. Live3000+independentshadow3000steps, each500 network+Adamhashmustmatch; onlyfinalgzipActor exported/readbackand usedforeval. FixedT221 is samehyperparametercenter-onlyreference, notfreshtrainingmultiseedcausalclaim. Firstevaluateall8tasks x27grid=216, including3m/4m/5m, notonlyold6centers. IfANDONLYall216safeGOAL andallstartsvalid/complete, evaluatepredeclared64newoffsetheldout(±1.5cm/±1deg)neverusedintraining. Elseheldoutnotrun/missing, not0/64. Limitedsame8traincourses/localnoise, notallpossibleunder5m/generalization/multimap. Hard280commands/6533520allphysics/1000s/4MiBown/global1GiB/free40GiB. Stableengineeringgate requires216/216+64/64+8centeroriginaltest+gzipreadback/reconstruction. Noautomaticlongdistance/promotion/nexttraining orrerun.',
        references=['https://arxiv.org/abs/1806.05635','https://proceedings.mlr.press/v78/laskey17a.html','https://arxiv.org/abs/2306.02437','https://arxiv.org/abs/2108.13264'])
    files=list(HERE.glob('*.py'))+[ROOT/'stage/experiments/t221_left_success/left_support.py',ROOT/'stage/experiments/t221_left_success/protocol.json',ROOT/'stage/runs/t221_left_success/verified_success.npz',ROOT/'stage/experiments/t222_stability/acceptance_support.py',ROOT/'stage/experiments/t219_rnd_runtime/support.py',ROOT/'stage/experiments/t219_rnd_runtime/native_compat/g5b.py',Path(plan['checkpoint']),Path(plan['retention_targets'])]+[Path(j['exact_snapshot']) for j in jobs]+[Path(r['path']) for r in sources]
    plan['input_hashes']={str(p):sha(p) for p in files};size=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file());assert size+plan['peak_bytes']<plan['global_bytes'] and shutil.disk_usage(ROOT).free>40*1024**3
    OUT.mkdir()
    for p in (HERE/'protocol.json',OUT/'protocol.json'):p.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(state='REGISTERED',sources=len(sources),dev=len(jobs),heldout=len(holdout),runs_MiB=size/1024**2)))
if __name__=='__main__':main()
