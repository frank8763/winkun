from pathlib import Path
import ast,hashlib,json,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t285_trajectory_constraints'
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()


def main():
    assert not OUT.exists() and not (HERE/'protocol.json').exists()
    src=ROOT/'stage/runs/t283_capped_task_balance';drift=ROOT/'stage/runs/t284_feedback_drift';base=json.loads((src/'protocol.json').read_text())
    analysis=json.loads((drift/'analysis.json').read_text());first=analysis['first_xy_threshold_crossings']['0.01']['index'];assert first==8
    with np.load(drift/'parent.npz') as z:n=len(z['states'])
    arms=dict(prefix_before_1cm=list(range(first+1)),uniform_trajectory=np.rint(np.linspace(0,n-1,9)).astype(int).tolist())
    train=[json.loads(l) for l in (src/'cases.jsonl').read_text().splitlines()];train=[r for r in train if r['role']=='training' and r['id'].startswith('capped_task_balance_r0_')];assert len(train)==16
    reports=json.loads((src/'round_updates.jsonl').read_text().splitlines()[0]);balance=reports['updates'][0]['fixed_round_balance']
    paths=[*HERE.glob('*.py'),src/'protocol.json',src/'cases.jsonl',src/'round_updates.jsonl',drift/'historical_candidate.pt.gz',drift/'parent.npz',drift/'analysis.json',drift/'summary.json',
        ROOT/'stage/docs/long_distance_20261101/T284_AUDIT.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T284.json']+[Path(r['path']) for r in train]
    hashes=dict(base['input_hashes']);hashes.update({str(f):sha(f) for f in paths})
    for f,h in hashes.items():assert sha(f)==h,f
    mib=1024**2;critical=next(j for j in base['acceptance_guard_jobs'] if j['id']=='dev_task6_24')
    p=dict(experiment='T285',source_protocol=str(src/'protocol.json'),rejected_candidate=str(drift/'historical_candidate.pt.gz'),rejected_policy_hash='e63afdb61ae5a59bdfdc9d74df885d17053b51a3f75462f8e25a9c187cd3c859',
        parent_trajectory=str(drift/'parent.npz'),historical_training=train,balance=balance,arms=arms,relative_eigenvalue_cutoff=1e-8,
        critical_job=critical,remaining_guard_jobs=[j for j in base['acceptance_guard_jobs'] if j['id']!=critical['id']],six_m_centers=base['centers'],local_centers=base['local_centers'],
        input_hashes=hashes,max_commands=50,max_physics=1400000,wall_seconds=900,postrun_audit_seconds=300,
        model_bytes=3*mib,full_episode_bytes=384*1024,compact_episode_bytes=12288,
        payload_bytes=6*mib+4*384*1024+46*12288,metadata_bytes=768*1024,final_bytes=9*mib,global_bytes=1024**3,min_free_bytes=40*1024**3,
        optimizer_calls=0,navigation_training_updates=0,
        contract='Two fixed posthoc consumed-data frozen transformations of identical T283 historical round0 factor.0625 actual Adam displacement. Prefix0..8 through first1cm divergence vs nine uniform original whole-parent-trajectory states, each18 outputJacobianrows. Row-normalized float64 Gram eigenspace projection relative eigenvalue cutoff1e-8 absolute1e-12; zero rows1e-12; no ridge/rank/stepsize/anchor grid. Actual float32 model loading, output change and original balanced RL surrogate/softretention/trust limits measured. Original first16 data has exact parent behavior, reused not freshindependent. Two models retained even rejection. Numeric-pass each critical old5m twice exact; ifsafe other19oldguard; if20safe two6m andtwo1m centers frozen diagnostic. 50allcommands1.4Mallphysics900s9MiB/global1GiB/free40GiB, no new Adam or navigation training. Full275/54/16/78 NOT executed, cannot promote/claim6m. ParentT255andallfailedmodelsretained. Distinct fromT211twofirstorderconstraints butprojectionknownOGD/AdamNSCLpriorart; nooriginality/causality/stabilityclaim. Uniqueendeddirnoresume, source remoteexact before native, end300s0opt0physaudit plusfullremoteRAMbackup. NoPhase/map/planner/waypoint/teacher/router.')
    assert len(p['remaining_guard_jobs'])==19 and len(p['six_m_centers'])==len(p['local_centers'])==2
    assert p['payload_bytes']+p['metadata_bytes']<=p['final_bytes']
    used=sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file());assert used+p['final_bytes']<=p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    OUT.mkdir();blob=json.dumps(p,ensure_ascii=False,indent=2)+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob)
    print(json.dumps(dict(state='REGISTERED',full_reserved_global=used+p['final_bytes'],protocol_sha256=sha(HERE/'protocol.json'))))


if __name__=='__main__':main()
