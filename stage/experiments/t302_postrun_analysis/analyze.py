"""Zero-optimizer reconciliation of rollback and historical exact overlap."""
from pathlib import Path
import sys,json,gzip,os,shutil,hashlib,collections
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t302_balanced_bridge_learning'))
from t302_learning import *

def lines(path):return list(map(json.loads,gzip.decompress(Path(path).read_bytes()).decode().splitlines()))
def main():
    out=ROOT/'stage/runs/t302_balanced_bridge_learning';dest=ROOT/'stage/docs/distance_20261102/T302_POSTHOC.json';assert not dest.exists();p=json.loads((out/'protocol.json').read_text());s=json.loads((out/'summary.json').read_text());assert s['all_workers_stopped'] and not s['inflight_jobs'];workers=json.loads((out/'workers.json').read_text());pids=[workers['parent'],*workers['workers'],json.loads((out/'launch.json').read_text())['pid']]
    for pid in pids:
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise AssertionError(('Live/reused',pid))
    raw=lines(out/'cases.jsonl.gz');counts=[]
    for role,want in [('parent_bridge_centers',3),('old20_guard',None),('auxiliary_centers',3),('old_regression',275),('development',54),('heldout',16),('independent_reload',78)]:
        rr=[r for r in raw if r['role']==role];counts.append(dict(role=role,executed=len(rr),expected=want,safe=sum(r['complete'] and r['success'] and not r['collision'] for r in rr),contact=sum(r['collision'] for r in rr),timeout=sum(r['timeout'] for r in rr),invalid=sum(bool(r['error']) for r in rr),censored=sum(not r['complete'] for r in rr),not_executed=max(0,want-len(rr)) if want else None))
    report=s['supervised'];exp=s['arms']['balanced_bridge']['export'];assert C.sha(exp['path'])==exp['sha256'];saved=L.read(exp['path']);assert S.digest(saved['policy'])==exp['policy_hash']==report['policy_hash']
    exact_rollback=report['retained_steps']==0 and report['policy_hash']==p['policy_hash'];oldsource=ROOT/'stage/runs/t287_time_aligned_credit/cases.jsonl.gz';references={(r['role'],r['source_reset_id']):r for r in lines(oldsource) if r['role'] in ['old_regression','development']};overlap=[]
    if exact_rollback:
        for row in raw:
            if row['role'] not in ['old_regression','development']:continue
            key=(row['role'],row['source_reset_id']);ref=references.get(key)
            if ref is None:overlap.append(dict(role=key[0],id=key[1],reference_missing=True));continue
            assert row['array_hashes']==ref['array_hashes'] and row['integration_sha256']==ref['integration_sha256'];overlap.append(dict(role=key[0],id=key[1],all12arrays_and_integration_exact=True))
        parents={r['source_reset_id']:r for r in raw if r['role']=='parent_bridge_centers'}
        for row in [r for r in raw if r['role']=='auxiliary_centers']:
            ref=parents[row['source_reset_id']];assert row['array_hashes']==ref['array_hashes'] and row['integration_sha256']==ref['integration_sha256']
    safe='stage/runs/t255_overnight/s2_reset_final_checkpoint.pt.gz';assert C.sha(ROOT/safe)=='63e63e2b444867b26622a2ab0f0c4227bc38d48a7cd2b5a3b378d9b0266006df'
    records=[]
    for row in [r for r in raw if r['role']=='old20_guard']:
        if not(row['complete'] and row['success'] and not row['collision']):records.append(dict(id=row['source_reset_id'],policy_hash=row['policy_hash'],contact=row['collision'],timeout=row['timeout'],physics_steps=row['physics_steps']))
    allrows=[dict(role=r['role'],id=r['source_reset_id'],safe=r['success'] and not r['collision'],contact=r['collision'],timeout=r['timeout'],physics_steps=r['physics_steps'],policy_hash=r['policy_hash']) for r in raw if r['role'] in ['parent_bridge_centers','auxiliary_centers']]
    C.save(dest,dict(scope='Posthoc0native0optimizer bookkeeping. Native failures of rejected proposal are OLD20 outcomes only; proposal6m/LEFTbridge not evaluated, cannot claim it improves or fails6m. Final unchangedsafe policy evaluation is distinct from T299research10/54.',native_steps=0,optimizer_calls=0,navigation_updates=0,counts=counts,attempted_supervised_steps=report['attempted_steps'],retained_supervised_steps=report['retained_steps'],initial_metrics=report['initial'],final_metrics=report['final'],rejected_blocks=[r for r in report['blocks'] if not r['accepted']],old_guard_failures=records,final_policy_exact_safeT255=exact_rollback,final_policy_hash=exp['policy_hash'],historical_overlap=overlap,historical_exact_case_count=sum(r.get('all12arrays_and_integration_exact',False) for r in overlap),parent_candidate_bridge_rows=allrows,all_native_pids_absent=pids,global_logical_bytes=S.used(ROOT/'stage/runs'),free_GiB=shutil.disk_usage(ROOT).free/1024**3,source_sha256=C.sha(Path(__file__)),input_protocol_sha256=C.sha(out/'protocol.json')));print('T302 rollback/overlap posthoc reconciled0native0optimizer')

if __name__=='__main__':main()
