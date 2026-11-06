"""Completed Stage originals and evidence: RAM archive and exact readback only."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T
LEGACY=['t228_balanced_stability','t231_retained_behavior_repair','t239_ar_policy_gradient','t240_retention_ablation','t243_induced_retention_repair','t245_multitask_pcgrad']
old=T.B.require_stopped
def stopped(d):
    if d.name not in [*LEGACY,'t396_cold_reserve','t396_iterative_recovery']:return old(d)
    s=json.loads((d/'summary.json').read_text())
    allowed=['COMPLETE','FAILED']+(['ENDED_WITH_ACCEPTANCE_AND_POSTPROCESSING_ERROR'] if d.name=='t228_balanced_stability' else [])
    assert s['state'] in allowed and s['all_workers_stopped']
    if d.name not in LEGACY:
        assert s['state']=='COMPLETE'
        a=ROOT/'stage/docs/review_20261106/T396_COLD_AUDIT.json' if d.name=='t396_cold_reserve' else d/'audit.json'
        assert json.loads(a.read_text())['state']=='PASS'
    if s.get('pid'):assert not Path('/proc',str(s['pid'])).exists()
    if (d/'workers.json').exists():
        w=json.loads((d/'workers.json').read_text())
        for pid in [w.get('parent'),*w.get('workers',[])]:
            if pid:assert not Path('/proc',str(pid)).exists(),pid
    if d.name=='t396_iterative_recovery':
        for r in json.loads((d/'local_manifest.json').read_text())['files']:
            assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256']
if __name__=='__main__':
    T.install();T.B.require_stopped=stopped
    groups={'T396-COLD-ORIGINALS':LEGACY,'T396-COLD':['t396_cold_reserve'],'T396':['t396_iterative_recovery']}
    if sys.argv[1] in groups:T.B.GROUPS[sys.argv[1]]=groups[sys.argv[1]];T.B.main(sys.argv[1])
    elif sys.argv[1:] == ['source_sync']:
        import sync_source
        sync_source.main()
    else:raise ValueError('Completed Stage evidence only')
