"""Completed-run RAM archive and exact source sync only."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T
original=T.B.require_stopped
def stopped(d):
    if d.name in ['t403_cold_exact_reserve','t404_last_narrow','t405_wide_admission']:
        s=json.loads((d/'summary.json').read_text());a=json.loads((d/'audit.json').read_text());assert s['state']=='COMPLETE' and a['state']=='PASS' and s['all_workers_stopped']
        assert not Path('/proc',str(s['pid'])).exists()
        if (d/'workers.json').exists():
            for pid in json.loads((d/'workers.json').read_text()).get('workers',[]):assert not Path('/proc',str(pid)).exists()
        if (d/'local_manifest.json').exists():
            for r in json.loads((d/'local_manifest.json').read_text())['files']:assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256']
        return
    return original(d)
if __name__=='__main__':
    T.install();T.B.require_stopped=stopped;mode=sys.argv[1]
    if mode in ['T403','T404','T405']:
        T.B.GROUPS[mode]=[{'T403':'t403_cold_exact_reserve','T404':'t404_last_narrow','T405':'t405_wide_admission'}[mode]];T.B.main(mode)
    elif mode=='source_sync':
        import sync_source;sync_source.main()
    else:raise ValueError(mode)
