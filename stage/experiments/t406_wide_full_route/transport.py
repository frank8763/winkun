"""RAM evidence archive for ended T406, preserving every byte locally."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'));import transport as T
original=T.B.require_stopped
def stopped(d):
    if d.name=='t406_wide_full_route':
        s=json.loads((d/'summary.json').read_text());assert s['state']=='COMPLETE' and s['all_workers_stopped'] and json.loads((d/'audit.json').read_text())['state']=='PASS' and not Path('/proc',str(s['pid'])).exists()
        for pid in json.loads((d/'workers.json').read_text())['workers']:assert not Path('/proc',str(pid)).exists()
        for r in json.loads((d/'local_manifest.json').read_text())['files']:assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256']
        return
    return original(d)
if __name__=='__main__':
    T.install();T.B.require_stopped=stopped
    if sys.argv[1]=='T406':T.B.GROUPS['T406']=['t406_wide_full_route'];T.B.main('T406')
    elif sys.argv[1]=='source_sync':
        import sync_source;sync_source.main()
    else:raise ValueError(sys.argv[1])
