"""Explicit post-run transport; never imported/called by the offline driver."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as TRANSFER
ORIGINAL=TRANSFER.B.require_stopped
def stopped(directory):
    if directory.name=='t409_cold_npz_reserve':
        s=json.loads((directory/'summary.json').read_text());a=json.loads((directory/'audit.json').read_text());assert s['state']=='COMPLETE' and a['state']=='PASS' and a['all_original_NPZ_archive_bytes_exact'];assert not Path('/proc',str(s['pid'])).exists();return
    if directory.name=='t410_day_offline_chain':
        r=json.loads((ROOT/'stage/docs/day_20261107/FINAL_REPORT.json').read_text());assert r['all_owned_children_stopped'] and not Path('/proc',str(r['pid'])).exists()
        for row in r['completed']:
            if 'manifest' not in row:continue
            for f in json.loads(Path(row['manifest']['path']).read_text())['files']:assert hashlib.sha256((ROOT/f['path']).read_bytes()).hexdigest()==f['sha256']
        return
    return ORIGINAL(directory)
if __name__=='__main__':
    TRANSFER.install();TRANSFER.B.require_stopped=stopped
    if sys.argv[1]=='source_sync':
        import sync_source
        sync_source.main()
    else:
        group=sys.argv[1];assert group in ['T409','T410'];TRANSFER.B.GROUPS[group]=['t409_cold_npz_reserve' if group=='T409' else 't410_day_offline_chain'];TRANSFER.B.main(group)
