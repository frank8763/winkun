"""Explicit completed raw-byte RAM transport."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T
ORIGINAL=T.B.require_stopped
def stopped(d):
    if d.name=='t411_closed_teacher_recovery':
        s=json.loads((d/'summary.json').read_text());assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists() and json.loads((d/'audit.json').read_text())['state']=='PASS'
        for r in json.loads((d/'local_manifest.json').read_text())['files']:assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256']
        return
    return ORIGINAL(d)
if __name__=='__main__':
    T.install();T.B.require_stopped=stopped
    if sys.argv[1]=='source_sync':
        import sync_source
        sync_source.main()
    else:assert sys.argv[1]=='T411';T.B.GROUPS['T411']=['t411_closed_teacher_recovery'];T.B.main('T411')
