"""Closed immutable T397 bytes; RAM archive and exact source synchronization."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T
original=T.B.require_stopped
def stopped(d):
    if d.name!='t397_handoff_duration':return original(d)
    s=json.loads((d/'summary.json').read_text());a=json.loads((d/'audit.json').read_text())
    assert s['state']=='COMPLETE' and a['state']=='PASS' and s['all_native_sessions_closed'] and s['all_workers_stopped']
    assert s['Actor_updates']==s['optimizer_updates']==s['backward_calls']==0
    for f in ['summary.json','workers.json']:
        v=json.loads((d/f).read_text())
        for pid in [v.get('pid'),v.get('parent'),*v.get('workers',[])]:
            if pid:assert not Path('/proc',str(pid)).exists()
    for r in json.loads((d/'local_manifest.json').read_text())['files']:assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256']
if __name__=='__main__':
    T.install();T.B.require_stopped=stopped
    if sys.argv[1:]==['T397']:T.B.GROUPS['T397']=['t397_handoff_duration'];T.B.main('T397')
    elif sys.argv[1:]==['source_sync']:
        import sync_source
        sync_source.main()
    else:raise ValueError('Closed T397 evidence or exact source only')
