"""Online-only byte transport; never called by the sealed offline driver."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T
original=T.B.require_stopped
def stopped(d):
    if d.name=='t401_cold_delta_reserve':
        s=json.loads((d/'summary.json').read_text());a=json.loads((d/'audit.json').read_text());assert s['state']=='COMPLETE' and a['state']=='PASS' and s['native']==s['optimizer']==0
        assert not Path('/proc',str(s['pid'])).exists();return
    if d.name=='t402_offline_night':
        s=json.loads((d/'summary.json').read_text());a=json.loads((d/'audit.json').read_text());assert s['state']=='COMPLETE' and a['state']=='PASS' and s['all_workers_stopped']
        for name in ['summary.json','launch.json','driver_started.json']:
            assert not Path('/proc',str(json.loads((d/name).read_text())['pid'])).exists()
        for row in json.loads((d/'local_manifest.json').read_text())['files']:assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256']
        return
    return original(d)
if __name__=='__main__':
    T.install();T.B.require_stopped=stopped
    if sys.argv[1:]==['T401']:T.B.GROUPS['T401']=['t401_cold_delta_reserve'];T.B.main('T401')
    elif sys.argv[1:]==['T402']:T.B.GROUPS['T402']=['t402_offline_night'];T.B.main('T402')
    elif sys.argv[1:]==['source_sync']:
        import sync_source
        sync_source.main()
    else:raise ValueError('Only closed evidence backup or source sync')
