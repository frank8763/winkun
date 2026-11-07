"""Completed T412 raw bytes: RAM archive/full member SHA; no Git binaries."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as TRANSFER
ORIGINAL=TRANSFER.B.require_stopped

def stopped(directory):
    if directory.name=='t412_closed_expert_likelihood':
        summary=json.loads((directory/'summary.json').read_text())
        audit=json.loads((directory/'audit.json').read_text())
        assert summary['state']=='COMPLETE' and audit['state']=='PASS'
        pids={summary['pid']}
        for name in ['main_started.json','workers.json']:
            path=directory/name
            if not path.exists():continue
            value=json.loads(path.read_text())
            pids.update(value[key] for key in ['pid','parent'] if type(value.get(key)) is int)
            pids.update(value.get('workers',[]))
        for pid in pids:assert not Path('/proc',str(pid)).exists(),('Live or reused native PID',pid)
        for row in json.loads((directory/'local_manifest.json').read_text())['files']:
            assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256']
        return
    return ORIGINAL(directory)

if __name__=='__main__':
    TRANSFER.install();TRANSFER.B.require_stopped=stopped
    if sys.argv[1]=='source_sync':
        import sync_source
        sync_source.main()
    else:
        assert sys.argv[1]=='T412'
        TRANSFER.B.GROUPS['T412']=['t412_closed_expert_likelihood']
        TRANSFER.B.main('T412')
