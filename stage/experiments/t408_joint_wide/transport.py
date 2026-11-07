"""RAM-only completed T408 evidence archive and exact source synchronization."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T
original=T.B.require_stopped

def stopped(directory):
    if directory.name=='t408_joint_wide':
        summary=json.loads((directory/'summary.json').read_text())
        assert summary['state']=='COMPLETE' and summary['all_workers_stopped'] and summary['all_native_sessions_closed']
        assert json.loads((directory/'audit.json').read_text())['state']=='PASS'
        pids={summary['pid']}
        workers=json.loads((directory/'workers.json').read_text())
        pids.add(workers['parent']);pids.update(workers['workers'])
        for pid in pids:
            assert not Path('/proc',str(pid)).exists(),('Live or reused native PID',pid)
        for row in json.loads((directory/'local_manifest.json').read_text())['files']:
            assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256']
        return
    if directory.name=='t407_cold_exact_reserve':
        summary=json.loads((directory/'summary.json').read_text());audit=json.loads((directory/'audit.json').read_text())
        assert summary['state']=='COMPLETE' and summary['native']==summary['Adam']==0 and summary['unique_bytes_removed']==0
        assert audit['state']=='PASS' and audit['all_original_bytes_local_canonical_and_remote']
        assert not Path('/proc',str(summary['pid'])).exists()
        return
    return original(directory)

if __name__=='__main__':
    T.install();T.B.require_stopped=stopped
    if sys.argv[1] in ['T408','T407']:
        group=sys.argv[1];T.B.GROUPS[group]=['t408_joint_wide' if group=='T408' else 't407_cold_exact_reserve'];T.B.main(group)
    elif sys.argv[1]=='source_sync':
        import sync_source
        sync_source.main()
    else:
        raise ValueError(sys.argv[1])
