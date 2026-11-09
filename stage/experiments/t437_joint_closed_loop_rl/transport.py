"""Completed Stage T436/T437 evidence transport, RAM-only archives."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T

def stopped(folder):
    read=lambda f:json.loads(Path(f).read_text())
    if folder.name=='t436_frozen_failure_trace':
        assert read(folder/'summary.json')['state']=='COMPLETE' and read(folder/'audit.json')['state']=='PASS'
        pids=[read(folder/'main_started.json')['pid']]+read(folder/'workers.json')['workers']
        assert all(not Path('/proc',str(x)).exists() for x in pids)
        assert not Path('/proc',str(read(folder/'launch.json')['pid'])).exists()
    else:
        assert folder.name=='t437_joint_closed_loop_rl'
        assert read(folder/'SEALED.json')['all_scientific_processes_stopped']
        child=folder/'fresh_rl';assert read(child/'audit.json')['state']=='PASS'
        pids=[read(child/'main_started.json')['pid']]+read(child/'workers.json')['workers']
        assert all(not Path('/proc',str(x)).exists() for x in pids)

if __name__=='__main__':
    T.install();T.B.require_stopped=stopped
    if sys.argv[1]=='source_sync':
        import sync_source
        sync_source.main()
    else:
        group=sys.argv[1];assert group in ['T436','T437']
        T.B.GROUPS[group]=['t436_frozen_failure_trace' if group=='T436' else 't437_joint_closed_loop_rl']
        import small_parts
        small_parts.main(group)
