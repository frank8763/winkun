"""Stopped frozen acceptance, RAM-only evidence backup and exact source sync."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T
def stopped(folder):
    assert folder.name=='t424_frozen_six_acceptance'
    read=lambda f:json.loads(Path(f).read_text())
    summary=read(folder/'summary.json');assert summary['state']=='COMPLETE' and summary['all_workers_stopped'] and summary['all_sessions_closed']
    assert read(folder/'audit.json')['state']=='PASS_FROZEN_ACCEPTANCE_EVIDENCE_ONLY'
    ids=[read(folder/'main_started.json')['pid'],*read(folder/'workers.json')['workers']]
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    for f,h in read(folder/'protocol.json')['inputs'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
if __name__=='__main__':
    assert sys.argv[1:] in [['T424'],['source_sync']]
    T.install()
    if sys.argv[1]=='T424':
        T.B.require_stopped=stopped;T.B.GROUPS['T424']=['t424_frozen_six_acceptance'];import small_parts;small_parts.main('T424')
    else:
        import sync_source;sync_source.main()
