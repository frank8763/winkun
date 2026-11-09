"""Archive stopped last Stage attempt and optional lossless storage ledger."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T
def read(f):return json.loads(Path(f).read_text())
def stopped(folder):
    if folder.name=='t435_relay_reserve':
        s=read(folder/'summary.json');assert s['state']=='COMPLETE' and read(folder/'audit.json')['state']=='PASS'
        assert not Path('/proc',str(s['pid'])).exists();return
    assert folder.name=='t440_final_functional_constraint'
    assert read(folder/'SEALED.json')['all_scientific_processes_stopped'] and read(folder/'audit.json')['state']=='PASS'
    if (folder/'native_started.json').exists():
        pids=[read(folder/'native_started.json')['pid']]+read(folder/'workers.json')['workers']
        assert all(not Path('/proc',str(pid)).exists() for pid in pids)
if __name__=='__main__':
    T.install()
    if sys.argv[1]=='source_sync':
        import sync_source
        sync_source.main()
    else:
        assert sys.argv[1]=='T440';T.B.GROUPS['T440']=['t440_final_functional_constraint','t435_relay_reserve'];T.B.require_stopped=stopped
        import small_parts
        small_parts.main('T440')
