"""Post-seal RAM-only archive and exact source sync; no scientific network need."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T

def stopped(folder):
    read=lambda f:json.loads(Path(f).read_text())
    if folder.name=='t435_relay_reserve':
        assert read(folder/'summary.json')['state']=='COMPLETE' and read(folder/'audit.json')['state']=='PASS';return
    assert folder.name=='t434_night_relay' and read(folder/'SEALED.json')['all_scientific_processes_stopped']
    for name in ['fresh_rl','iterative_recovery','representation_rl','gate','eight']:
        d=folder/name
        if not (d/'main_started.json').exists():continue
        ids=[read(d/'main_started.json')['pid']]+read(d/'workers.json')['workers']
        assert all(not Path('/proc',str(pid)).exists() for pid in ids)
        if (d/'summary.json').exists() and read(d/'summary.json')['state']=='COMPLETE':assert read(d/'audit.json')['state']=='PASS'
    for path,h in read(folder/'protocol.json')['inputs'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
if __name__=='__main__':
    assert sys.argv[1:] in [['T434'],['source_sync']];T.install()
    if sys.argv[1]=='T434':
        T.B.require_stopped=stopped;T.B.GROUPS['T434']=['t434_night_relay','t435_relay_reserve'];import small_parts;small_parts.main('T434')
    else:
        import sync_source;sync_source.main()
