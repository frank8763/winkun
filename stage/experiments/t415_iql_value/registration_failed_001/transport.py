"""Back up necessary finished IQL values and auditable data/source references."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T
ORIGINAL=T.B.require_stopped
def stopped(folder):
    if folder.name!='t415_iql_value':return ORIGINAL(folder)
    read=lambda f:json.loads(Path(f).read_text());s=read(folder/'summary.json');a=read(folder/'audit.json')
    assert s['state']=='COMPLETE' and a['state']=='PASS' and not Path('/proc',str(s['pid'])).exists()
    for f,h in read(folder/'protocol.json')['inputs'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
if __name__=='__main__':
    T.install();T.B.require_stopped=stopped
    assert sys.argv[1:] in [['T415'],['source_sync']]
    if sys.argv[1]=='T415':T.B.GROUPS['T415']=['t415_iql_value'];T.B.main('T415')
    else:
        import sync_source
        sync_source.main()
