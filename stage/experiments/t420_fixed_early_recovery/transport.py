"""Finished hold17 recovery RAM archive and exact source synchronization."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T
ORIGINAL=T.B.require_stopped
def stopped(folder):
    if folder.name!='t420_fixed_early_recovery':return ORIGINAL(folder)
    s=json.loads((folder/'summary.json').read_text());assert s['state']=='COMPLETE' and json.loads((folder/'audit.json').read_text())['state']=='PASS'
    for file in ['main_started.json','workers.json']:
        v=json.loads((folder/file).read_text())
        for pid in [v[k] for k in ['pid','parent'] if type(v.get(k)) is int]+v.get('workers',[]):assert not Path('/proc',str(pid)).exists()
    for f,h in json.loads((folder/'protocol.json').read_text())['inputs'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
if __name__=='__main__':
    T.install();T.B.require_stopped=stopped
    if sys.argv[1]=='T420':T.B.GROUPS['T420']=['t420_fixed_early_recovery'];T.B.main('T420')
    else:
        assert sys.argv[1]=='source_sync';import sync_source;sync_source.main()
