"""Reuse RAM transport for finished Stage-only diagnostic evidence."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T
ORIGINAL=T.B.require_stopped
def stopped(folder):
    if folder.name!='t419_same_history_diagnosis':return ORIGINAL(folder)
    s=json.loads((folder/'summary.json').read_text());assert s['state']=='COMPLETE' and json.loads((folder/'audit.json').read_text())['state']=='PASS' and not Path('/proc',str(s['pid'])).exists()
    for f,h in json.loads((folder/'protocol.json').read_text())['inputs'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
if __name__=='__main__':
    T.install();T.B.require_stopped=stopped
    if sys.argv[1]=='T419':T.B.GROUPS['T419']=['t419_same_history_diagnosis'];T.B.main('T419')
    else:
        assert sys.argv[1]=='source_sync';import sync_source;sync_source.main()
