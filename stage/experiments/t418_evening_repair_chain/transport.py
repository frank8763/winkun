"""RAM-only backup of stopped, independently audited child runs."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T
ORIGINAL=T.B.require_stopped
def stopped(folder):
    if folder.parent.name!='t418_evening_repair_chain':return ORIGINAL(folder)
    s=json.loads((folder/'summary.json').read_text());a=json.loads((folder/'audit.json').read_text())
    assert s['state']=='COMPLETE' and a['state']=='PASS'
    for file in ['main_started.json','workers.json']:
        value=json.loads((folder/file).read_text());pids=[value[k] for k in ['pid','parent'] if type(value.get(k)) is int]+value.get('workers',[])
        for pid in pids:assert not Path('/proc',str(pid)).exists(),pid
    for f,h in json.loads((folder/'protocol.json').read_text())['inputs'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
if __name__=='__main__':
    T.install();T.B.require_stopped=stopped
    name=sys.argv[1];assert name in ['source_sync','T418-repair','T418-protect','T418-rl']
    if name=='source_sync':
        import sync_source
        sync_source.main()
    else:
        T.B.GROUPS[name]=['t418_evening_repair_chain/'+name.split('-',1)[1]];T.B.main(name)
