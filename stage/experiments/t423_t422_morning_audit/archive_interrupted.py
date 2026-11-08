"""Archive original stopped T422 interruption evidence without fake summaries.

Zero simulation/learning/deletion. The completed search audit and interrupted
learning are kept as different factual states. No original output is changed.
"""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T
OUT=ROOT/'stage/runs/t422_feedback_recovery_night'
def stopped(folder):
    assert folder==OUT
    read=lambda f:json.loads(Path(f).read_text())
    assert read(folder/'search/summary.json')['state']=='COMPLETE'
    assert read(folder/'search/audit.json')['state']=='PASS'
    assert not (folder/'learning/summary.json').exists()
    assert not (folder/'SEALED.json').exists()
    ids=set()
    for name in ['launch.json','driver_started.json','search/main_started.json','learning/main_started.json','learning/workers.json']:
        row=read(folder/name)
        ids.update(row[k] for k in ['pid','parent'] if type(row.get(k)) is int)
        ids.update(row.get('workers',[]))
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    for path,digest in read(folder/'protocol.json')['inputs'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    events=[json.loads(line) for line in (folder/'learning/optimizer_calls.jsonl').read_text().splitlines()]
    assert events[-1]['tag']=='BC3' and events[-1]['which']=='shadow' and events[-1]['where']=='AFTER' and events[-1]['k']==150
    assert sum(e['where']=='AFTER' for e in events)==1567
if __name__=='__main__':
    assert sys.argv[1:] in [['archive'],['source_sync']]
    T.install()
    if sys.argv[1]=='archive':
        T.B.require_stopped=stopped;T.B.GROUPS['T422-INTERRUPTED']=['t422_feedback_recovery_night']
        import small_parts
        small_parts.main('T422-INTERRUPTED')
    else:
        import sync_source
        sync_source.main()
