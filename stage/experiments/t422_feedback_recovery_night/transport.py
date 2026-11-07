"""Optional RAM archive after all scientific files are sealed and stopped."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T
ORIGINAL=T.B.require_stopped
def stopped(folder):
    if folder.name!='t422_feedback_recovery_night':return ORIGINAL(folder)
    read=lambda f:json.loads(Path(f).read_text())
    assert read(folder/'SEALED.json')['all_scientific_processes_stopped']
    for child in ['search','learning']:
        d=folder/child
        if not (d/'main_started.json').exists():continue
        ids=[read(d/'main_started.json')['pid']]
        if (d/'workers.json').exists():ids+=read(d/'workers.json').get('workers',[])
        for pid in ids:assert not Path('/proc',str(pid)).exists(),pid
        if (d/'summary.json').exists() and read(d/'summary.json')['state']=='COMPLETE':assert read(d/'audit.json')['state']=='PASS'
    for file,h in read(folder/'protocol.json')['inputs'].items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==h,file
if __name__=='__main__':
    assert sys.argv[1:] in [['T422'],['source_sync']]
    T.install();T.B.require_stopped=stopped
    if sys.argv[1]=='T422':
        T.B.GROUPS['T422']=['t422_feedback_recovery_night'];import small_parts;small_parts.main('T422')
    else:
        import sync_source;sync_source.main()
