"""Archive stopped T437 plus its separate import-recovery audit evidence."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stopped(folder):
    old=ROOT/'stage/runs/t437_joint_closed_loop_rl';rec=ROOT/'stage/runs/t438_t437_audit_recovery'
    assert folder in [old,rec]
    assert read(rec/'summary.json')['state']=='COMPLETE'
    assert read(rec/'summary.json')['original_run_all_bytes_unchanged']
    assert read(rec/'audit.json')['state']=='PASS' and read(rec/'audit.json')['historical_discarded_Adam']==128
    assert read(old/'fresh_rl/summary.json')['state']=='COMPLETE'
    pids=[read(old/f)['pid'] for f in ['launch.json','driver_started.json','fresh_rl/main_started.json']]
    pids+=read(old/'fresh_rl/workers.json')['workers']+[read(rec/'started.json')['pid']]
    assert all(not Path('/proc',str(pid)).exists() for pid in pids)
    protocol=read(rec/'protocol.json')
    assert {str(f.relative_to(ROOT)):sha(f) for f in old.rglob('*') if f.is_file()}==protocol['before_inventory']
    for path,h in read(old/'protocol.json')['inputs'].items():assert sha(path)==h,path
if __name__=='__main__':
    T.install()
    if sys.argv[1:] == ['source_sync']:
        import sync_source
        sync_source.main()
    else:
        assert sys.argv[1:]==['T437']
        T.B.GROUPS['T437']=['t437_joint_closed_loop_rl','t438_t437_audit_recovery'];T.B.require_stopped=stopped
        import small_parts
        small_parts.main('T437')
