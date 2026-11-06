"""Stopped-chain archive adapter; all original failures/models retained."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T
original=T.B.require_stopped
def read(p):return json.loads(Path(p).read_text())
def absent(pid):assert not Path('/proc',str(pid)).exists(),pid
def stopped(directory):
    if directory.name!='t391_actual_student_recovery':return original(directory)
    final=read(directory/'local_final_report.json')
    assert final['state'] in ['COMPLETE_CONDITIONAL_STUDENT_EVALUATED','COMPLETE_NO_SUCCESSFUL_RECOVERY_NO_STUDENT']
    assert final['all_native_groups_stopped']
    for name in final['completed_children']:
        child=directory/name;s=read(child/'summary.json');audit=read(child/'audit.json')
        assert s['state']=='COMPLETE' and s['all_workers_stopped'] and audit['state']=='PASS'
        absent(s['pid'])
        w=read(child/'workers.json')
        for pid in [w.get('parent'),*w.get('workers',[])]:
            if pid:absent(pid)
        for row in read(child/'local_manifest.json')['files']:
            assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256']
    for row in read(directory/'local_manifest.json')['files']:
        assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256']
if __name__=='__main__':
    T.install();T.B.require_stopped=stopped
    if sys.argv[1:]==['T391']:
        T.B.GROUPS['T391']=['t391_actual_student_recovery'];T.B.main('T391')
    elif sys.argv[1:]==['source_sync']:
        import sync_source
        sync_source.main()
    else:raise ValueError('Completed T391 evidence or source sync only')
