"""Stopped offline driver adapter; immutable manifests and audits required."""
from pathlib import Path
import json,sys,hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T
def read(f):return json.loads(Path(f).read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def absent(pid):assert not Path('/proc',str(pid)).exists(),pid
original=T.B.require_stopped
def stopped(directory):
    if directory.name!='t384_offline_elite_chain':return original(directory)
    d=ROOT/'stage/docs/overnight_20261105';final=read(d/'FINAL_REPORT.json')
    assert final['state']=='FINITE_OFFLINE_CHAIN_COMPLETE' and final['complete'] and final['all_child_groups_stopped']
    assert final['student']=='NOT_RUN_TEACHER_GATE' and len(final['completed'])==2
    assert not (d/'NIGHT_ERROR.json').exists()
    absent(read(directory/'launch.json')['pid']);absent(read(directory/'driver_started.json')['pid'])
    for folder in ['teacher_A','teacher_B']:
        out=directory/folder;s=read(out/'summary.json');a=read(out/'audit.json')
        assert s['state']=='COMPLETE' and s['all_workers_stopped'] and a['state']=='PASS'
        absent(s['pid']);assert not s['teacher_safe_GOAL'] and s['Actor']==s['optimizer']==0
        for row in read(out/'local_manifest.json')['files']:assert sha(ROOT/row['path'])==row['sha256']
    for row in read(directory/'local_manifest.json')['files']:assert sha(ROOT/row['path'])==row['sha256']
if __name__=='__main__':
    T.install();T.B.require_stopped=stopped
    if sys.argv[1:]==['T384-NIGHT']:
        T.B.GROUPS['T384-NIGHT']=['t384_offline_elite_chain'];T.B.main('T384-NIGHT')
    elif sys.argv[1:]==['T383-METADATA']:
        assert read(ROOT/'stage/runs/t383_cold_night_reserve/audit.json')['state']=='PASS'
        T.B.GROUPS['T383-METADATA']=['t383_cold_night_reserve'];T.B.main('T383-METADATA')
    elif sys.argv[1:]==['source_sync']:
        import sync_source
        sync_source.main()
    else:raise ValueError('Stopped verified night evidence or source sync only')
