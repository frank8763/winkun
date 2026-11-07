"""RAM-only archive of a stopped, audited terminal-capture pilot."""
from pathlib import Path
import sys,json,hashlib

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T

ORIGINAL=T.B.require_stopped

def stopped(folder):
    if folder.name!='t421_terminal_capture':return ORIGINAL(folder)
    read=lambda f:json.loads(Path(f).read_text())
    summary=read(folder/'summary.json');audit=read(folder/'audit.json')
    assert summary['state']=='COMPLETE' and summary['all_sessions_closed'] is True and audit['state']=='PASS'
    assert summary.get('Actor_updates',0)==summary.get('optimizer_steps',0)==0
    ids=set()
    for name in ['main_started.json','summary.json']:
        value=read(folder/name)
        ids.update(value[key] for key in ['pid','main_pid'] if type(value.get(key)) is int)
    assert ids
    for pid in ids:assert not Path('/proc',str(pid)).exists(),('Live or reused native PID',pid)
    for file,digest in read(folder/'protocol.json')['inputs'].items():
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest,file

if __name__=='__main__':
    assert sys.argv[1:] in [['T421'],['source_sync']]
    T.install();T.B.require_stopped=stopped
    if sys.argv[1]=='T421':
        T.B.GROUPS['T421']=['t421_terminal_capture'];T.B.main('T421')
    else:
        import sync_source
        sync_source.main()
