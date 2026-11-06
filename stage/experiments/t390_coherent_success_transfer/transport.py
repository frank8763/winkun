"""Transport-only finalization; no simulation, fitting or evidence deletion."""
from pathlib import Path
import json,sys,hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T
original=T.B.require_stopped
def stopped(directory):
    if directory.name=='t390_coherent_success_transfer':
        assert json.loads((directory/'audit.json').read_text())['state']=='PASS'
        return original(directory)
    if directory.name!='t389_transfer_contract':return original(directory)
    s=json.loads((directory/'summary.json').read_text())
    assert s['state']=='PASS' and s['native']==s['optimizer']==s['geometry']==s['backward']==0
    assert not Path('/proc',str(s['pid'])).exists()
    for p,h in s['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
if __name__=='__main__':
    T.install();T.B.require_stopped=stopped
    if sys.argv[1:]==['T389-T390']:
        T.B.GROUPS['T389-T390']=['t389_transfer_contract','t390_coherent_success_transfer'];T.B.main('T389-T390')
    elif sys.argv[1:]==['source_sync']:
        import sync_source
        sync_source.main()
    else:raise ValueError('Completed T389/T390 or source sync only')
