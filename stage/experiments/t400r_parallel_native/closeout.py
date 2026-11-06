"""Close zero-native engineering failures; no training, no simulator startup."""
from t400r_contract import *

def main():
    p=read(OUT/'protocol.json');verify(p);r=read(OUT/'not_started.json')
    assert r['native']==r['optimizer']==r['backward']==0
    assert not (OUT/'launch.json').exists() and not (OUT/'main_started.json').exists()
    failed=ROOT/'stage/runs/t400_parallel_native';s=read(failed/'summary.json')
    assert s['state']=='FAILED' and s['physics']==s['commands']==0
    for name in ['launch.json','driver_started.json','main_started.json']:
        assert not Path('/proc',str(read(failed/name)['pid'])).exists()
    for row in read(failed/'local_manifest.json')['files']:assert sha(ROOT/row['path'])==row['sha256']
    N.manifest(OUT)
    sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
    import transport as T
    original=T.B.require_stopped
    def stopped(directory):
        if directory.name=='t400_parallel_native':
            ss=read(directory/'summary.json');assert ss['state']=='FAILED' and ss['physics']==ss['commands']==0
            for name in ['launch.json','driver_started.json','main_started.json']:assert not Path('/proc',str(read(directory/name)['pid'])).exists()
        elif directory.name=='t400r_parallel_native':
            rr=read(directory/'not_started.json');assert rr['native']==rr['optimizer']==0 and not (directory/'launch.json').exists()
        else:return original(directory)
        for row in read(directory/'local_manifest.json')['files']:assert sha(ROOT/row['path'])==row['sha256']
    T.install();T.B.require_stopped=stopped;T.B.GROUPS['T400R']=['t400_parallel_native','t400r_parallel_native']
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T400R.json';assert not receipt.exists()
    T.B.main('T400R')

if __name__=='__main__':main()
