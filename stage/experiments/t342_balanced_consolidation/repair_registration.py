"""Zero-native registration repair for an unlaunched preflight SemLock error."""
from t342_core import *

def main():
    assert not (OUT/'launch.json').exists() and not (OUT/'status.json').exists() and not (OUT/'preflight.json').exists()
    old=read(HERE/'preflight_failed_001/protocol.json');assert read(HERE/'protocol.json')==old==read(OUT/'protocol.json')
    ledger=read(HERE/'preflight_failed_001/ledger.json');assert ledger['physics_steps']==ledger['geometry_commands']==0 and ledger['discarded_synthetic_Adam_calls_completed']==4
    p=copy.deepcopy(old);p['preflight_discarded_Adam']=8;p['preflight_engineering_repair']='Retain shared_counts strong reference through spawned worker exit; original4+fresh4 discarded clones,0native'
    for f in [*HERE.glob('*.py'),DOC/'T342_PLAN.md',HERE/'preflight_failed_001/ledger.json']:
        p['input_hashes'][str(f)]=K.C.sha(f)
    assert all(p[k]==old[k] for k in old if k not in ['input_hashes','preflight_discarded_Adam'])
    save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);print('T342 prelaunch engineering repair registered0native/optimizer; original failed files preserved')

if __name__=='__main__':main()
