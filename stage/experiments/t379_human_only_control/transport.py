"""Completed-run-only archive and source synchronization."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T

if __name__=='__main__':
    T.install()
    if sys.argv[1:]==['T379']:
        import json
        r=ROOT/'stage/runs/t379_human_only_control'
        assert json.loads((r/'audit.json').read_text())['state']=='PASS'
        T.B.GROUPS['T379']=[r.name];T.B.main('T379')
    elif sys.argv[1:]==['T378-COMPLETE']:
        import json
        r=ROOT/'stage/runs/t378_gradient_conflict'
        assert json.loads((r/'audit.json').read_text())['state']=='PASS'
        T.B.GROUPS['T378-COMPLETE']=[r.name];T.B.main('T378-COMPLETE')
    elif sys.argv[1:]==['source_sync']:
        import sync_source
        sync_source.main()
    else:raise ValueError('Completed T379/T378 closure or exact source synchronization only')
