"""Archive only completed, independently audited native teacher evidence."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T
if __name__=='__main__':
    T.install()
    if sys.argv[1:]==['T381']:
        r=ROOT/'stage/runs/t381_native_mppi_teacher'
        assert json.loads((r/'audit.json').read_text())['state']=='PASS'
        T.B.GROUPS['T381']=[r.name];T.B.main('T381')
    elif sys.argv[1:]==['source_sync']:
        import sync_source
        sync_source.main()
    else:raise ValueError('T381 archive or exact source synchronization only')
