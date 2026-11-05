"""Completed T380 raw backup and exact source synchronization."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T

if __name__=='__main__':
    T.install()
    if sys.argv[1:]==['T380']:
        assert json.loads((ROOT/'stage/runs/t380_teacher_model_audit/audit.json').read_text())['state']=='PASS'
        T.B.GROUPS['T380']=['t380_teacher_model_audit'];T.B.main('T380')
    elif sys.argv[1:]==['source_sync']:
        import sync_source
        sync_source.main()
    else:raise ValueError('Completed T380 archive or exact source synchronization only')
