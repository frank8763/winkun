"""Reuse verified RAM-only transport, registering this completed group in memory."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
import transport as T

if __name__=='__main__':
    T.install()
    if sys.argv[1:]==['T378']:
        T.B.GROUPS['T378']=['t378_gradient_conflict']
        T.B.main('T378')
    elif sys.argv[1:]==['source_sync']:
        import sync_source
        sync_source.main()
    else:
        raise ValueError('T378 backup or exact source synchronization only')
