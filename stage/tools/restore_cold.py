"""Restore one explicitly requested historical file, never the whole archive."""
from pathlib import Path
import sys,importlib.util,json
root=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('stage_remote_history_restore',root/'stage/experiments/t413_remote_working_set/cold_store.py');cold=importlib.util.module_from_spec(spec);spec.loader.exec_module(cold)
if __name__=='__main__':
 assert len(sys.argv) in [2,3]
 print(json.dumps(cold.restore(sys.argv[1],len(sys.argv)==3 and sys.argv[2]=='--verify-only'),indent=2))
