"""Same T649 episodic PG, canonical geometry and task-scoped field boundary."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t649_fresh_outcome_pg'))
import t649_learning as Prior

def outcome(row):
    boundary_applies='fullfield_start_pose' in row['request']['job']
    return float(row['success'] and not row['collision'] and (not boundary_applies or not row.get('field_boundary',False)))

Prior.outcome=outcome
L,P,E,torch,np=Prior.L,Prior.P,Prior.E,Prior.torch,Prior.np
state,load,export,make,fit=Prior.state,Prior.load,Prior.export,Prior.make,Prior.fit
read,save,sha,digest=Prior.read,Prior.save,Prior.sha,Prior.digest
