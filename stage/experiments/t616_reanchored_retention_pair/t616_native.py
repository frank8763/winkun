"""Use validated native worker; label forced-prefix recovery tests explicitly."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t608_current_policy_memory'))
import t608_native as Previous
inspect=Previous.inspect
class Pool(Previous.Pool):
 def receive(self,i):
  v=super().receive(i)
  if isinstance(v,tuple):
   a,r=v
   if 'student_takeover_first' in a:r.update(forced_prefix_recovery_diagnostic=True,deployment_from_original_start=False,execution_type='forced_reference_prefix_then_single_student',teacher_action_suffix=False)
  return v
