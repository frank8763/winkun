"""Deterministic red/green reproduction of shared-RSS false rejection; zero native."""
from pathlib import Path
import tempfile,ast,json,time
from t628s_memory import measure
ROOT=Path(__file__).resolve().parents[3];DOC=ROOT/'stage/docs/day_20261117/t628s'
def main():
 with tempfile.TemporaryDirectory(prefix='memory_fixture_',dir=DOC) as tmp:
  f=Path(tmp);proc=f/'proc';cg=f/'cgroup/stage-t628s.service';(proc/'self').mkdir(parents=True);cg.mkdir(parents=True);(proc/'self/cgroup').write_text('0::/stage-t628s.service\n')
  for name,val in {'memory.current':3072*2**20,'memory.swap.current':0,'memory.max':2**32,'memory.swap.max':0}.items():(cg/name).write_text(str(val))
  for pid in [10,20,30,40]:
   p=proc/str(pid);p.mkdir();(p/'status').write_text('VmRSS:\t1500000 kB\n');(p/'smaps_rollup').write_text('Pss:\t760000 kB\n')
  old=ROOT/'stage/experiments/t628_fresh_noise_curriculum/t628_runtime.py';tree=ast.parse(old.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='ram_check');code=ast.get_source_segment(old.read_text(),node);ns={'Path':lambda *parts:proc.joinpath(*parts[1:])};exec(code,ns)
  red=False
  try:ns['ram_check']({'working_RAM_cap_bytes':2**32},[10,20,30,40])
  except AssertionError:red=True
  assert red,'Original real guard must reproduce false rejection'
  green=measure([10,20,30,40],2**32,'stage-t628s.service',proc,f/'cgroup');assert green['diagnostic_process_RSS_sum']>2**32>green['cgroup_memory_current']
  (cg/'memory.current').write_text(str(2**32+1));over=False
  try:measure([10,20,30,40],2**32,'stage-t628s.service',proc,f/'cgroup')
  except AssertionError:over=True
  assert over,'Actual over-cap memory must still reject'
  (cg/'memory.current').write_text(str(3072*2**20));(cg/'memory.swap.max').write_text('max');swap=False
  try:measure([10,20,30,40],2**32,'stage-t628s.service',proc,f/'cgroup')
  except (AssertionError,ValueError):swap=True
  assert swap
 report=dict(state='PASS_RED_OLD_GUARD_GREEN_UNIQUE_CGROUP',old_real_guard_false_rejection_reproduced=True,new_guard_uses_unique_cgroup_not_sum_RSS=True,real_over_cap_still_rejected=True,swap_policy_enforced=True,fixture=green,new_native=0,new_optimizer=0,epoch=time.time());(DOC/'MEMORY_PREFLIGHT.json').write_text(json.dumps(report,indent=2)+'\n');print('T628S_MEMORY_OLD_RED_NEW_GREEN_OVER_CAP_RED_PASS')
if __name__=='__main__':main()
