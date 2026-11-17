"""Enforce unique whole-cgroup resident charge, not a sum of shared-page RSS."""
from pathlib import Path

def measure(pids,cap,unit,proc_root=Path('/proc'),sys_root=Path('/sys/fs/cgroup')):
 line=next(x for x in (proc_root/'self/cgroup').read_text().splitlines() if x.startswith('0::'));relative=line.split('::',1)[1].lstrip('/');folder=sys_root/relative
 assert folder.name==unit and '..' not in Path(relative).parts
 current=int((folder/'memory.current').read_text());swapped=int((folder/'memory.swap.current').read_text());limit=int((folder/'memory.max').read_text());swaplimit=int((folder/'memory.swap.max').read_text());rss=0;pss=0
 for pid in pids:
  path=proc_root/str(pid)
  if not path.exists():continue
  for row in (path/'status').read_text().splitlines():
   if row.startswith('VmRSS:'):rss+=int(row.split()[1])*1024
  if (path/'smaps_rollup').exists():
   for row in (path/'smaps_rollup').read_text().splitlines():
    if row.startswith('Pss:'):pss+=int(row.split()[1])*1024
 assert limit==cap and swaplimit==0,'Cgroup hard RAM/no-swap policy'
 assert current+swapped<=cap,'Actual unique cgroup memory cap'
 return dict(cgroup=str(folder),cgroup_memory_current=current,cgroup_swap_current=swapped,cgroup_memory_max=limit,cgroup_swap_max=swaplimit,diagnostic_process_RSS_sum=rss,diagnostic_process_PSS_sum=pss,RSS_sum_not_used_as_unique_charge=True,includes_light_driver_workers_and_file_cache=True)
