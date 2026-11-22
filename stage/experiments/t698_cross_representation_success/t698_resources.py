"""New-job-only eight physical CPU admission; Phase process metadata only."""
from pathlib import Path
import os, subprocess
CPUS=[8,9,10,11,12,13,14,15]
PREFIX='/home/linghu/Documents/Codex/2026-10-31/new-chat/phase-/'

def resources():
    def physical(cpu):
        folder=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        return tuple((folder/f).read_text().strip() for f in ['physical_package_id','core_id'])
    ours={physical(c) for c in CPUS};assert len(ours)==8
    phase=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            args=(proc/'cmdline').read_bytes().split(b'\0')
            if not any(a.decode(errors='replace').startswith(PREFIX) for a in args):continue
            mask=set()
            for thread in (proc/'task').iterdir():
                try:mask.update(os.sched_getaffinity(int(thread.name)))
                except ProcessLookupError:pass
            overlap=sorted(c for c in mask if physical(c) in ours)
            phase.append(dict(pid=int(proc.name),allowed_cpus=sorted(mask),physical_overlap=overlap))
            assert not overlap,('Phase physical CPU overlap; do not launch',proc.name,overlap)
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return dict(state='PASS',stage_cpus=CPUS,stage_physical_cores=sorted(ours),phase_process_metadata_only=phase,
                CPUQuota='800%',Nice=19,CPUWeight=10,MemoryMax=8*2**30,MemorySwapMax=0,GPU='cuda0geometry_only_cpu_exact_deployment')

def gpu():
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    others=[int(x.strip()) for x in output.splitlines() if x.strip().isdigit() and int(x.strip())!=os.getpid()]
    assert not others,('GPU occupied; yield without switching device',others)
    return dict(state='PASS',device='cuda:0',other_compute_pids=[])
