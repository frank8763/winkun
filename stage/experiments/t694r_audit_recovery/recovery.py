"""Recover only the uncommitted native audit; no fit or simulator can run.

The original audit predicates and rows are reused verbatim. Only the manifest
verification schedule changes: complete before/after, lightweight between rows.
"""
from pathlib import Path
import hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t694r_audit_recovery';DOC=ROOT/'stage/docs/weekend_20261120_23/t694r'
PARENT=ROOT/'stage/runs/t694_full_history_compatibility'
sys.path.insert(0,str(ROOT/'stage/experiments/t694_full_history_compatibility'))
import t694_runtime as Old
sys.path.insert(0,str(HERE))
import t694r_wait as Wait
from status_io import save
read,sha=Old.read,Old.sha


def parent_plan():
    root=read(PARENT/'protocol.json');child=read(PARENT/'evaluation/protocol.json')
    assert child['_root_sha256']==sha(PARENT/'protocol.json')
    return dict(root,**child)


class ManifestGate:
    def __init__(self,p,deadline,probe):
        self.p=p;self.deadline=deadline;self.probe=probe;self.last=0.;self.full_calls=0;self.calls=0;self.snapshots={}
    def light(self):
        self.calls+=1;assert time.time()<self.deadline-5
        if time.time()-self.last>5:self.last=time.time();self.probe()
    def full(self):
        self.light();self.full_calls+=1
        assert sha(ROOT/'stage/experiments/t694_full_history_compatibility/protocol.json')==sha(PARENT/'protocol.json')
        current={}
        for f,h in self.p['inputs'].items():
            assert sha(f)==h,f
            stat=Path(f).stat();current[f]=(stat.st_ino,stat.st_size,stat.st_mtime_ns)
        if self.snapshots:assert self.snapshots==current
        self.snapshots=current
    def check(self,*unused):
        if self.full_calls==0:self.full()
        else:self.light()
    def finish(self):
        self.full();assert self.full_calls==2


def actual_stopped():
    report=read(PARENT/'FINAL_REPORT.json');seal=read(PARENT/'SEALED.json')
    assert report['state']=='FAILED' and seal['all_scientific_processes_stopped']
    pids=seal['scientific_pids']+[report['helper_pid'],read(PARENT/'DRIVER_STARTED.json')['pid']]
    assert all(not Path('/proc',str(pid)).exists() for pid in pids)
    assert read(PARENT/'fit/audit.json')['state']=='PASS'
    assert read(PARENT/'evaluation/summary.json')['state']=='COMPLETE'
    assert not (PARENT/'evaluation/audit.json').exists()


def audit():
    root=read(OUT/'protocol.json');assert root['authorization_approved']
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json');actual_stopped()
    assert time.time()<root['absolute_all_stop']
    os.sched_setaffinity(0,{14});Old.torch.set_num_threads(1);Old.torch.use_deterministic_algorithms(True)
    def forbidden(*args,**kw):raise AssertionError('Audit recovery forbids native and optimizer stepping')
    Old.torch.optim.Adam.step=forbidden
    import mujoco
    mujoco.mj_step=forbidden
    p=parent_plan();gate=ManifestGate(p,min(time.time()+root['audit_seconds'],root['absolute_all_stop']),Wait.resources)
    captured=[];real_save=Old.save
    def redirected(path,value):
        assert Path(path)==PARENT/'evaluation/audit.json'
        captured.append(value)
    # No original code/protocol/output is edited. All scientific assertions of
    # the unfinished audit run inside the exact historical function.
    Old.OUT=PARENT/'evaluation';Old.DOC=Old.ROOTDOC/'evaluation'
    Old.plan=lambda:p;Old.verify=gate.check;Old.T.verify=gate.check;Old.save=redirected
    started=time.time()
    try:
        Old.evaluate_audit();gate.finish()
    finally:Old.save=real_save
    assert len(captured)==1 and captured[0]['state']=='PASS'
    assert not (PARENT/'evaluation/audit.json').exists()
    result=dict(captured[0],recovered_original_audit_without_fit_or_native_reexecution=True,
        original_failed_root_preserved=True,original_root=str(PARENT),original_summary_sha256=sha(PARENT/'evaluation/summary.json'),
        full_manifest_passes=gate.full_calls,light_checks=gate.calls,full_input_files=len(p['inputs']),
        source_hashes_and_metadata_unchanged_at_both_endpoints=True,elapsed_seconds=time.time()-started,
        new_Actor=0,new_Value=0,new_optimizer=0,new_native=0,new_commands=0)
    save(OUT/'audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='full407'}),flush=True)


if __name__=='__main__':audit()
