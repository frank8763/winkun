"""One independent frozen diagnostic, audit, manifest and bounded finalization."""
from pathlib import Path
import os
import signal
import subprocess
import sys
import time
import traceback
import control as R

ROOT,HERE,OUT,DOC,read,save,sha=R.ROOT,R.HERE,R.OUT,R.DOC,R.read,R.save,R.sha


def main():
    p=read(OUT/'protocol.json');R.setup();R.R.verify(p,True)
    assert not (OUT/'launch.json').exists()
    start=time.time();end=p['absolute_all_stop']
    assert start+p['main_seconds']+p['audit_seconds']+p['transport_seconds']+p['source_seconds']+30<end
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    save(OUT/'launch.json',dict(pid=os.getpid(),epoch=start,source_remote=receipt['remote_commit'],
        root_tree=receipt['root_tree'],protocol_sha256=sha(OUT/'protocol.json'),
        all_stop=end,CPU=[14,15],CPUQuota='200%',GPU=False,new_Adam_cap=0,new_RL_cap=0))
    ids=[];error=None
    def call(action,cap):
        with (OUT/(action+'.log')).open('xb') as log:
            c=subprocess.Popen(['taskset','-c','14,15',str(ROOT/'run_stage.sh'),str(HERE/'control.py'),action],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            ids.append(c.pid);save(DOC/'STATUS.json',dict(state=action,pid=c.pid,driver_pid=os.getpid(),epoch=time.time()))
            try:c.wait(timeout=cap);assert c.returncode==0,(action,c.returncode)
            except BaseException:
                try:os.killpg(c.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:c.wait(timeout=2)
                except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
                raise
    try:
        call('main',p['main_seconds']);ids+=R.R.G.stopped(OUT)
        call('audit',p['audit_seconds']);assert read(OUT/'audit.json')['state']=='PASS'
        files=sorted(f for f in OUT.rglob('*') if f.is_file() and f.name not in ['driver.log','local_manifest.json'] and f.suffix not in ['.pyc','.lock'])
        members=[dict(path=str(f.relative_to(ROOT)),sha256=sha(f),bytes=f.stat().st_size) for f in files]
        save(OUT/'local_manifest.json',dict(state='SEALED_LOCAL_ALL_MEMBER_SHA',files=members))
        assert all(sha(ROOT/r['path'])==r['sha256'] for r in members)
    except BaseException:
        error=traceback.format_exc();save(DOC/'DRIVER_ERROR.json',dict(error=error,epoch=time.time(),no_restart=True))
    gone=all(not Path('/proc',str(pid)).exists() for pid in ids)
    save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=gone,scientific_pids=ids,epoch=time.time()))
    report=dict(state='COMPLETE' if error is None else 'FAILED',error=error,
        summary=read(OUT/'summary.json') if (OUT/'summary.json').exists() else None,
        all_scientific_processes_stopped=gone,new_Adam_cap=0,new_RL_cap=0,remote='PENDING',
        known_success_data_completion=True,no_global_model_promotion=True,epoch=time.time())
    save(DOC/'FINAL_REPORT.json',report)
    if gone and time.time()+p['transport_seconds']+p['source_seconds']+20<end:
        try:
            c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'T589'],cwd=ROOT,capture_output=True,timeout=p['transport_seconds'])
            assert c.returncode==0,c.stdout.decode(errors='replace')[-1000:]+c.stderr.decode(errors='replace')[-1000:]
            report['remote']=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T589.json')['state']
        except BaseException:save(DOC/'REMOTE_PENDING.json',dict(error=traceback.format_exc(),epoch=time.time()))
    save(DOC/'FINAL_REPORT.json',report)
    text='T589已知成功任务完整历史补齐结束：'+report['state']+'；12项完整成功历史复验见summary；remote='+report['remote']+'。不称完整407或泛化通过。\n'
    (DOC/'FINAL_REPORT.md').write_text(text)
    status=ROOT/'stage/docs/RESEARCH_STATUS.md';status.write_text('2026-11-16 '+text+'\n'+status.read_text())
    try:
        paths=[str(DOC.relative_to(ROOT)),'stage/docs/RESEARCH_STATUS.md']
        for name in ['source_manifest_T589.json','release_receipt_T589.json']:
            f=ROOT/'stage/docs/long_distance_20261101'/name
            if f.exists():paths.append(str(f.relative_to(ROOT)))
        subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True,capture_output=True)
        subprocess.run(['git','commit','-m','Report T589 missing known-success full-history capture'],cwd=ROOT,check=True,capture_output=True)
        if time.time()+p['source_seconds']+5<end:
            c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'source_sync'],cwd=ROOT,capture_output=True,timeout=p['source_seconds'])
            assert c.returncode==0
    except BaseException:save(DOC/'SOURCE_PENDING.json',dict(error=traceback.format_exc(),epoch=time.time()))


if __name__=='__main__':main()
