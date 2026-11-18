"""One bounded main, then independent audit; standard-library supervisor."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import time
import traceback
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t632_taskwise_recovery'
DOC=ROOT/'stage/docs/day_20261118/t632'


def save(f,v):Path(f).write_text(json.dumps(v,indent=2)+'\n')
def read(f):return json.loads(Path(f).read_text())


def main():
    p=read(OUT/'protocol.json')
    assert p['authorization_approved'] and (OUT/'protocol.json').read_bytes()==(HERE/'protocol.json').read_bytes()
    assert time.time()+p['main_seconds']+p['audit_seconds']+120<p['absolute_all_stop']
    assert not (OUT/'launch.json').exists() and not (OUT/'DRIVER_STARTED.json').exists()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    for f,h in p['inputs'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
    save(OUT/'DRIVER_STARTED.json',dict(pid=os.getpid(),epoch=time.time(),source_remote=receipt['remote_commit']))
    error=None;children=[]
    for action,seconds in [('main',p['main_seconds']),('audit',p['audit_seconds'])]:
        try:
            assert time.time()+seconds+60<p['absolute_all_stop']
            with (OUT/(action+'.log')).open('xb') as log:
                c=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'t632_runtime.py'),action],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8'))
                children.append(c.pid)
                try:c.wait(timeout=seconds);assert c.returncode==0,(action,c.returncode)
                except BaseException:
                    try:os.killpg(c.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:c.wait(timeout=3)
                    except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
                    raise
        except BaseException:
            error=traceback.format_exc();break
    if (OUT/'worker_generations.json').exists():
        for g in read(OUT/'worker_generations.json'):children.extend([g['parent'],*g['workers']])
    gone=all(not Path('/proc',str(i)).exists() for i in children)
    save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=gone,scientific_pids=children,epoch=time.time()))
    summary=read(OUT/'summary.json') if (OUT/'summary.json').exists() else None
    report=dict(state='FAILED' if error else 'COMPLETE',error=error,summary=summary,
                all_scientific_processes_stopped=gone,
                new_optimizer=(summary or {}).get('Actor_updates_started',{}).get('live',0),
                completed_optimizer=(summary or {}).get('Actor_updates_completed',{}).get('live',0),
                remote='PENDING_LOCAL_RAW_RETAINED',no_navigation_promotion=True)
    # Include failed producer counts directly: never omit a failed final block.
    save(OUT/'FINAL_REPORT.json',report);save(DOC/'FINAL_REPORT.json',report)
    files=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,
                sha256=hashlib.sha256(f.read_bytes()).hexdigest())
           for f in sorted(OUT.rglob('*')) if f.is_file() and f.name!='local_manifest.json']
    save(OUT/'local_manifest.json',dict(state='LOCAL_RAW_SHA_ONLY',files=files))
    # Science is sealed before any upload. Network failure cannot erase local
    # data or restart a producer. One attempt, inside the approved all-stop.
    if gone and time.time()+p['transport_seconds']+p['source_seconds']+60<p['absolute_all_stop']:
        try:
            c=subprocess.run([str(ROOT/'run_stage.sh'),str(HERE/'transport.py')],
                cwd=ROOT,capture_output=True,timeout=p['transport_seconds'],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
            assert c.returncode==0,c.stderr.decode(errors='replace')[-1200:]
            report['remote']=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T632.json')['state']
            report['previous_T629_remote']=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T629.json')['state']
        except BaseException:
            report['backup_error']=traceback.format_exc()
            receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T632.json'
            if receipt.exists():report['remote']=read(receipt)['state']
    save(DOC/'FINAL_REPORT.json',report)
    f=ROOT/'stage/docs/RESEARCH_STATUS.md'
    f.write_text('2026-11-18 T632分任务恢复学习结束：'+report['state']+
        '，注册合格恢复轨迹='+str(len(p['field_recovery_rows']))+'，Actor更新='+str(report['new_optimizer'])+'，remote='+report['remote']+
        '；非全局/泛化晋升。详day_20261118/t632/FINAL_REPORT。\n\n'+f.read_text())
    try:
        paths=[str(DOC.relative_to(ROOT)),'stage/docs/RESEARCH_STATUS.md']
        for tag in ['T629','T632']:
            paths.extend(str(x.relative_to(ROOT)) for pattern in [f'release_receipt_{tag}*.json',f'source_manifest_{tag}*.json']
                for x in (ROOT/'stage/docs/long_distance_20261101').glob(pattern))
        subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True,capture_output=True)
        subprocess.run(['git','commit','-m','Report bounded T632 taskwise learning and full raw backup'],cwd=ROOT,check=True,capture_output=True)
        if time.time()+p['source_seconds']+15<p['absolute_all_stop']:
            subprocess.run([str(ROOT/'run_stage.sh'),str(ROOT/'stage/experiments/t629_offline_night_relay/t629_transport.py'),'source_sync'],
                cwd=ROOT,check=True,capture_output=True,timeout=p['source_seconds'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    except BaseException:save(DOC/'SOURCE_PENDING.json',dict(error=traceback.format_exc(),raw_preserved=True))


if __name__=='__main__':main()
