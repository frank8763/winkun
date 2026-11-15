"""Single bounded sine student fit, conditional native panel, audit and RAM backup."""
from pathlib import Path
import os,signal,subprocess,sys,time,traceback
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t564_runtime as R
ROOT,OUT,DOC,read,save,sha=R.ROOT,R.OUT,R.DOC,R.read,R.save,R.sha

def main():
    p=R.configuration();R.verify(p,True);assert not (OUT/'launch.json').exists();start=time.time();end=p['absolute_all_stop']
    assert start+p['main_seconds']+p['audit_seconds']+p['transport_seconds']+p['source_seconds']+30<end
    assert start+p['main_seconds']<p['absolute_native_stop']
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    save(OUT/'launch.json',dict(pid=os.getpid(),epoch=start,source_remote=receipt['remote_commit'],root_tree=receipt['root_tree'],protocol_sha256=sha(OUT/'protocol.json'),all_stop=end,Adam_completed_at_launch=0,max_supervised_Adam=0,max_shadow_Adam=0,max_supervised_SGD=128,max_shadow_SGD=128,CPU_total_one_core=True,GPU=False));pids=[];error=None
    def call(action,cap):
        assert time.time()+cap+10<end
        with (OUT/(action+'.log')).open('xb') as log:
            c=subprocess.Popen(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'t564_runtime.py'),action],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);pids.append(c.pid);save(DOC/'STATUS.json',dict(state=action,pid=c.pid,driver_pid=os.getpid(),epoch=time.time()))
            try:c.wait(timeout=cap);assert c.returncode==0,(action,c.returncode)
            except BaseException:
                try:os.killpg(c.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:c.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
                raise
        assert not Path('/proc',str(c.pid)).exists()
    try:
        call('main',p['main_seconds']);pids+=R.G.stopped(OUT);call('audit',p['audit_seconds']);assert read(OUT/'audit.json')['state']=='PASS'
        files=sorted(f for f in OUT.rglob('*') if f.is_file() and f.name!='local_manifest.json' and f.name!='driver.log' and f.suffix not in ['.lock','.pyc'])
        members=[dict(path=str(f.relative_to(ROOT)),sha256=sha(f),bytes=f.stat().st_size) for f in files];save(OUT/'local_manifest.json',dict(state='SEALED_LOCAL_ALL_MEMBER_SHA',files=members));assert all(sha(ROOT/m['path'])==m['sha256'] for m in members)
    except BaseException:error=traceback.format_exc();save(DOC/'DRIVER_ERROR.json',dict(error=error,epoch=time.time(),no_restart=True))
    gone=all(not Path('/proc',str(i)).exists() for i in pids);save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=gone,scientific_pids=pids,finalizer_writes_only_docs_after_this_point=True,epoch=time.time()))
    summary=read(OUT/'summary.json') if (OUT/'summary.json').exists() else None
    report=dict(state='COMPLETE' if error is None else 'FAILED',error=error,summary=summary,all_scientific_processes_stopped=gone,remote='PENDING',new_RL=0,globally_promoted=False,epoch=time.time());save(DOC/'FINAL_REPORT.json',report)
    if gone and time.time()+p['transport_seconds']+p['source_seconds']+15<end:
        try:
            c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'T564'],cwd=ROOT,capture_output=True,timeout=p['transport_seconds']);assert c.returncode==0,c.stdout.decode(errors='replace')[-1200:]+c.stderr.decode(errors='replace')[-1200:]
            report['remote']=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T564.json')['state']
        except BaseException:save(DOC/'REMOTE_PENDING.json',dict(error=traceback.format_exc(),epoch=time.time()))
    save(DOC/'FINAL_REPORT.json',report)
    f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text('2026-11-15 T564平滑策略蒸馏试验结束，'+report['state']+'，真实监督SGD与条件原生结果见day_20261115/t564/FINAL_REPORT，remote='+report['remote']+'。本轮监督表示迁移，不晋升，不自动追加学习或重启。\n\n'+f.read_text())
    try:
        subprocess.run(['git','add','stage/docs/day_20261115/t564','stage/docs/RESEARCH_STATUS.md','stage/docs/long_distance_20261101'],cwd=ROOT,check=True,capture_output=True);subprocess.run(['git','commit','-m','Report T564 bounded smooth policy consolidation'],cwd=ROOT,check=True,capture_output=True)
        if time.time()+p['source_seconds']+5<end:
            c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'source_sync'],cwd=ROOT,capture_output=True,timeout=p['source_seconds']);assert c.returncode==0
    except BaseException:save(DOC/'SOURCE_PENDING.json',dict(error=traceback.format_exc(),epoch=time.time()))
if __name__=='__main__':main()
