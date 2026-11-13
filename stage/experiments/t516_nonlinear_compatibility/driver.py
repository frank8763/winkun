"""One offline pilot and independent reconstruction, then archive and stop."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t516_nonlinear_compatibility';DOC=ROOT/'stage/docs/day_20261113'
def read(f):return json.loads(Path(f).read_text())
def save(f,d):
    f=Path(f);tmp=Path(str(f)+'.partial');tmp.write_text(json.dumps(d,ensure_ascii=False)+'\n');tmp.replace(f)
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def main():
    p=read(OUT/'protocol.json');end=p['absolute_all_stop'];assert not (OUT/'launch.json').exists()
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    save(OUT/'launch.json',dict(pid=os.getpid(),epoch=time.time(),all_stop=end,source_remote=source['remote_commit'],protocol_sha256=sha(OUT/'protocol.json'),CPU=[14,15],CPUQuota='100%',GPU=False))
    def call(action,seconds):
        with (OUT/(action+'.log')).open('xb') as log:
            c=subprocess.Popen(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'t516_runtime.py'),action,'fit'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:c.wait(timeout=seconds);assert c.returncode==0,(action,c.returncode)
            except BaseException:
                try:os.killpg(c.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:c.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:os.killpg(c.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    c.wait()
                raise
    try:
        folder=OUT/'fit';folder.mkdir();save(OUT/'status.json',dict(state='OFFLINE_COMPATIBILITY_ACTIVE',epoch=time.time()))
        call('main',1203);s=read(folder/'summary.json');assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists()
        call('audit',1200);a=read(folder/'audit.json');assert a['state']=='PASS' and not Path('/proc',str(a['pid'])).exists()
        files=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(folder.rglob('*')) if f.is_file()]
        save(folder/'local_manifest.json',dict(state='ALL_LOCAL_MEMBER_SHA_VERIFIED',files=files));assert all(sha(ROOT/f['path'])==f['sha256'] for f in files)
        save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=[s['pid'],a['pid']],finalizer_writes_only_docs_after_this_point=True,epoch=time.time()))
        save(OUT/'status.json',dict(state='SCIENTIFIC_FINISHED',offline_joint_gate=s['result']['offline_joint_gate'],epoch=time.time()))
        remote='PENDING'
        try:
            c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'T516'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(1200,end-time.time()-180))
            assert c.returncode==0,c.stdout.decode(errors='replace')[-2000:];remote=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T516.json')['state']
        except BaseException:save(DOC/'T516_REMOTE_PENDING.json',dict(error=traceback.format_exc(),raw_preserved=True))
        report=dict(state='COMPLETE_AUDIT_PASS',offline_joint_gate=s['result']['offline_joint_gate'],result=s['result'],accounting=dict(physics=0,commands=0,main_seconds=s['elapsed_s'],live_Actor_Adam=p['fit_steps'],shadow_Actor_Adam=p['fit_steps'],audit_discarded_historical_Actor_Adam=p['fit_steps'],preflight_discarded_Adam=4,Value_Adam=0,newPG=0),remote=remote,all_scientific_processes_stopped=True,epoch=time.time())
        save(DOC/'T516_FINAL_REPORT.json',report)
        f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text('2026-11-13 T516非线性离线兼容试验结束，offline_joint_gate='+str(report['offline_joint_gate'])+'，512live/512shadow/512历史audit+4pref，0native/0新PG/Value0，原父策略冻结/零残差回退，remote='+remote+'；教师30训练29同轨迹blocked验证非导航留出，旧19和前缀保持指标见day_20261113/T516_FINAL_REPORT。无新自主验收或晋升，field最佳11/24不变，Phase隔离。\n\n'+f.read_text())
        subprocess.run(['git','add','stage/docs/day_20261113','stage/docs/long_distance_20261101','stage/docs/RESEARCH_STATUS.md'],cwd=ROOT,check=True)
        subprocess.run(['git','commit','-m','Report offline nonlinear recovery compatibility and unchanged navigation baseline'],cwd=ROOT,check=True)
        try:
            c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'source_sync'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(180,end-time.time()-5))
            assert c.returncode==0,c.stdout.decode(errors='replace')[-2000:]
        except BaseException:save(DOC/'T516_SOURCE_PENDING.json',dict(error=traceback.format_exc(),local_committed=True))
    except BaseException:
        save(DOC/'T516_DRIVER_ERROR.json',dict(error=traceback.format_exc(),raw_preserved=True,no_restart=True,epoch=time.time()));raise
if __name__=='__main__':main()
