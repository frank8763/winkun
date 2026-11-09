"""Single offline finalizer. Native learning and relay never need internet."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t449_offline_joint_recovery';DOC=ROOT/'stage/docs/overnight_20261109'
def read(f):return json.loads(f.read_text())
def save(f,v):f.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def child(script,*args,seconds):
    assert seconds>0
    c=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/script),*args],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        b,_=c.communicate(timeout=seconds)
        if c.returncode:raise RuntimeError(b.decode(errors='replace')[-6500:])
        return b.decode(errors='replace')[-1500:]
    except BaseException:
        try:os.killpg(c.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:c.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:os.killpg(c.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            c.wait()
        raise
def main():
    p=read(OUT/'protocol.json');launch=read(OUT/'launch.json');end=launch['all_stop'];save(OUT/'driver_started.json',dict(pid=os.getpid(),epoch=time.time(),all_stop=end,science_requires_network=False))
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('T449 hard all-stop')))
    report=dict(state='RUNNING',remote='PENDING',six_gate=False,eight_started=False)
    try:
        try:child('run.py','main',seconds=min(p['main_seconds']+5,launch['native_stop']-time.time()+5))
        except BaseException:save(DOC/'MAIN_ERROR.json',dict(error=traceback.format_exc(),raw_preserved=True,no_training_restart=True))
        assert (OUT/'summary.json').exists(),'No complete summary: preserve partial and refuse false completion'
        s=read(OUT/'summary.json');ids=[s['pid']]+[pid for g in read(OUT/'worker_generations.json') for pid in g['workers']];assert all(not Path('/proc',str(pid)).exists() for pid in ids)
        audit=None
        if s['state']=='COMPLETE':
            try:
                child('run.py','audit',seconds=min(p['audit_seconds'],end-time.time()-900));audit=read(OUT/'audit.json');ids.append(audit['pid'])
            except BaseException:save(DOC/'AUDIT_ERROR.json',dict(error=traceback.format_exc(),raw_preserved=True,no_repeat=True))
        assert all(not Path('/proc',str(pid)).exists() for pid in ids)
        save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=ids,epoch=time.time()))
        files=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(OUT.rglob('*')) if f.is_file() and f.suffix not in ['.lock','.partial','.pyc']]
        save(DOC/'LOCAL_MANIFEST.json',dict(state='ALL_LOCAL_MEMBERS_HASHED',files=files,remote='PENDING'))
        report.update(state='COMPLETE_AUDITED_LOCAL' if audit and audit['state']=='PASS' else 'STOPPED_EVIDENCE_PRESERVED_NOT_AUDIT_PASS',summary=s,audit=audit)
        save(DOC/'FINAL_REPORT.json',report)
        # All pre-registered scientific stages already finished before network.
        try:
            child('transport.py','T449',seconds=min(620,end-time.time()-200));report['remote']=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T449.json')['state']
        except BaseException:save(DOC/'BACKUP_ERROR.json',dict(error=traceback.format_exc(),remote='PENDING',science_NOT_restarted=True,all_local_raw_preserved=True))
        native_gate=bool(s.get('gate') and s['gate'].get('native_gate'));report['six_gate']=bool(native_gate and audit and audit['state']=='PASS' and report['remote']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED')
        save(DOC/'FINAL_REPORT.json',report)
        selected=s.get('selected',{});body=[f"# 2026-11-09 Stage离线接力真实结果",f"状态：{report['state']}；远端：{report['remote']}。",f"物理{s['physics']}，全部命令{s['commands']}，主管线{s['elapsed_s']:.3f}秒。",f"选定研究模型：宽六米{selected.get('wide54')}；旧275：{selected.get('old275')}。",f"完整六米门槛：{report['six_gate']}；八米未启动。",f"门槛原始记录：{s.get('gate')}。","监督、教师混合和自身PPO分开；所有拒绝模型、失败、未执行/删失保留。" ];(DOC/'FINAL_REPORT.md').write_text('\n\n'.join(body)+'\n')
        status=ROOT/'stage/docs/RESEARCH_STATUS.md';status.write_text(f"2026-11-10 T449离线有限三机制接力已收尾，state={report['state']}，remote={report['remote']}，physics={s['physics']} commands={s['commands']}；详overnight_20261109/FINAL_REPORT。six_gate={report['six_gate']}，8未进入，safeT255冻结/Phase隔离，不重启或追加旧预算。\n\n"+status.read_text())
        subprocess.run(['git','add','stage/docs/overnight_20261109','stage/docs/RESEARCH_STATUS.md','stage/docs/long_distance_20261101'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        subprocess.run(['git','commit','-m','Report finite offline Stage tail-recovery-PPO relay'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:child('transport.py','source_sync',seconds=min(180,end-time.time()-5))
        except BaseException:save(DOC/'SOURCE_SYNC_ERROR.json',dict(error=traceback.format_exc(),local_committed=True,remote='PENDING'))
    except BaseException:save(DOC/'DRIVER_ERROR.json',dict(error=traceback.format_exc(),raw_preserved=True,no_training_restart=True,epoch=time.time()))
if __name__=='__main__':main()
