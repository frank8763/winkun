"""Single night orchestrator: one launch, then stopped-run audit and archive.

No restart, added episode, changed threshold or new mechanism on failure.
"""
from t336_core import *
import subprocess

def main():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'launch.json').exists()
    began=time.time();src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    def status(phase,**extra):save(DOC/'NIGHT_DRIVER.json',dict(phase=phase,pid=os.getpid(),updated_epoch=time.time(),all_work_deadline_epoch=p['absolute_all_work_deadline_epoch'],**extra))
    def child(script,args,deadline,log):
        seconds=int(deadline-time.time());assert seconds>10
        with Path(log).open('w') as f:
            return subprocess.run(['/usr/bin/timeout','--signal=TERM','--kill-after=2s',str(seconds)+'s','/usr/bin/taskset','-c','0',str(ROOT/'run_stage.sh'),str(script),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
    def source_sync():
        # Authenticated CLI transport; preserve the existing exact root-tree and
        # ancestry guards instead of changing publication semantics.
        sys.path.insert(0,str(ROOT/'stage/experiments/t224_archive'));import archive,sync_source
        def request(method,url,payload=None,headers=None):
            assert url.startswith('https://api.github.com/repos/frank8763/winkun/')
            cmd=[str(archive.GH),'api','--method',method,url.removeprefix('https://api.github.com/')]
            if payload is not None:return subprocess.check_output([*cmd,'--input','-'],input=json.dumps(payload).encode(),timeout=60)
            return subprocess.check_output(cmd,timeout=60)
        archive.req=request;sync_source.main()
    status('LAUNCHING_NATIVE')
    with (OUT/'run.log').open('xb') as f:
        proc=subprocess.Popen(['systemd-inhibit','--what=sleep','--mode=block','--why=Stage bounded reverse-curriculum overnight',
            '/usr/bin/timeout','--signal=TERM','--kill-after=2s',str(int(p['absolute_deadline_epoch']-time.time()))+'s',
            '/usr/bin/taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'main'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    save(OUT/'launch.json',dict(pid=proc.pid,launch_epoch=time.time(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        remote_commit=src['remote_commit'],single_launch=True,absolute_deadline_epoch=p['absolute_deadline_epoch'],absolute_training_deadline_epoch=p['absolute_training_deadline_epoch'],orchestrator_pid=os.getpid()))
    status('NATIVE_ACTIVE',native_wrapper_pid=proc.pid)
    code=proc.wait();status('NATIVE_EXITED',native_returncode=code)
    assert (OUT/'summary.json').exists(),'No final summary; retain raw; no restart'
    s=read(OUT/'summary.json');assert s['all_workers_stopped'] and not Path('/proc/'+str(s['pid'])).exists()
    status('INDEPENDENT_AUDIT')
    audit_rc=child(HERE/'audit.py',[],min(time.time()+p['audit_seconds'],p['absolute_all_work_deadline_epoch']-900),DOC/'audit.log')
    status('FULL_RAW_REMOTE_BACKUP',audit_returncode=audit_rc)
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T336.json'
    backup_rc=0
    if not receipt.exists():backup_rc=child(ROOT/'stage/experiments/t260_distance_archive/small_parts.py',['T336'],min(time.time()+610,p['absolute_all_work_deadline_epoch']-180),DOC/'backup.log')
    backed=receipt.exists() and read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    report=dict(producer=s,audit_returncode=audit_rc,backup_returncode=backup_rc,full_raw_remote_readback=backed,
        audit_pass=(DOC/'T336_AUDIT.json').exists() and read(DOC/'T336_AUDIT.json')['state']=='PASS',
        no_7m_8m_training=True,night_physics=s['physics_steps'],night_commands=s['commands'],
        summary='真实最终证据；课程/中心不称六米稳定；未执行单列。',elapsed_s=time.time()-began)
    save(DOC/'FINAL_REPORT.json',report)
    (DOC/'FINAL_REPORT.md').write_text('# Stage 2026-11-03 夜间实际结果\n\n'+
        '主运行 '+s['state']+'；全部物理 '+str(s['physics_steps'])+'；全部命令 '+str(s['commands'])+'。\n\n'+
        '两臂结果：\n\n```json\n'+json.dumps(s['arms'],ensure_ascii=False,indent=2)+'\n```\n\n'+
        '原六米完整门槛：\n\n```json\n'+json.dumps(s.get('gate'),ensure_ascii=False,indent=2)+'\n```\n\n'+
        '独立审计通过：'+str(report['audit_pass'])+'；全部原件远端逐SHA读回：'+str(backed)+'。七八米未训练。\n\n'+
        'Actor/Q实际live/shadow Adam：'+json.dumps(s['updates'])+'。缺测不是失败率。标准SAC/RLPD/反向课程不称原创。\n')
    # All experiment writers have stopped. Only now update the source summary.
    research=ROOT/'stage/docs/RESEARCH_STATUS.md';old=research.read_text()
    research.write_text('2026-11-03夜间T336已结束，producer='+s['state']+'，'+str(s['commands'])+'命令/'+str(s['physics_steps'])+'物理；独立audit='+str(report['audit_pass'])+'，完整原件远端读回='+str(backed)+'。两臂/真实更新/原275与six54等以overnight_20261103/FINAL_REPORT.json与原summary为准，不把宽80秒课程或中心叫原六米稳定，不进七八米。无自动重启/扩预算，Phase完全隔离。\n\n'+old)
    status('SOURCE_FINALIZATION',audit_pass=report['audit_pass'],full_raw_remote_readback=backed)
    if time.time()<p['absolute_all_work_deadline_epoch']-120:
        # Never sweep unrelated staged user work into this automatic commit.
        assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
        paths=['stage/docs/RESEARCH_STATUS.md','stage/docs/overnight_20261103']
        for pattern in ['*T336*']:
            paths += [str(f.relative_to(ROOT)) for f in (ROOT/'stage/docs/long_distance_20261101').glob(pattern) if f.is_file()]
        subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
        if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip():subprocess.run(['git','commit','-m','Record bounded Stage reverse-online night results and verified evidence'],cwd=ROOT,check=True)
        source_sync()
    status('FINISHED',audit_pass=report['audit_pass'],full_raw_remote_readback=backed)

if __name__=='__main__':
    try:main()
    except BaseException:
        save(DOC/'NIGHT_DRIVER_ERROR.json',dict(state='ORCHESTRATION_FAILED_NO_RESTART',pid=os.getpid(),error=traceback.format_exc(),epoch=time.time()))
        raise
