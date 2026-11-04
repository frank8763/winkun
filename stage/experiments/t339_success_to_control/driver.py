"""One launch, then bounded audit, complete raw backup and exact source sync."""
from t339_core import *
import subprocess

def source_sync():
    sys.path.insert(0,str(ROOT/'stage/experiments/t224_archive'));import archive,sync_source
    def request(method,url,payload=None,headers=None):
        assert url.startswith('https://api.github.com/repos/frank8763/winkun/')
        cmd=[str(archive.GH),'api','--method',method,url.removeprefix('https://api.github.com/')]
        if payload is not None:return subprocess.check_output([*cmd,'--input','-'],input=json.dumps(payload).encode(),timeout=60)
        return subprocess.check_output(cmd,timeout=60)
    archive.req=request;sync_source.main()

def main():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'launch.json').exists()
    began=time.time();end=began+p['all_work_seconds'];native_end=began+p['wall_seconds']
    src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    def status(phase,**extra):save(DOC/'T339_DRIVER.json',dict(phase=phase,pid=os.getpid(),updated_epoch=time.time(),all_work_deadline_epoch=end,**extra))
    def child(script,args,seconds,log):
        seconds=min(seconds,int(end-time.time())-180);assert seconds>10
        with Path(log).open('w') as f:
            return subprocess.run(['/usr/bin/timeout','--signal=TERM','--kill-after=2s',str(seconds)+'s','/usr/bin/taskset','-c','0',str(ROOT/'run_stage.sh'),str(script),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
    status('LAUNCHING_NATIVE')
    with (OUT/'run.log').open('xb') as f:
        proc=subprocess.Popen(['systemd-inhibit','--what=sleep','--mode=block','--why=Stage bounded CAPG recovery pilot',
            '/usr/bin/timeout','--signal=TERM','--kill-after=2s',str(p['wall_seconds'])+'s',
            '/usr/bin/taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'run.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    save(OUT/'launch.json',dict(pid=proc.pid,launch_epoch=began,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        remote_commit=src['remote_commit'],single_launch=True,native_deadline_epoch=native_end,training_deadline_epoch=native_end-p['final_reserve_seconds'],orchestrator_pid=os.getpid()))
    status('NATIVE_ACTIVE',native_wrapper_pid=proc.pid);code=proc.wait();status('NATIVE_EXITED',native_returncode=code)
    assert (OUT/'summary.json').exists(),'No summary; preserve all raw, no restart'
    s=read(OUT/'summary.json');assert s['all_workers_stopped'] and not Path('/proc/'+str(s['pid'])).exists()
    status('INDEPENDENT_AUDIT')
    audit_rc=child(HERE/'audit.py',[],p['audit_seconds'],DOC/'T339_audit.log')
    status('FULL_RAW_REMOTE_BACKUP',audit_returncode=audit_rc)
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T339.json';backup_rc=0
    if not receipt.exists():backup_rc=child(ROOT/'stage/experiments/t260_distance_archive/small_parts.py',['T339'],900,DOC/'T339_backup.log')
    backed=receipt.exists() and read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    passed=(DOC/'T339_AUDIT.json').exists() and read(DOC/'T339_AUDIT.json')['state']=='PASS'
    report=dict(producer=s,audit_pass=passed,audit_returncode=audit_rc,backup_returncode=backup_rc,full_raw_remote_readback=backed,
        no_7m_8m_training=True,no_full_six_acceptance=True,learned48_diagnostic=s.get('result',{}).get('learned48_diagnostic',False),elapsed_s=time.time()-began)
    save(DOC/'T339_FINAL_REPORT.json',report)
    (DOC/'T339_FINAL_REPORT.md').write_text('# T339实际结果\n\n'+
        '固定48秒宽场景恢复pilot，非原起点六米验收。主运行：'+s['state']+'。\n\n'+
        '冻结初始控制（初始old275为T338历史复用）：\n\n```json\n'+json.dumps(s.get('result',{}).get('before'),ensure_ascii=False,indent=2)+'\n```\n\n'+
        '一次更新前后确定性控制结果：\n\n```json\n'+json.dumps(s.get('result',{}),ensure_ascii=False,indent=2)+'\n```\n\n'+
        '物理步 '+str(s['physics_steps'])+'；命令 '+str(s['commands'])+'；live/shadow Adam '+json.dumps(s['updates'])+'。\n\n'+
        '独立审计 '+str(passed)+'；全部原件远端逐SHA读回 '+str(backed)+'。最终完整old275/six54/fresh16/reload78本pilot未执行，不将缺测当失败率，不晋升/不进7或8。CAPG/PPO已有方法，不称原创或保证成功。\n')
    research=ROOT/'stage/docs/RESEARCH_STATUS.md';research.write_text('2026-11-04 T339结束，producer='+s['state']+'，'+str(s['commands'])+'命令/'+str(s['physics_steps'])+'物理，独立audit='+str(passed)+'/全原件远端逐SHA='+str(backed)+'。48秒确定性改善='+str(s.get('result',{}).get('learned48_diagnostic',False))+'；固定48秒宽80课程恢复与原起点分列；最终完整旧275/原six54/fresh16/reload78未执行，非六米稳定，不晋升7/8，不重启扩预算。详见review_20261104/T339_FINAL_REPORT和原始summary/rounds。Stage/Phase隔离。\n\n'+research.read_text())
    status('EVIDENCE_FINISHED_SOURCE_SYNC_NEXT',audit_pass=passed,full_raw_remote_readback=backed)
    assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
    paths=['stage/docs/RESEARCH_STATUS.md']
    for folder in [DOC,ROOT/'stage/docs/long_distance_20261101']:
        paths += [str(f.relative_to(ROOT)) for f in folder.glob('*T339*') if f.is_file() and f.suffix in ('.md','.json')]
    subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
    if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip():subprocess.run(['git','commit','-m','Record bounded Stage CAPG recovery results and evidence'],cwd=ROOT,check=True)
    source_sync()

if __name__=='__main__':
    try:main()
    except BaseException:
        save(DOC/'T339_DRIVER_ERROR.json',dict(state='ORCHESTRATION_FAILED_NO_RESTART',pid=os.getpid(),error=traceback.format_exc(),epoch=time.time()));raise
