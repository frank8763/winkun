"""One writer, two bounded Critic-only arms, no native simulators or Actor steps."""
from pathlib import Path
import time,os,subprocess,signal,fcntl,traceback,json,shutil
import t625r_runtime as R
M=R.M
L=M.L

def main():
    p=M.config();assert p['authorization_approved'];assert not (M.OUT/'launch.json').exists()
    assert L.sha(M.HERE/'protocol.json')==L.sha(M.OUT/'protocol.json')
    for f,h in p['inputs'].items():assert L.sha(f)==h,f
    used=lambda folder:sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())
    assert used(M.ROOT/'stage/runs')-used(M.OUT)+p['max_bytes']+131072<2**30
    assert shutil.disk_usage(M.ROOT).free>=40*2**30
    receipt=L.read(M.ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=M.ROOT,text=True).strip()
    locks=[]
    for name in ['.stage_writer.lock','.stage_gpu.lock']:
        f=(M.ROOT/'stage/runs'/name).open('a+b');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Another GPU compute job is active; preserve Phase resources'
    L.save(M.OUT/'launch.json',dict(pid=os.getpid(),epoch=time.time(),source_remote=receipt['remote_commit'],CPU=p['stage_cpus'],GPU=True));children=[];pids=[];error=None
    def call(action,arm,seconds):
        with (M.OUT/arm/(action+'.log')).open('xb') as log:
            c=subprocess.Popen([str(M.ROOT/'run_stage.sh'),str(M.HERE/'t625r_runtime.py'),action,arm],cwd=M.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);pids.append(c.pid);L.save(M.DOC/'STATUS.json',dict(state=action,arm=arm,pid=c.pid,driver_pid=os.getpid(),epoch=time.time()))
            try:c.wait(timeout=seconds);assert c.returncode==0,(action,arm,c.returncode)
            except BaseException:
                try:os.killpg(c.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:c.wait(timeout=2)
                except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
                raise
    try:
        for arm in ['regression','rank']:
            now=time.time()
            if not (now+p['learning_seconds']+15<p['absolute_learning_stop'] and now+p['main_seconds']+p['audit_seconds']+p['finalization_seconds']<p['absolute_all_stop']):
                children.append(dict(arm=arm,state='NOT_ADMITTED_FULL_WORST_TIME'));break
            assert used(M.OUT)+900000+131072<p['max_bytes'],'Full arm storage admission denied'
            folder=M.OUT/arm;folder.mkdir();call('main',arm,p['main_seconds']);call('audit',arm,p['audit_seconds']);assert L.read(folder/'audit.json')['state']=='PASS'
            members=[dict(path=str(f.relative_to(M.ROOT)),sha256=L.sha(f),bytes=f.stat().st_size) for f in sorted(folder.rglob('*')) if f.is_file()];L.save(folder/'local_manifest.json',dict(files=members));s=L.read(folder/'summary.json');children.append(dict(arm=arm,state='COMPLETE',initial_MSE=s['initial']['MSE'],final_MSE=s['final']['MSE'],critic_live=p['updates_per_arm'],critic_shadow=p['updates_per_arm'],critic_audit=p['updates_per_arm'],actor_updates=0))
    except BaseException:error=traceback.format_exc();L.save(M.DOC/'DRIVER_ERROR.json',dict(error=error,raw_preserved=True,no_restart=True))
    gone=all(not Path('/proc',str(i)).exists() for i in pids);assert gone
    L.save(M.OUT/'SEALED.json',dict(all_scientific_processes_stopped=gone,scientific_pids=pids,epoch=time.time()));report=dict(state='FAILED' if error else 'COMPLETE',error=error,children=children,actor_updates=0,native_commands=0,physics=0,remote='PENDING',navigation_validation=None,policy_Q_certified=False,epoch=time.time());L.save(M.DOC/'FINAL_REPORT.json',report)
    if time.time()+p['transport_seconds']+p['source_seconds']+15<p['absolute_all_stop']:
        try:
            subprocess.run([str(M.ROOT/'run_stage.sh'),str(M.HERE/'t625r_transport.py'),'T625R'],cwd=M.ROOT,check=True,capture_output=True,timeout=p['transport_seconds']);report['remote']=L.read(M.ROOT/'stage/docs/long_distance_20261101/release_receipt_T625R.json')['state']
        except BaseException:L.save(M.DOC/'REMOTE_PENDING.json',dict(error=traceback.format_exc()))
    L.save(M.DOC/'FINAL_REPORT.json',report);f=M.ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text('2026-11-17 T625R GPU行为回报预测对照结束：'+report['state']+'；Actor更新0/native0，非导航通过/非目标策略Q或动作梯度认证，结果见t625r/FINAL_REPORT，remote='+report['remote']+'。\n\n'+f.read_text())
    try:
        paths=[str(M.DOC.relative_to(M.ROOT)),'stage/docs/RESEARCH_STATUS.md']
        for n in ['release_receipt_T625R.json','source_manifest_T625R.json']:
            if (M.ROOT/'stage/docs/long_distance_20261101'/n).exists():paths.append('stage/docs/long_distance_20261101/'+n)
        subprocess.run(['git','add','--',*paths],cwd=M.ROOT,check=True,capture_output=True);subprocess.run(['git','commit','-m','Report bounded GPU critic calibration with frozen Actor'],cwd=M.ROOT,check=True,capture_output=True)
        if time.time()+p['source_seconds']+5<p['absolute_all_stop']:subprocess.run([str(M.ROOT/'run_stage.sh'),str(M.HERE/'t625r_transport.py'),'source_sync'],cwd=M.ROOT,check=True,capture_output=True,timeout=p['source_seconds'])
    except BaseException:L.save(M.DOC/'SOURCE_PENDING.json',dict(error=traceback.format_exc()))

if __name__=='__main__':main()
