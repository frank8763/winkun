"""Standard-library registration, resource admission and bounded supervisor."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t643_old_policy_recovery'
DOC=ROOT/'stage/docs/day_20261118/t643'
PHASE_PREFIX='/home/linghu/Documents/Codex/2026-10-31/new-chat/phase-/'


def read(f):return json.loads(Path(f).read_text())
def save(f,v):Path(f).write_text(json.dumps(v,indent=2)+'\n')
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def resources():
    cpus=[8,10,12,14]
    def physical(cpu):
        base=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        return tuple((base/f).read_text().strip() for f in ['physical_package_id','core_id'])
    ours={physical(c) for c in cpus};assert len(ours)==4
    phase=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            argv=(proc/'cmdline').read_bytes().split(b'\0')
            # Match separate argv elements, never this script's own source text.
            if not any(v.decode(errors='replace').startswith(PHASE_PREFIX) for v in argv):continue
            mask=set()
            for thread in (proc/'task').iterdir():
                try:mask.update(os.sched_getaffinity(int(thread.name)))
                except ProcessLookupError:pass
            overlap=sorted(c for c in mask if physical(c) in ours)
            phase.append(dict(pid=int(proc.name),allowed_cpus=sorted(mask),physical_overlap=overlap))
            assert not overlap, ('Phase physical CPU overlap; do not launch',proc.name,overlap)
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return dict(state='PASS',stage_cpus=cpus,stage_physical_cores=sorted(ours),phase_process_metadata_only=phase,
                GPU=False,CPUQuota='400%',Nice=19,CPUWeight=10,MemoryMax=4*2**30,MemorySwapMax=0)


def register():
    assert not OUT.exists()
    p=read(HERE/'protocol.json');assert not p['authorization_approved']
    assert read(DOC/'PREFLIGHT.json')['state']=='PASS_ZERO_NATIVE_ZERO_OPTIMIZER'
    parent=read(ROOT/'stage/runs/t642_tail_retention/protocol.json');actual=read(ROOT/'stage/runs/t642_tail_retention/summary.json')
    assert parent['authorization_approved'] and actual['state']=='COMPLETE'
    assert read(ROOT/'stage/runs/t642_tail_retention/audit.json')['state']=='PASS'
    assert actual['physics']+p['max_physics']<=parent['max_physics'] and actual['commands']+p['max_commands']<=parent['max_commands']
    actual_bytes=sum(f.stat().st_size for f in (ROOT/'stage/runs/t642_tail_retention').rglob('*') if f.is_file())
    assert actual_bytes+p['max_bytes']<=parent['max_bytes']
    for f,h in p['inputs'].items():assert sha(f)==h,f
    save(DOC/'RESOURCE_PREFLIGHT.json',resources())
    now=time.time();p.update(authorization_approved=True,registration_epoch=now,
        absolute_native_stop=min(now+600,parent['absolute_native_stop']),
        absolute_all_stop=min(now+1200,parent['absolute_all_stop']),absolute_learning_stop=now,
        user_authorization='User explicitly 下一步开始 and asks algorithm improvement after T642 failure; bounded frozen actual-state recovery within approved T642 remainder.')
    assert now+p['main_seconds']+p['audit_seconds']+p['transport_seconds']+p['source_seconds']+90<p['absolute_all_stop']
    save(DOC/'AUTHORIZATION.json',dict(user_reply='下一步开始，然后回答我怎么改进才能将这个论文继续推进？改算法，还是加入什么？',
       parent_T642_actual_physics=actual['physics'],new_physics=p['max_physics'],combined_physics=actual['physics']+p['max_physics'],parent_limit_physics=parent['max_physics'],
       parent_T642_actual_commands=actual['commands'],new_commands=p['max_commands'],combined_commands=actual['commands']+p['max_commands'],parent_limit_commands=parent['max_commands'],
       new_optimizer=0,new_budget_or_deadline_extension=False,all_stop=p['absolute_all_stop'],parent_all_stop=parent['absolute_all_stop']))
    p['inputs'][str(DOC/'AUTHORIZATION.json')]=sha(DOC/'AUTHORIZATION.json');p['inputs'][str(DOC/'PREFLIGHT.json')]=sha(DOC/'PREFLIGHT.json')
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    print('T643_REGISTERED_REMAINING_T642_BUDGET',p['absolute_all_stop'])


def run_action(name,seconds,children,p):
    assert time.time()+seconds+30<p['absolute_all_stop']
    with (OUT/(name+'.log')).open('xb') as log:
        c=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'t643_runtime.py'),name],cwd=ROOT,
           stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        children.append(c.pid)
        try:
            c.wait(timeout=seconds);assert c.returncode==0,(name,c.returncode)
        except BaseException:
            try:os.killpg(c.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:c.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
            raise


def driver():
    p=read(OUT/'protocol.json');assert p['authorization_approved']
    assert (OUT/'protocol.json').read_bytes()==(HERE/'protocol.json').read_bytes()
    assert not (OUT/'DRIVER_STARTED.json').exists() and not (OUT/'launch.json').exists()
    assert time.time()+p['main_seconds']+p['audit_seconds']+p['transport_seconds']+p['source_seconds']+120<p['absolute_all_stop']
    resources()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    save(OUT/'DRIVER_STARTED.json',dict(pid=os.getpid(),epoch=time.time(),source_remote=receipt['remote_commit']))
    children=[];error=None
    try:
        run_action('main',p['main_seconds'],children,p)
        run_action('audit',p['audit_seconds'],children,p)
    except BaseException:error=traceback.format_exc()
    if (OUT/'worker_generations.json').exists():
        for g in read(OUT/'worker_generations.json'):children.extend([g['parent'],*g['workers']])
    ids=sorted(set(children));gone=all(not Path('/proc',str(i)).exists() for i in ids)
    save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=gone,scientific_pids=ids,epoch=time.time()))
    s=read(OUT/'summary.json') if (OUT/'summary.json').exists() else None
    report=dict(state='FAILED' if error else 'COMPLETE',error=error,summary=s,
       independent_audit=read(OUT/'audit.json') if (OUT/'audit.json').exists() else None,
       all_scientific_processes_stopped=gone,remote='PENDING_LOCAL_RAW_RETAINED',
       scope='One neural teacher first-divergence recovery, nominal control reused from exact T639 evidence, full causal carry, no learning or global promotion.')
    save(OUT/'FINAL_REPORT.json',report);save(DOC/'FINAL_REPORT.json',report)
    files=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f))
           for f in sorted(OUT.rglob('*')) if f.is_file() and f.name not in ['local_manifest.json','driver.log']]
    save(OUT/'local_manifest.json',dict(state='LOCAL_RAW_SHA_ONLY',files=files))
    if gone and time.time()+p['transport_seconds']+p['source_seconds']+60<p['absolute_all_stop']:
        try:
            c=subprocess.run([str(ROOT/'run_stage.sh'),str(HERE/'transport.py')],cwd=ROOT,capture_output=True,
                timeout=p['transport_seconds'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
            assert c.returncode==0,c.stderr.decode(errors='replace')[-1600:]
            report['remote']=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T643.json')['state']
        except BaseException:report['backup_error']=traceback.format_exc()
    save(DOC/'FINAL_REPORT.json',report)
    status=ROOT/'stage/docs/RESEARCH_STATUS.md'
    status.write_text('2026-11-18 T643固定神经教师恢复资格验证结束：'+report['state']+
        '，remote='+report['remote']+'；结果见day_20261118/t643/FINAL_REPORT；恢复资格见完整报告；0学习，不称自主全局晋升。\n\n'+status.read_text())
    paths=[str(DOC.relative_to(ROOT)),'stage/docs/RESEARCH_STATUS.md']
    for pattern in ['release_receipt_T643*.json','source_manifest_T643*.json']:
        paths.extend(str(f.relative_to(ROOT)) for f in (ROOT/'stage/docs/long_distance_20261101').glob(pattern))
    try:
        subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True,capture_output=True)
        subprocess.run(['git','commit','-m','Report first-divergence neural recovery qualification and full raw archive'],cwd=ROOT,check=True,capture_output=True)
        if time.time()+p['source_seconds']+15<p['absolute_all_stop']:
            subprocess.run([str(ROOT/'run_stage.sh'),str(ROOT/'stage/experiments/t629_offline_night_relay/t629_transport.py'),'source_sync'],
                 cwd=ROOT,capture_output=True,check=True,timeout=p['source_seconds'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    except BaseException:save(DOC/'SOURCE_PENDING.json',dict(error=traceback.format_exc()))
    print(json.dumps(dict(state=report['state'],remote=report['remote'],summary=s)),flush=True)


if __name__=='__main__':
    if sys.argv[1]=='resources':print(json.dumps(resources()))
    else:globals()[sys.argv[1]]()
