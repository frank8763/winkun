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
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'stage/experiments/t638_memory_ready_recovery'))
import t645r_memory as Memory

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t645r_transport_recovery'
DOC=ROOT/'stage/docs/day_20261118/t645r'
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
                GPU='learning_and_history_only_yield_other_compute',CPUQuota='400%',Nice=19,CPUWeight=10,MemoryMax=4*2**30,MemorySwapMax=0)


def register():
    assert not OUT.exists()
    p=read(HERE/'protocol.json');old=ROOT/'stage/runs/t645_iterative_recovery_relay';original=read(old/'protocol.json')
    report=read(ROOT/'stage/docs/day_20261118/t645/FINAL_REPORT.json')
    assert not report['science_started'] and report['all_scientific_processes_stopped'] and report['remote']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert not (old/'launch.json').exists() and not (old/'optimizer_calls.jsonl').exists()
    assert not p['authorization_approved'] and read(DOC/'PREFLIGHT.json')['state']=='PASS_ZERO_NATIVE_ZERO_OPTIMIZER'
    for f,h in p['inputs'].items():assert sha(f)==h,f
    assert p['absolute_all_stop']==original['absolute_all_stop']
    assert time.time()+p['preload_seconds']+p['input_audit_seconds']+p['main_seconds']+p['audit_seconds']+p['transport_seconds']+p['source_seconds']+120<p['absolute_all_stop']
    for key in ['max_physics','max_commands','new_Actor_Adam','shadow_Actor_Adam','independent_historical_Actor_Adam','max_rounds']:
        assert p[key]==original[key],key
    save(DOC/'RESOURCE_PREFLIGHT.json',resources());p.update(authorization_approved=True,registration_epoch=time.time())
    save(DOC/'AUTHORIZATION.json',dict(basis='Engineering recovery of unused approved T645 science; user authorizes supervision/repair and finite three-round budget. Original producer FAILED preserved.',
        original_authorization=str(ROOT/'stage/docs/day_20261118/t645/AUTHORIZATION.json'),original_science_started=False,
        previous_preload_seconds=report['preload_seconds_used'],remaining_preload_seconds=p['preload_seconds'],
        total_record_limit=original['max_bytes'],previous_record_bytes=p['previous_record_bytes'],remaining_record_bytes=p['max_bytes'],
        max_physics=p['max_physics'],max_commands=p['max_commands'],Actor_each=p['new_Actor_Adam'],all_stop=p['absolute_all_stop'],no_new_window_or_quota=True))
    p['inputs'][str(DOC/'AUTHORIZATION.json')]=sha(DOC/'AUTHORIZATION.json');p['inputs'][str(DOC/'PREFLIGHT.json')]=sha(DOC/'PREFLIGHT.json')
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'REGISTERED.json',dict(state='REGISTERED_NOT_LAUNCHED',all_stop=p['absolute_all_stop'],predecessor_failed=True))
    print('T645R_REGISTERED_NO_BUDGET_OR_DEADLINE_EXTENSION')


def run_action(name,seconds,children,p):
    assert time.time()+seconds+30<p['absolute_all_stop']
    with (OUT/(name+'.log')).open('xb') as log:
        c=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'t645r_runtime.py'),name],cwd=ROOT,
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
    assert time.time()+p['preload_seconds']+p['input_audit_seconds']+p['main_seconds']+p['audit_seconds']+p['transport_seconds']+p['source_seconds']+120<p['absolute_all_stop']
    resources()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    save(OUT/'DRIVER_STARTED.json',dict(pid=os.getpid(),epoch=time.time(),source_remote=receipt['remote_commit']))
    children=[];error=None;server=None;raw=None;science_started=False
    preload_start=time.time()
    try:
        def prepare_guard():
            assert time.time()<min(preload_start+p['preload_seconds'],p['absolute_all_stop']-p['main_seconds']-p['audit_seconds']-p['transport_seconds']-p['source_seconds']-60)
        raw=Memory.preload(p,OUT,prepare_guard)
        server=Memory.Server(OUT/'raw.sock',raw,p['memory_sources'],p['absolute_all_stop'])
        run_action('verify_inputs',p['input_audit_seconds'],children,p)
        science_started=True
        run_action('main',p['main_seconds'],children,p)
        assert server.error is None,server.error
        run_action('audit',p['audit_seconds'],children,p)
        assert server.error is None,server.error
    except BaseException:error=traceback.format_exc()
    finally:
        if server:
            server.close()
            Memory.save(OUT/'RAM_BROKER_REPORT.json',dict(state='CLOSED',requests=len(server.calls),
                main_reads=sum(v[0]=='main' for v in server.calls),audit_reads=sum(v[0]=='audit' for v in server.calls),
                error=server.error,raw_members=len(raw),raw_bytes=sum(map(len,raw.values())),temporary_disk_cache_bytes=0))
            server.raw={}
        raw=None
    if (OUT/'worker_generations.json').exists():
        for g in read(OUT/'worker_generations.json'):children.extend([g['parent'],*g['workers']])
    ids=sorted(set(children));gone=all(not Path('/proc',str(i)).exists() for i in ids)
    save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=gone,scientific_pids=ids,epoch=time.time()))
    s=read(OUT/'summary.json') if (OUT/'summary.json').exists() else None
    report=dict(state='FAILED' if error else 'COMPLETE',error=error,summary=s,
       independent_audit=read(OUT/'audit.json') if (OUT/'audit.json').exists() else None,
       all_scientific_processes_stopped=gone,remote='PENDING_LOCAL_RAW_RETAINED',
       scope='Up to3 distinct verified recovery aggregation rounds from original T501; original384 retention; one conditional same-candidate407; no universal safety or generalization claim.',science_started=science_started,preload_seconds_used=time.time()-preload_start if not science_started else read(OUT/'DATA_READY.json')['elapsed_s'])
    save(OUT/'FINAL_REPORT.json',report);save(DOC/'FINAL_REPORT.json',report)
    files=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f))
           for f in sorted(OUT.rglob('*')) if f.is_file() and f.name not in ['local_manifest.json','driver.log']]
    save(OUT/'local_manifest.json',dict(state='LOCAL_RAW_SHA_ONLY',files=files))
    if gone and time.time()+p['transport_seconds']+p['source_seconds']+60<p['absolute_all_stop']:
        try:
            c=subprocess.run([str(ROOT/'run_stage.sh'),str(HERE/'transport.py')],cwd=ROOT,capture_output=True,
                timeout=p['transport_seconds'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
            assert c.returncode==0,c.stderr.decode(errors='replace')[-1600:]
            report['remote']=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T645R.json')['state']
        except BaseException:report['backup_error']=traceback.format_exc()
    save(DOC/'FINAL_REPORT.json',report)
    status=ROOT/'stage/docs/RESEARCH_STATUS.md'
    status.write_text('2026-11-18 T645R完整内存数据供给恢复学习结束：'+report['state']+
        '，remote='+report['remote']+'；结果见day_20261118/t645r/FINAL_REPORT；完整结果含恢复资格、实际更新和条件407；未测项null，不称泛化。\n\n'+status.read_text())
    paths=[str(DOC.relative_to(ROOT)),'stage/docs/RESEARCH_STATUS.md']
    for pattern in ['release_receipt_T645R*.json','source_manifest_T645R*.json']:
        paths.extend(str(f.relative_to(ROOT)) for f in (ROOT/'stage/docs/long_distance_20261101').glob(pattern))
    try:
        subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True,capture_output=True)
        subprocess.run(['git','commit','-m','Report full384 retention and qualified recovery candidate with raw archive'],cwd=ROOT,check=True,capture_output=True)
        if time.time()+p['source_seconds']+15<p['absolute_all_stop']:
            subprocess.run([str(ROOT/'run_stage.sh'),str(ROOT/'stage/experiments/t629_offline_night_relay/t629_transport.py'),'source_sync'],
                 cwd=ROOT,capture_output=True,check=True,timeout=p['source_seconds'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    except BaseException:save(DOC/'SOURCE_PENDING.json',dict(error=traceback.format_exc()))
    print(json.dumps(dict(state=report['state'],remote=report['remote'],summary=s)),flush=True)


if __name__=='__main__':
    if sys.argv[1]=='resources':print(json.dumps(resources()))
    else:globals()[sys.argv[1]]()
