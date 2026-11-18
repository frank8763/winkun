"""Finite offline branch supervisor; no LLM/network science dependency."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t646_offline_night';DOC=ROOT/'stage/docs/overnight_20261118'
sys.path.insert(0,str(ROOT/'stage/experiments/t645_iterative_recovery_relay'))
import control as Resources


def read(f):return json.loads(Path(f).read_text())
def save(f,v):Path(f).write_text(json.dumps(v,indent=2)+'\n')
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def run_action(action,name,seconds,p,children):
    assert time.time()+seconds+30<p['absolute_all_stop']
    folder=OUT/name
    with (folder/(action+'.log')).open('xb') as log:
        c=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'t646_runtime.py'),action,name],cwd=ROOT,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CUBLAS_WORKSPACE_CONFIG=':4096:8'))
        children.append(c.pid)
        try:c.wait(timeout=seconds);assert c.returncode==0,(action,name,c.returncode)
        except BaseException:
            try:os.killpg(c.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:c.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
            raise


def seal_science(children):
    ids=list(children)
    for path in OUT.glob('*/worker_generations.json'):
        for g in read(path):ids.extend([g['parent'],*g['workers']])
    ids=sorted(set(ids));gone=all(not Path('/proc',str(i)).exists() for i in ids)
    save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=gone,scientific_pids=ids,epoch=time.time()))
    assert gone


def main():
    p=read(OUT/'protocol.json')
    assert p['authorization_approved'] and sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert not (OUT/'DRIVER_STARTED.json').exists()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    inputs=read(OUT/'OFFLINE_INPUTS.json');assert len(inputs['members'])==384
    for row in inputs['members']:assert sha(row['offline_path'])==row['sha256']
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    Resources.resources()
    save(OUT/'DRIVER_STARTED.json',dict(pid=os.getpid(),epoch=time.time(),source_remote=receipt['remote_commit'],offline_inputs_ready=True,all_stop=p['absolute_all_stop']))
    children=[];results=[];error=None;main_used=0.
    try:
        for name in p['arms']:
            assert time.time()+p['main_seconds']+p['audit_seconds']+p['input_audit_seconds']+p['finalization_reserve_seconds']<p['absolute_all_stop']
            Resources.resources();save(OUT/'status.json',dict(state='BRANCH_ACTIVE',arm=name,epoch=time.time(),completed=len(results)))
            started=time.time();run_action('verify_inputs',name,p['input_audit_seconds'],p,children);main_used+=time.time()-started
            started=time.time();run_action('main',name,p['main_seconds'],p,children);main_used+=time.time()-started
            run_action('audit',name,p['audit_seconds'],p,children)
            s=read(OUT/name/'summary.json');a=read(OUT/name/'audit.json');assert a['state']=='PASS'
            results.append(dict(arm=name,state=s['state'],rounds=len(s['rounds']),physics=s['physics'],commands=s['commands'],
                live_started=s['actual_live_Actor_started'],stop_reason=s['stop_reason'],final407=s['final407'],retained=s['retained']))
            assert sum(v['physics'] for v in results)<=p['night_max_physics']
            assert sum(v['commands'] for v in results)<=p['night_max_commands']
            assert sum(v['live_started'] for v in results)<=p['night_Actor_each_cap']
            save(OUT/'PROGRESS.json',dict(state='AUDITED_BRANCH_COMPLETE',results=results,epoch=time.time()))
            save(OUT/'LEDGER.json',dict(physics=sum(v['physics'] for v in results),commands=sum(v['commands'] for v in results),Actor_live_each=sum(v['live_started'] for v in results)))
        if time.time()+p['profile_seconds']+p['profile_audit_seconds']+p['finalization_reserve_seconds']<p['absolute_all_stop'] and time.time()+p['profile_seconds']<p['absolute_native_stop']:
            started=time.time();run_action('profile_main','profile',p['profile_seconds'],p,children);main_used+=time.time()-started
            run_action('profile_audit','profile',p['profile_audit_seconds'],p,children)
            assert read(OUT/'profile/audit.json')['state']=='PASS'
            s=read(OUT/'profile/summary.json');results.append(dict(arm='profile',physics=s['physics'],commands=s['commands'],live_started=0,success=s['success'],fresh=s['fresh']))
            assert sum(v['physics'] for v in results)<=p['night_max_physics'] and sum(v['commands'] for v in results)<=p['night_max_commands']
        else:save(OUT/'PROFILE_NOT_RUN.json',dict(reason='insufficient full remaining time'))
        available=min(p['coverage_seconds'],p['night_main_seconds']-main_used,p['absolute_all_stop']-time.time()-p['coverage_audit_seconds']-p['finalization_reserve_seconds']-60)
        if available>900:
            child=read(OUT/'coverage/protocol.json');child['main_seconds']=available-30;child['audit_seconds']=p['coverage_audit_seconds'];save(OUT/'coverage/protocol.json',child)
            run_action('coverage_main','coverage',available,p,children)
            run_action('coverage_audit','coverage',p['coverage_audit_seconds'],p,children)
            assert read(OUT/'coverage/audit.json')['state']=='PASS'
            s=read(OUT/'coverage/summary.json');results.append(dict(arm='coverage',physics=s['physics'],commands=s['commands'],live_started=0,success=s['success'],fresh=s['fresh']))
            assert sum(v['physics'] for v in results)<=p['night_max_physics'] and sum(v['commands'] for v in results)<=p['night_max_commands']
        else:save(OUT/'COVERAGE_NOT_RUN.json',dict(reason='insufficient full remaining time'))
    except BaseException:error=traceback.format_exc()
    seal_science(children)
    report=dict(state='FAILED' if error else 'COMPLETE',error=error,results=results,
        all_scientific_processes_stopped=True,remote='PENDING_LOCAL_OFFLINE_RAW_RETAINED',
        no_unseen_geometry_or_robot_claim=True,science_did_not_require_network=True)
    save(OUT/'FINAL_REPORT.json',report);save(DOC/'FINAL_REPORT.json',report)
    files=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f))
        for f in sorted(OUT.rglob('*')) if f.is_file() and f.name not in ['driver.log','local_manifest.json']]
    save(OUT/'local_manifest.json',dict(state='LOCAL_ALL_ORIGINAL_BYTES_SHA',files=files))
    # One bounded export only after every scientific process is gone. The
    # input cache is already remotely archived; never upload another copy.
    if time.time()+p['backup_seconds']+p['source_seconds']+60<p['absolute_all_stop']:
        try:
            c=subprocess.run([str(ROOT/'run_stage.sh'),str(HERE/'transport.py')],cwd=ROOT,capture_output=True,
               timeout=p['backup_seconds'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
            assert c.returncode==0,c.stderr.decode(errors='replace')[-1400:]
            report['remote']=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T646.json')['state']
        except BaseException:report['backup_error']=traceback.format_exc()
    save(OUT/'FINAL_REPORT.json',report);save(DOC/'FINAL_REPORT.json',report)
    status=ROOT/'stage/docs/RESEARCH_STATUS.md'
    status.write_text('2026-11-19 T646离线多机制夜间接力结束：'+report['state']+'，remote='+report['remote']+
       '；全部实际轮次/384保持/条件407见overnight_20261118/FINAL_REPORT，不称泛化或真机通过。\n\n'+status.read_text())
    try:
        paths=[str(DOC.relative_to(ROOT)),'stage/docs/RESEARCH_STATUS.md']
        for pattern in ['release_receipt_T646*.json','source_manifest_T646*.json']:
            paths.extend(str(f.relative_to(ROOT)) for f in (ROOT/'stage/docs/long_distance_20261101').glob(pattern))
        subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True,capture_output=True)
        subprocess.run(['git','commit','-m','Report finite offline T646 mechanisms and preserved evidence'],cwd=ROOT,check=True,capture_output=True)
        if time.time()+p['source_seconds']+15<p['absolute_all_stop']:
            subprocess.run([str(ROOT/'run_stage.sh'),str(ROOT/'stage/experiments/t629_offline_night_relay/t629_transport.py'),'source_sync'],
                cwd=ROOT,check=True,capture_output=True,timeout=p['source_seconds'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    except BaseException:save(DOC/'SOURCE_PENDING.json',dict(error=traceback.format_exc()))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
