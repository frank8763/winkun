"""Bounded full54 intervention supervisor; no automatic parameter sweeps."""
from pathlib import Path
import hashlib,json,os,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t495_drive_head_cause';DOC=ROOT/'stage/docs/review_20261112'
def read(f):return json.loads(Path(f).read_text())
def save(f,d):
    p=Path(f);q=Path(str(p)+'.partial');q.write_text(json.dumps(d,ensure_ascii=False)+'\n');q.replace(p)
def main():
    p=read(OUT/'protocol.json');source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert source['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    assert not (OUT/'launch.json').exists()
    save(OUT/'launch.json',dict(pid=os.getpid(),epoch=time.time(),all_stop=p['absolute_all_stop'],source_remote=source['remote_commit'],protocol_sha256=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),CPU=[14,15],CPUQuota='100%',GPU=False))
    try:
        for action,cap in [('main',p['main_seconds']+5),('audit',p['audit_seconds'])]:
            remain=min(cap,p['absolute_all_stop']-time.time()-10);assert remain>0
            with (OUT/(action+'.log')).open('xb') as log:
                c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),action],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=remain)
                assert c.returncode==0,(action,c.returncode)
        files=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(OUT.rglob('*')) if f.is_file() and f.suffix not in ['.partial','.lock']]
        save(OUT/'local_manifest.json',dict(state='LOCAL_ALL_SHA_PASS_REMOTE_PENDING',files=files))
        result=read(OUT/'summary.json')
        save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,finalizer_writes_only_docs_after_this_point=True,epoch=time.time()))
        remote='PENDING'
        try:
            remain=min(580,p['absolute_all_stop']-time.time()-180);assert remain>40
            c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'t495_transport.py'),'T494_T495'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=remain)
            assert c.returncode==0,c.stdout.decode(errors='replace')[-2500:]
            remote=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T495.json')['state']
        except BaseException:save(DOC/'T495_REMOTE_PENDING.json',dict(error=traceback.format_exc(),raw_preserved=True))
        save(DOC/'T495_FINAL_REPORT.json',dict(state='COMPLETE_AUDIT_PASS',summary=result,audit=read(OUT/'audit.json'),remote=remote,not_new_RL_training=True,epoch=time.time()))
        status=ROOT/'stage/docs/RESEARCH_STATUS.md'
        status.write_text('2026-11-12 T495单一油门神经头因果校准完成，'+str(result['result']['wide8'])+'；仅开发验收，0新增Actor/Value Adam，不称新RL算法或八米稳定。独立auditPASS，远端'+remote+'，旧模型保留；详review_20261112/T495_FINAL_REPORT。Phase隔离。\n\n'+status.read_text())
        subprocess.run(['git','add','stage/docs/review_20261112','stage/docs/long_distance_20261101','stage/docs/RESEARCH_STATUS.md'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        subprocess.run(['git','commit','-m','Report bounded fourteen-timeout drive-head causal pilot'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            remain=min(120,p['absolute_all_stop']-time.time()-5);assert remain>0
            c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'t495_transport.py'),'source_sync'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=remain)
            assert c.returncode==0,c.stdout.decode(errors='replace')[-2500:]
        except BaseException:save(DOC/'T495_SOURCE_PENDING.json',dict(error=traceback.format_exc(),local_committed=True))
    except BaseException:
        save(DOC/'T495_DRIVER_ERROR.json',dict(error=traceback.format_exc(),raw_preserved=True,no_restart=True,epoch=time.time()));raise
if __name__=='__main__':main()
