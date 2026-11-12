"""Single bounded supervisor: native diagnosis, then one independent audit."""
from pathlib import Path
import hashlib,json,os,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t494_timeout_cause';DOC=ROOT/'stage/docs/review_20261112'
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
        save(DOC/'T494_FINAL_REPORT.json',dict(state='COMPLETE_AUDIT_PASS',summary=read(OUT/'summary.json'),audit=read(OUT/'audit.json'),remote='PENDING',no_training=True,epoch=time.time()))
    except BaseException:
        save(DOC/'T494_DRIVER_ERROR.json',dict(error=traceback.format_exc(),raw_preserved=True,no_restart=True,epoch=time.time()));raise
if __name__=='__main__':main()
