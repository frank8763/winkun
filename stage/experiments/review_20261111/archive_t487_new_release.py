"""One completed-run RAM archive on a dedicated low-asset-count Release.

No native/optimizer or original run mutation. Same manifest, binary assembly
and every-member SHA verifier as the registered archival helper.
"""
from pathlib import Path
import importlib.util,json,sys,time,urllib.error
ROOT=Path(__file__).resolve().parents[3];DOC=ROOT/'stage/docs/review_20261111'

def main():
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T487.json'
    assert not receipt.exists()
    assert time.time()+600<1794398866.5001101
    f=ROOT/'stage/experiments/t487_eight_only/transport.py'
    spec=importlib.util.spec_from_file_location('_t487_dedicated_release',f);T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T);T.install()
    T.stopped(ROOT/'stage/runs/t487_eight_only')
    T.T.B.GROUPS['T487']=['t487_eight_only'];T.T.B.require_stopped=T.stopped
    sys.path.insert(0,str(ROOT/'stage/experiments/t260_distance_archive'));import small_parts as S
    original=S.api;tag='stage-eight-only-2026-11-11'
    def api(method,path,*args,**kw):
        if method=='GET' and path=='releases/tags/stage-distance-2026-11-01':
            try:return original('GET','releases/tags/'+tag)
            except urllib.error.HTTPError as error:
                if error.code!=404:raise
                source=json.loads((ROOT/'stage/runs/t487_eight_only/launch.json').read_text())['source_remote']
                return original('POST','releases',dict(tag_name=tag,target_commitish=source,name='Stage wide-eight-only research evidence — 2026-11-11',draft=False,make_latest='false',
                    body='Completed Stage T487 only: wide8 development35/54, LEFT11/27 RIGHT24/27, one contact18timeouts. Eight-only research acceptance passed; no old/six/fresh navigation/reload validation, no global stability or safety claim. Full original evidence/models/failures; per-member SHA verification is recorded separately. No Phase data.'))
        return original(method,path,*args,**kw)
    S.api=api;S.main('T487')
    d=json.loads(receipt.read_text());assert d['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    (DOC/'T487_BACKUP_COMPLETION.json').write_text(json.dumps(dict(state=d['state'],epoch=time.time(),url=d['url'],members=d['members'],sha256=d['sha256'],segments=len(d['parts']),native=0,optimizer=0,original_pending_error_preserved=True,original_run_not_modified=True),indent=2)+'\n')
if __name__=='__main__':main()
