"""Completed-evidence transport only: new immutable asset name on HTTP422.

No science/model changes. Original partial assets, errors, manifest retained.
All assembled archive and per-member verifier logic is the original helper.
"""
from pathlib import Path
import importlib.util,json,sys,time,urllib.error,urllib.parse
ROOT=Path(__file__).resolve().parents[3]
DOC=ROOT/'stage/docs/review_20261111'
HERE=ROOT/'stage/experiments/t486_adam_memory_projection'

def main():
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T486.json'
    assert not receipt.exists(),'Already verified, no duplicate'
    assert time.time()<1794397742.2378583-180,'Original overall cutoff reserve'
    spec=importlib.util.spec_from_file_location('_t486_transport_recovery',HERE/'transport.py')
    T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T);T.install()
    T.T.B.GROUPS['T486']=['t486_adam_memory_projection','t483_local_log_reserve'];T.T.B.require_stopped=T.stopped
    for folder in T.T.B.GROUPS['T486']:T.stopped(ROOT/'stage/runs'/folder)
    sys.path.insert(0,str(ROOT/'stage/experiments/t260_distance_archive'));import small_parts as S
    original=S.req;suffix=str(int(time.time()))
    def upload(method,url,*args,**kw):
        try:return original(method,url,*args,**kw)
        except urllib.error.HTTPError as error:
            if method!='POST' or error.code!=422 or 'uploads.github.com/' not in url:raise
            parsed=urllib.parse.urlsplit(url);query=urllib.parse.parse_qs(parsed.query)
            name=query['name'][0];query['name']=[name.removesuffix('.bin')+'-verified-recovery-'+suffix+'.bin']
            recovery=urllib.parse.urlunsplit(parsed._replace(query=urllib.parse.urlencode(query,doseq=True)))
            return original(method,recovery,*args,**kw)
    S.req=upload
    S.main('T486')
    d=json.loads(receipt.read_text());assert d['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    (DOC/'T486_ARCHIVE_RECOVERY.json').write_text(json.dumps(dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',epoch=time.time(),
        original_HTTP422_and_partial_assets_preserved=True,native=0,optimizer=0,source_manifest_unchanged=True,
        archive_sha256=d['sha256'],members=d['members'],segments=len(d['parts'])),indent=2)+'\n')
if __name__=='__main__':main()
