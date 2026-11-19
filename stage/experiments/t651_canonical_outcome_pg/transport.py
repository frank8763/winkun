"""Bounded post-science RAM exports; verified TLS fallback, no local deletion."""
from pathlib import Path
import hashlib,importlib.util,json,signal,sys,time,urllib.error,urllib.parse,urllib.request
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'stage/runs/t651_canonical_outcome_pg'
spec=importlib.util.spec_from_file_location('_t651_archive',ROOT/'stage/experiments/t493_wide8_night/transport.py')
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)

def read(f):return json.loads(Path(f).read_text())
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None

def client():
    A.OLD.install();t=A.OLD.T;curl=t.response
    def response(method,url,body=None,headers=None):
        host=urllib.parse.urlparse(url).hostname
        assert host in ['api.github.com','uploads.github.com','release-assets.githubusercontent.com','objects.githubusercontent.com']
        if host not in ['api.github.com','uploads.github.com']:assert not any(k.lower()=='authorization' for k in (headers or {}))
        try:
            request=urllib.request.Request(url,data=body,headers=headers or {},method=method)
            try:result=urllib.request.build_opener(NoRedirect()).open(request,timeout=15)
            except urllib.error.HTTPError as e:
                if e.code!=302:raise
                result=e
            with result:
                data=result.read(4*2**20+1);assert len(data)<=4*2**20
                return result.code,{k.lower():v for k,v in result.headers.items()},data
        except (urllib.error.URLError,TimeoutError,OSError):
            # Retry GET safely. Never retry ambiguous POST writes automatically.
            if method!='GET':raise
            return curl(method,url,body,headers)
    t.response=response;return t


def main():
    which=sys.argv[1];assert which in ['old','current'];name='T646' if which=='old' else 'T651'
    folder=ROOT/'stage/runs/t646_offline_night' if which=='old' else OUT
    p=read(OUT/'protocol.json')
    deadline=min(time.time()+p['old_backup_seconds' if which=='old' else 'backup_seconds']-5,p['absolute_all_stop']-p['source_seconds']-30)
    assert deadline>time.time()+30
    def stopped():
        s=read(folder/'SEALED.json');assert s['all_scientific_processes_stopped']
        assert all(not Path('/proc',str(i)).exists() for i in s['scientific_pids'])
    stopped();A.stopped=stopped
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T651 bounded evidence transfer')))
    signal.alarm(max(1,int(deadline-time.time())))
    t=client();release=t.B.api('GET','releases/tags/stage-eight-only-2026-11-11');receipts=[]
    for child in sorted(f for f in folder.iterdir() if f.is_dir() and f.name!='offline_inputs'):
        files=sorted(f for f in child.rglob('*') if f.is_file() and f.suffix not in ['.pyc','.lock','.partial'])
        groups=[];batch=[];n=0
        for f in files:
            if batch and n+f.stat().st_size>32*2**20:groups.append(batch);batch=[];n=0
            batch.append(f);n+=f.stat().st_size
        if batch:groups.append(batch)
        for i,group in enumerate(groups):receipts.append(A.archive(f'{name}_post_{child.name}_{i}',group,release,deadline,t))
    files=sorted(f for f in folder.iterdir() if f.is_file() and f.name!='driver.log' and f.suffix not in ['.pyc','.lock','.partial'])
    if files:receipts.append(A.archive(f'{name}_post_root',files,release,deadline,t))
    t.B.write(ROOT/f'stage/docs/long_distance_20261101/release_receipt_{name}_post.json',dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',group=name,child_receipts=[r['group'] for r in receipts],bytes=sum(r['bytes'] for r in receipts),files_deleted=0,offline_input_originals_previously_archived=True))
    signal.alarm(0);print(name+'_POST_RAW_REMOTE_VERIFIED')

if __name__=='__main__':main()
