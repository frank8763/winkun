"""Bounded post-science RAM exports; verified TLS fallback, no local deletion."""
from pathlib import Path
import hashlib,importlib.util,json,signal,sys,time,urllib.error,urllib.parse,urllib.request
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'stage/runs/t650_reset_contract_repair'
spec=importlib.util.spec_from_file_location('_t649_archive',ROOT/'stage/experiments/t493_wide8_night/transport.py')
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
    p=read(OUT/'protocol.json');deadline=min(time.time()+p['backup_seconds']-5,p['absolute_all_stop']-p['source_seconds']-30)
    def stopped():
        s=read(OUT/'SEALED.json');assert s['all_scientific_processes_stopped'] and all(not Path('/proc',str(i)).exists() for i in s['scientific_pids'])
    stopped();A.stopped=stopped;signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T650 bounded backup')));signal.alarm(max(1,int(deadline-time.time())))
    t=client();release=t.B.api('GET','releases/tags/stage-eight-only-2026-11-11')
    files=sorted(f for f in OUT.rglob('*') if f.is_file() and f.name not in ['driver.log','affinity.log'] and f.suffix not in ['.lock','.partial','.pyc'])
    A.archive('T650_all',files,release,deadline,t);signal.alarm(0);print('T650_FULL_RAW_REMOTE_VERIFIED')

if __name__=='__main__':main()
