"""Verified TLS with bounded retries; no credentials or signed URLs in output."""
from pathlib import Path
import sys,time,urllib.request,urllib.error,urllib.parse
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t637_joint_coverage_recovery'))
import t637_fetch as Old
ORIGINAL_CLIENT=Old.S.client


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None


def response(method,url,body=None,headers=None):
    assert method=='GET' and body is None
    host=urllib.parse.urlparse(url).hostname
    assert host in ['api.github.com','release-assets.githubusercontent.com','objects.githubusercontent.com']
    if host!='api.github.com':assert not any(k.lower()=='authorization' for k in (headers or {}))
    last=None
    for attempt in range(3):
        try:
            opener=urllib.request.build_opener(NoRedirect())
            request=urllib.request.Request(url,headers=headers or {},method=method)
            try:result=opener.open(request,timeout=10)
            except urllib.error.HTTPError as e:
                if e.code!=302:raise
                result=e
            with result:
                data=result.read(4*2**20+1)
                assert len(data)<=4*2**20,'Asset part exceeds fixed transport cap'
                return result.code,{k.lower():v for k,v in result.headers.items()},data
        except (urllib.error.URLError,TimeoutError,OSError) as e:
            last=type(e).__name__
            if attempt<2:time.sleep(.5)
    raise RuntimeError('Bounded verified TLS GET failed: '+str(last))


def client():
    t=ORIGINAL_CLIENT();t.response=response;return t


if __name__=='__main__':
    Old.S.client=client
    Old.main()
