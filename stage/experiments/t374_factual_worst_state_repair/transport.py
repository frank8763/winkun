"""Transport-only curl TLS fallback; original archive/member checks stay active.

Credentials are supplied through an inherited configuration pipe, never argv,
logs or a disk file. Asset redirects are validated and fetched without auth.
"""
from pathlib import Path
import sys,os,json,subprocess,uuid,urllib.error,urllib.parse,time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t260_distance_archive'))
import backup as B
import archive as A


def response(method,url,body=None,headers=None):
    marker=('\nSTAGE_CURL_META_'+uuid.uuid4().hex+'\n').encode()
    config='url = '+json.dumps(url)+'\n'+''.join('header = '+json.dumps(k+': '+v)+'\n' for k,v in (headers or {}).items())
    read_fd,write_fd=os.pipe()
    try:
        os.write(write_fd,config.encode());os.close(write_fd);write_fd=None
        args=['curl','--silent','--show-error','--connect-timeout','8','--max-time','30','--config',f'/dev/fd/{read_fd}','--request',method,'--write-out',marker.decode()+'%{http_code}\n%{header_json}']
        if body is not None:args+=['--data-binary','@-']
        result=subprocess.run(args,input=body,stdout=subprocess.PIPE,stderr=subprocess.PIPE,pass_fds=(read_fd,))
        if result.returncode:raise urllib.error.URLError('curl transport exit '+str(result.returncode))
        data,metadata=result.stdout.rsplit(marker,1);code,raw=metadata.split(b'\n',1);fields=json.loads(raw)
        return int(code),{k.lower():v[0] for k,v in fields.items()},data
    finally:
        os.close(read_fd)
        if write_fd is not None:os.close(write_fd)


def req(method,url,payload=None,headers=None):
    assert urllib.parse.urlparse(url).hostname in ('api.github.com','uploads.github.com')
    fields={'Authorization':'Bearer '+A.token(),'Accept':'application/vnd.github+json','User-Agent':'Stage-byte-verified-curl-recovery',**(headers or {})}
    body=json.dumps(payload).encode() if isinstance(payload,dict) else payload
    if isinstance(payload,dict):fields['Content-Type']='application/json'
    for attempt in range(3 if method=='GET' else 1):
        try:
            code,_,data=response(method,url,body,fields)
            if code>=400:raise urllib.error.HTTPError(url,code,'GitHub HTTP status '+str(code),{},None)
            return data
        except urllib.error.URLError:
            if method!='GET' or attempt==2:raise
    raise AssertionError('unreachable')


def readback(asset,size):
    fields={'Authorization':'Bearer '+A.token(),'Accept':'application/octet-stream','User-Agent':'Stage-evidence-curl-readback'}
    code,headers,data=response('GET',f'https://api.github.com/repos/{A.SLUG}/releases/assets/{asset["id"]}',headers=fields)
    if code==200:assert len(data)==size;return data
    assert code==302;url=headers['location'];assert urllib.parse.urlparse(url).hostname in ('release-assets.githubusercontent.com','objects.githubusercontent.com')
    for attempt in range(3):
        try:
            code,fields,data=response('GET',url,headers={'Range':f'bytes=0-{size-1}','User-Agent':'Stage-evidence-curl-readback'})
            assert code==206 and fields['content-range'].startswith(f'bytes 0-{size-1}/') and len(data)==size
            return data
        except urllib.error.URLError:
            if attempt==2:raise


def install():
    A.req=req;B.req=req;B.readback=readback


if __name__=='__main__':
    install()
    if sys.argv[1:]==['T374']:
        import small_parts
        small_parts.main('T374')
    elif sys.argv[1:]==['source_sync']:
        import sync_source
        sync_source.main()
    else:raise ValueError('Only completed T374 archive or exact source synchronization')
