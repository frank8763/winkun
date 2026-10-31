"""Bounded RAM-only original-evidence Release backup; never deletes evidence."""
from pathlib import Path
import sys,io,json,tarfile,time,urllib.request,urllib.error,urllib.parse,http.client,subprocess
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t255_overnight';DOC=ROOT/'stage/docs/overnight_20261031'
sys.path.insert(0,str(ROOT/'stage/experiments/t224_archive'))
from archive import api,req,sha,token,OPENER,verify_tar,write,SLUG
def main():
    assert not (DOC/'release_receipt.json').exists(),'No duplicate backup'
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.suffix not in ['.lock','.partial','.pyc'] and p.name not in ['backup.log','backup_process.json','supervisor_summary.json'])
    manifest=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in files];assert sum(r['bytes'] for r in manifest)<150*1024**2
    write(DOC/'source_manifest.json',dict(scope='T255 Stage all original learning, audit, failures, final models, optimizer states and exact traces. No Phase/chat/credentials. No deletion.',files=manifest));buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz',compresslevel=3) as archive:
        for p,r in zip(files,manifest):assert sha(p.read_bytes())==r['sha256'];archive.add(p,arcname=r['path'],recursive=False)
    data=buf.getvalue();verify_tar(data,manifest);tag='stage-overnight-2026-10-31'
    try:release=api('GET','releases/tags/'+tag)
    except urllib.error.HTTPError as e:
        if e.code!=404:raise
        source=json.loads((ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json').read_text())['remote_commit'];release=api('POST','releases',dict(tag_name=tag,target_commitish=source,name='Stage bounded overnight memory comparison (2026-10-31)',draft=False,body='Stage-only raw research evidence. Read MORNING_REPORT and summary for actual outcomes; no promise of success, generalized stability or novelty. Includes all failures and final Actor/Adam. Original models retained locally.'))
    assets=api('GET',f"releases/{release['id']}/assets?per_page=100");name='stage-overnight-'+sha(data)[:12]+'.tar.gz';asset=next((a for a in assets if a['name']==name),None)
    if asset is None:asset=json.loads(req('POST',f"https://uploads.github.com/repos/{SLUG}/releases/{release['id']}/assets?name="+urllib.parse.quote(name),data,{'Content-Type':'application/octet-stream'}))
    asset=api('GET',f"releases/assets/{asset['id']}");assert asset['size']==len(data) and asset['digest']=='sha256:'+sha(data)
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,req,fp,code,msg,headers,newurl):return None
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect());request=urllib.request.Request(f"https://api.github.com/repos/{SLUG}/releases/assets/{asset['id']}",headers={'Authorization':'Bearer '+token(),'Accept':'application/octet-stream','User-Agent':'Stage-byte-verified-archive'})
    try:
        with opener.open(request,timeout=30) as r:remote=r.read()
    except urllib.error.HTTPError as e:
        assert e.code==302;url=e.headers['Location'];assert urllib.parse.urlparse(url).hostname in ('release-assets.githubusercontent.com','objects.githubusercontent.com');parts=[]
        for begin in range(0,len(data),1024**2):
            end=min(len(data),begin+1024**2)-1
            for attempt in range(5):
                try:
                    with OPENER.open(urllib.request.Request(url,headers={'Range':f'bytes={begin}-{end}','User-Agent':'Stage-byte-verified-archive'}),timeout=35) as r:
                        assert r.status==206 and r.headers['Content-Range'].startswith(f'bytes {begin}-{end}/');b=r.read();assert len(b)==end-begin+1
                    break
                except (OSError,http.client.HTTPException):
                    if attempt==4:raise
            parts.append(b)
        remote=b''.join(parts)
    assert remote==data;verify_tar(remote,manifest);write(DOC/'release_receipt.json',dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',url=release['html_url'],asset_url=asset['browser_download_url'],asset_id=asset['id'],bytes=len(data),sha256=sha(data),members=len(manifest),temporary_disk_archive_bytes=0,files_deleted=0))
    report=DOC/'MORNING_REPORT.md';report.write_text(report.read_text()+f'\n原始数据/模型归档已完整远端读回、逐成员SHA核验：[GitHub Release]({release["html_url"]})。本地未删除。\n')
    source=ROOT/'stage/docs/RESEARCH_STATUS.md';summary=json.loads((OUT/'summary.json').read_text()) if (OUT/'summary.json').exists() else {};source.write_text('T255今夜独立有界协议已结束。5米严格门槛='+str(summary.get('stable_5m_gate',False))+'；实际分组结果和失败见overnight_20261031/MORNING_REPORT.md及stage/runs/t255_overnight/summary.json。不重启/续训；T239/T250与全部失败保留，Phase隔离。\n\n'+source.read_text())
    paths=[str(p.relative_to(ROOT)) for p in [DOC/'source_manifest.json',DOC/'release_receipt.json',report,source]]
    subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True);subprocess.run(['git','commit','-m','Record bounded Stage overnight outcomes and verified raw evidence backup'],cwd=ROOT,check=True)
    subprocess.run([str(ROOT/'run_stage.sh'),str(ROOT/'stage/experiments/t224_archive/sync_source.py')],cwd=ROOT,check=True,timeout=180)
if __name__=='__main__':main()
