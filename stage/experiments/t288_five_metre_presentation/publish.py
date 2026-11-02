from pathlib import Path
import sys,json,urllib.parse
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'stage/runs/t288_five_metre_presentation';DOC=ROOT/'stage/docs/presentation_20261102'
sys.path.insert(0,str(ROOT/'stage/experiments/t260_distance_archive'))
from backup import readback
from archive import api,req,sha,write,SLUG


def main():
    receipt=DOC/'VIDEO_DOWNLOAD_RECEIPT.json';assert not receipt.exists()
    assert json.loads((DOC/'VIDEO_AUDIT.json').read_text())['state']=='PASS'
    raw=json.loads((ROOT/'stage/docs/long_distance_20261101/release_receipt_T288.json').read_text());assert raw['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    s=json.loads((OUT/'summary.json').read_text());tag='stage-five-metre-videos-2026-11-02'
    import urllib.error
    try:release=api('GET','releases/tags/'+tag)
    except urllib.error.HTTPError as e:
        if e.code!=404:raise
        source=api('GET','git/ref/heads/main')['object']['sha']
        release=api('POST','releases',dict(tag_name=tag,target_commitish=source,name='Stage five-metre verified MuJoCo videos | 1080p 25fps',draft=False,
            body='Two frozen T255 s2_reset known-task five-metre native MuJoCo success reproductions, LEFT and RIGHT. No new training or long-distance success claim. Every original ten arrays and final integration hash match accepted baseline. Render-only overview/close-up, markers and retrospective trace. 0.8s introduction and1.6s goal pause; navigation nominal1x. H264/yuv420p/1920x1080/25fps. Complete original raw evidence separately archived; individual MP4 assets fully downloaded back and SHA verified.'))
    assets=api('GET',f'releases/{release["id"]}/assets?per_page=100');records=[]
    for recording in s['recordings']:
        meta=recording['movie'];path=Path(meta['path']);blob=path.read_bytes();assert sha(blob)==meta['sha256']
        name=path.name;asset=next((a for a in assets if a['name']==name),None)
        if asset is None:asset=json.loads(req('POST',f'https://uploads.github.com/repos/{SLUG}/releases/{release["id"]}/assets?name='+urllib.parse.quote(name),blob,{'Content-Type':'video/mp4'}))
        asset=api('GET',f'releases/assets/{asset["id"]}');assert asset['size']==len(blob) and asset['digest']=='sha256:'+meta['sha256']
        returned=readback(asset,len(blob));assert returned==blob
        records.append(dict(side=recording['side'],local_path=str(path),download_url=asset['browser_download_url'],asset_id=asset['id'],bytes=len(blob),sha256=sha(returned),resolution='1920x1080',fps=25,duration_s=meta['duration_s'],native_time_s=recording['native']['physics_steps']*.002,remaining_m=recording['native']['remaining_m']))
    write(receipt,dict(state='BOTH_INDIVIDUAL_MP4_FULL_REMOTE_BINARY_READBACK_SHA_VERIFIED',release_url=release['html_url'],files=records,raw_evidence_receipt=str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T288.json'),temporary_disk_archive_bytes=0))
    print(json.dumps(dict(state='DOWNLOADS_VERIFIED',files=records)),flush=True)


if __name__=='__main__':main()
