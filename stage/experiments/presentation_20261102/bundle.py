"""User-facing PPT material ZIP, RAM-only; verify full remote bytes and members."""
from pathlib import Path
import sys,io,json,zipfile,urllib.parse,hashlib
ROOT=Path(__file__).resolve().parents[3];DOC=ROOT/'stage/docs/presentation_20261102'
sys.path.insert(0,str(ROOT/'stage/experiments/t260_distance_archive'))
from backup import readback
from archive import api,req,write,SLUG
sha=lambda b:hashlib.sha256(b).hexdigest()

def main():
    dest=DOC/'MATERIALS_BUNDLE_RECEIPT.json';assert not dest.exists()
    videos=json.loads((DOC/'VIDEO_DOWNLOAD_RECEIPT.json').read_text());assert videos['state']=='BOTH_INDIVIDUAL_MP4_FULL_REMOTE_BINARY_READBACK_SHA_VERIFIED'
    files={}
    for row in videos['files']:
        path=Path(row['local_path']);blob=path.read_bytes();assert sha(blob)==row['sha256'];files[path.name]=blob
    for name in ['PPT_CONTENT.md','VIDEO_INFO.md','VIDEO_AUDIT.json','stage_method.png','stage_method.svg','stage_distance_results.png','stage_distance_results.svg']:files[name]=(DOC/name).read_bytes()
    files['README.txt']=('Stage汇报素材（2026-11-02）\n\n两段MP4均为1920x1080/25fps/H264，使用同一个冻结T255 s2_reset模型复现原五米成功任务。\n原始动作/120维观测轨迹与最终物理积分已精确核验；无训练、地图/规划器/路点/运行时教师。\n视频开头0.8秒、成功后1.6秒展示暂停，导航按正常1倍速播放。\n请将MP4保存到本地后插入PPT。PNG为1920x1080图，SVG为矢量图；PPT_CONTENT.md为六页汇报文字。\n六米仍0/54，七米/八米未进入；五米是当前原场景局部验证，不能表述为所有地图或从零纯RL。\n').encode()
    manifest=[dict(path=k,bytes=len(v),sha256=sha(v)) for k,v in files.items()]
    files['SHA256SUMS.txt']=''.join(f'{r["sha256"]}  {r["path"]}\n' for r in manifest).encode()
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w') as z:
        for name,blob in files.items():
            info=zipfile.ZipInfo(name,(2026,11,2,0,0,0));info.compress_type=zipfile.ZIP_STORED if name.endswith('.mp4') else zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16;z.writestr(info,blob)
    data=buffer.getvalue();name='stage_presentation_20261102-'+sha(data)[:12]+'.zip';release=api('GET','releases/tags/stage-five-metre-videos-2026-11-02');assets=api('GET',f'releases/{release["id"]}/assets?per_page=100')
    asset=next((a for a in assets if a['name']==name),None)
    if asset is None:asset=json.loads(req('POST',f'https://uploads.github.com/repos/{SLUG}/releases/{release["id"]}/assets?name='+urllib.parse.quote(name),data,{'Content-Type':'application/zip'}))
    asset=api('GET',f'releases/assets/{asset["id"]}');assert asset['size']==len(data) and asset['digest']=='sha256:'+sha(data)
    returned=readback(asset,len(data));assert returned==data
    with zipfile.ZipFile(io.BytesIO(returned)) as z:
        assert set(z.namelist())==set(files)
        for k,v in files.items():assert z.read(k)==v
    write(dest,dict(state='ZIP_FULL_REMOTE_BINARY_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',url=asset['browser_download_url'],asset_id=asset['id'],bytes=len(data),sha256=sha(data),members=len(files),files=manifest,temporary_disk_archive_bytes=0))
    print(json.dumps(dict(state='BUNDLE_VERIFIED',url=asset['browser_download_url'],bytes=len(data),members=len(files))),flush=True)

if __name__=='__main__':main()
