"""Archive only ended Stage T225–T229 raw evidence, RAM-only exact readback."""
from archive import *

def main():
    names=[]
    for i in range(225,230):
        candidates=list((ROOT/'stage/runs').glob(f't{i}_*'))
        if not candidates:continue
        assert len(candidates)==1
        folder=candidates[0];s=json.loads((folder/'summary.json').read_text());assert s['all_workers_stopped'] if 'all_workers_stopped' in s else s['state'] in ('COMPLETE','FAILED')
        names.append(folder.name)
    assert {'t225_paired_coverage','t226_local_exploration','t227_small_noise_recovery','t228_balanced_stability'}.issubset(names)
    tag='stage-local-stability-2026-10-31';receipt=DOC/'stability_release_receipt.json';assert not receipt.exists(),'Inspect existing verified receipt'
    files=sorted(f for name in names for f in (ROOT/'stage/runs'/name).rglob('*') if f.is_file() and f.suffix not in ('.lock','.partial','.pyc'))
    manifest=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in files];assert sum(r['bytes'] for r in manifest)<25*1024**2
    write(DOC/'stability_source_manifest.json',dict(scope='EndedStageT225–T229rawsources/failures/finalmodels/frozenacceptance; no Phase/credentials/chats',files=manifest))
    output=io.BytesIO()
    with tarfile.open(fileobj=output,mode='w:gz',compresslevel=3) as archive:
        for p,r in zip(files,manifest):assert sha(p.read_bytes())==r['sha256'];archive.add(p,arcname=r['path'],recursive=False)
    data=output.getvalue();verify_tar(data,manifest)
    try:release=api('GET','releases/tags/'+tag)
    except urllib.error.HTTPError as e:
        if e.code!=404:raise
        release=api('POST','releases',dict(tag_name=tag,target_commitish=api('GET','git/ref/heads/main')['object']['sha'],draft=False,name='Stage local 3m–5m stability investigation and retained failures',body='Stage only. Frozen paired exploration and bounded self-success consolidation, original unchanged native MuJoCo geometry/GOAL. Includes all failures, T226 recording failure and bounded T227 recovery; distinguish supervised consolidation from new RL updates. Actual gate outcomes in committed findings/summary; no assumed success/generalization/algorithm novelty. Raw models/trajectories in this Release rather than Git. RAM archive, full remote binary readback and every-member SHA/size verified. No Phase, chat histories or credentials.'))
    assets=api('GET',f"releases/{release['id']}/assets?per_page=100");name='stage-local-stability-'+sha(data)[:12]+'.tar.gz'
    existing=[a for a in assets if a['name'].startswith('stage-local-stability-') and a['name'].endswith('.tar.gz')]
    if existing:assert len(existing)==1;asset=api('GET',f"releases/assets/{existing[0]['id']}")
    else:asset=json.loads(req('POST',f"https://uploads.github.com/repos/{SLUG}/releases/{release['id']}/assets?name="+name,data,{'Content-Type':'application/octet-stream'}))
    remote=req('GET',f"https://api.github.com/repos/{SLUG}/releases/assets/{asset['id']}",headers={'Accept':'application/octet-stream'});assert len(remote)==asset['size'] and 'sha256:'+sha(remote)==asset['digest'];verify_tar(remote,manifest)
    index=(DOC/'stability_source_manifest.json').read_bytes();old=next((a for a in assets if a['name']=='stability_source_manifest.json'),None)
    if old:assert old['digest']=='sha256:'+sha(index)
    else:old=json.loads(req('POST',f"https://uploads.github.com/repos/{SLUG}/releases/{release['id']}/assets?name=stability_source_manifest.json",index,{'Content-Type':'application/octet-stream'}));assert old['size']==len(index) and old['digest']=='sha256:'+sha(index)
    write(receipt,dict(state='PUBLISHED_FULL_REMOTE_BINARY_AND_MEMBER_READBACK_VERIFIED',url=release['html_url'],asset_id=asset['id'],asset_url=asset['browser_download_url'],bytes=len(remote),sha256=sha(remote),members=len(manifest),source_roots=names,temporary_disk_archive_bytes=0,local_raw_evidence_deleted=0))
    print(json.dumps(dict(state='ARCHIVED_AND_FULLY_VERIFIED',url=release['html_url'],members=len(manifest),bytes=len(remote))),flush=True)
if __name__=='__main__':main()
