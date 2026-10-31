"""Remove only verified zero-update initialization copies; retain failure evidence."""
from support import *
def main():
    folder=ROOT/'stage/runs/t218_compact_rnd';receipt=folder/'initialization_copy_receipt.json'
    assert not receipt.exists(),'Already deduplicated'
    summary=json.loads((folder/'summary.json').read_text());plan=json.loads((folder/'protocol.json').read_text())
    assert summary['state']=='FAILED' and summary['physics_steps']==summary['commands']==0
    assert all(sum(summary[key].values())==0 for key in ('Actor_updates','Value_updates','RND_updates'))
    originals={kind:Path(plan[key]) for kind,key in [('Actor','checkpoint'),('RND','rnd_checkpoint')]}
    expected={kind:torch.load(path,map_location='cpu',weights_only=False)['policy' if kind=='Actor' else 'RND'] for kind,path in originals.items()}
    records=[]
    for arm,pair in summary['exports'].items():
        for kind,meta in pair.items():
            path=Path(meta['path']);assert path.parent==folder
            assert C.sha(path)==meta['sha256'] and path.stat().st_size==meta['bytes']
            obj=torch.load(io.BytesIO(gzip.decompress(path.read_bytes())),map_location='cpu',weights_only=False)
            key='policy' if kind=='Actor' else 'RND';assert digest(obj[key])==digest(expected[kind])
            assert all(torch.equal(v,expected[kind][k]) for k,v in obj[key].items())
            records.append(dict(path=str(path),sha256=C.sha(path),bytes=path.stat().st_size,
                source=str(originals[kind]),source_sha256=C.sha(originals[kind]),state_sha256=digest(obj[key]),
                payload_metadata={k:v for k,v in obj.items() if k!=key},state_key=key,
                reason='Failed before any update; tensor-exact redundant initialization export, not a learned model'))
    C.save(receipt,{'state':'VERIFIED_BEFORE_REMOVAL','original_summary_sha256':C.sha(folder/'summary.json'),'files':records,
        'preserved':'Original protocol, failure, logs, summary, initial source models remain; historical export paths resolved by this receipt.'})
    for rec in records:
        assert C.sha(rec['source'])==rec['source_sha256'] and C.sha(rec['path'])==rec['sha256']
        Path(rec['path']).unlink()
    data=json.loads(receipt.read_text());data['state']='REMOVED_VERIFIED_REDUNDANT_INITIALIZATION_COPIES';data['bytes_released']=sum(r['bytes'] for r in records);C.save(receipt,data)
    print(json.dumps({'bytes_released':data['bytes_released'],'all_runs_MiB':used(ROOT/'stage/runs')/1024**2}))
if __name__=='__main__':main()
