"""Finish the preserved zero-navigation storage-contract repair, once only."""
from pathlib import Path
import ast,hashlib,json,os
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t280_local_cost'


def main():
    assert not any((OUT/f).exists() for f in ['launch.json','status.json','summary.json','preflight.json'])
    p=json.loads((HERE/'protocol.json').read_text());assert not p.get('capacity_final_relocked')
    p.update(metadata_bytes=7*1024**2,final_bytes=64*1024**2,compact_episode_bytes=12288,capacity_final_relocked=True)
    p['preflight_engineering_repair']+='; preflight_failed_003 exact6MiB metadata upper6299809>6291456 stopped beforeclones; metadata7MiB within unchangedown64MiB. Inline relock missinghashlib also stopped beforeprotocol write; subsequent sourcehash check refused beforeclones. Final relock script explicit imports, no science change.'
    for file in HERE.glob('*.py'):
        ast.parse(file.read_text());p['input_hashes'][str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
    assert p['payload_bytes']+p['metadata_bytes']<p['final_bytes']
    used=lambda folder:sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())
    assert used(ROOT/'stage/runs')-used(OUT)+p['final_bytes']<p['global_bytes']
    for file in [HERE/'protocol.json',OUT/'protocol.json']:
        temp=file.with_suffix('.partial');temp.write_text(json.dumps(p,indent=2)+'\n');os.replace(temp,file)
    print(json.dumps(dict(state='ZERO_NAVIGATION_STORAGE_RELOCKED',own_bytes=p['final_bytes'],metadata_bytes=p['metadata_bytes'])),flush=True)


if __name__=='__main__':main()
