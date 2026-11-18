"""Network-only subprocess; emit a verified archive to a RAM pipe."""
from pathlib import Path
import io,json,hashlib,sys,tarfile
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t628_fresh_noise_curriculum'))
import t628_stream as S


def main():
    r=json.loads(Path(sys.argv[1]).read_text());T=S.client();parts=[]
    assert r['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    for p in r['parts']:
        asset=T.B.api('GET',f"releases/assets/{p['asset_id']}")
        b=T.readback(asset,p['bytes']);assert len(b)==p['bytes'] and hashlib.sha256(b).hexdigest()==p['sha256'];parts.append(b)
    b=b''.join(parts);assert hashlib.sha256(b).hexdigest()==r['sha256']
    # Original T501 archives use external, separately hashed manifests, rather
    # than the newer batch archive's embedded manifest.json.
    manifest=Path(sys.argv[1]).with_name(Path(sys.argv[1]).name.replace('release_receipt_','source_manifest_'))
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==r['manifest_sha256']
    rows=json.loads(manifest.read_text())['files'];T.B.verify_tar(b,rows)
    sys.stdout.buffer.write(b)


if __name__=='__main__':main()
