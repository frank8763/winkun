"""Durable verified native inputs, staged before an offline night."""
from pathlib import Path
import hashlib,io,json,os,sys,tarfile,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t646_offline_night';DOC=ROOT/'stage/docs/overnight_20261118'
sys.path.insert(0,str(ROOT/'stage/experiments/t645r_transport_recovery'))
import t645r_memory as Transport


def read(f):return json.loads(Path(f).read_text())
def save(f,v):Path(f).write_text(json.dumps(v,indent=2)+'\n')
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def stage():
    """Authorized preparation only: no native session, no optimizer, 10GiB cap."""
    folder=OUT/'offline_inputs';folder.mkdir(parents=True,exist_ok=True)
    source=read(ROOT/'stage/experiments/t645_iterative_recovery_relay/protocol.json')['memory_sources']
    groups={}
    for row in source:groups.setdefault(row.get('receipt','LOCAL'),[]).append(row)
    started=time.time();rows=[]
    for key,items in groups.items():
        assert time.time()-started<1200
        data=None;tar=None
        missing=[r for r in items if not (folder/(r['sha256']+'.npz')).exists()]
        if missing and key!='LOCAL':
            data=Transport.fetch(key,180);tar=tarfile.open(fileobj=io.BytesIO(data),mode='r:gz')
        for r in items:
            path=folder/(r['sha256']+'.npz')
            if not path.exists():
                blob=(ROOT/r['path']).read_bytes() if key=='LOCAL' else tar.extractfile(r['path']).read()
                Transport.member(blob,r)
                with path.open('xb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
            assert path.stat().st_size==r['bytes'] and sha(path)==r['sha256']
            rows.append(dict(r,offline_path=str(path)))
        if tar:tar.close()
        del data
        save(OUT/'INPUT_PREPARATION.json',dict(state='PREPARATION_ONLY_NO_SCIENCE',members=len(rows),bytes=sum(r['bytes'] for r in rows),epoch=time.time(),new_native=0,new_optimizer=0))
    assert len(rows)==384 and sum(r['bytes'] for r in rows)<128*2**20
    assert sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file())<10*2**30
    save(OUT/'OFFLINE_INPUTS.json',dict(state='ALL384_ORIGINAL_BYTES_READY_AND_SHA_VERIFIED',members=rows,bytes=sum(r['bytes'] for r in rows),seconds=time.time()-started,source_originals_not_modified=True))
    print('ALL384_OFFLINE_INPUTS_VERIFIED',len(rows),sum(r['bytes'] for r in rows),flush=True)


if __name__=='__main__':stage()
