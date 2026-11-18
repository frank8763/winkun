"""Full successful historical coverage; remote archives stay in RAM."""
from pathlib import Path
import hashlib,io,json,lzma,subprocess,sys,tarfile
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t636_joint_best_reproduction'))
import t636_runtime as B
L,P,E,torch,np=B.L,B.P,B.E,B.torch,B.np
read,save,sha,digest=B.read,B.save,B.sha,B.digest


def original_rows():
    f=ROOT/'stage/runs/t501_joint_command_repair/joint/cases.jsonl.t595.xz'
    old=[json.loads(z) for z in lzma.decompress(f.read_bytes()).decode().splitlines()]
    f=next((ROOT/'stage/runs/t502_fullfield_probe/probe').glob('cases.jsonl*'))
    data=f.read_bytes();data=lzma.decompress(data) if f.suffix=='.xz' else data
    import gzip
    if f.suffix=='.gz':data=gzip.decompress(data)
    field=[json.loads(z) for z in data.decode().splitlines()]
    assert len(old)==383 and len(field)==24
    return old,field


def selected():
    old,field=original_rows();rows=[r for r in old+field if r['success'] and not r['collision']]
    assert len(rows)==384 and len({r['job_id'] for r in rows})==384
    return rows


def groups(rows):
    result={}
    for r in rows:
        f=Path(r['path'])
        if f.exists():key='LOCAL'
        else:
            marker=read(str(f)+'.t629-remote.json')
            assert marker['sha256']==r['sha256']
            key=marker['receipt']
            receipt=read(key)
            assert receipt['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
            assert sha(marker['manifest'])==receipt['manifest_sha256']
        result.setdefault(key,[]).append(r)
    return result


def remote_bytes(receipt,timeout=180):
    c=subprocess.run([str(ROOT/'run_stage.sh'),str(Path(__file__).with_name('t637_fetch.py')),receipt],
         cwd=ROOT,capture_output=True,timeout=timeout)
    assert c.returncode==0,c.stderr.decode(errors='replace')[-1200:]
    data=c.stdout;r=read(receipt)
    assert hashlib.sha256(data).hexdigest()==r['sha256']
    return data


def arrays(rows,guard,only_first_archive=False):
    for key,rr in groups(rows).items():
        guard()
        if key=='LOCAL':
            for r in rr:
                assert sha(r['path'])==r['sha256'];yield E.N.S.arrays(r['path']),r
        else:
            data=remote_bytes(key)
            with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
                for r in rr:
                    blob=tar.extractfile(str(Path(r['path']).relative_to(ROOT))).read()
                    assert hashlib.sha256(blob).hexdigest()==r['sha256']
                    yield E.N.S.arrays(io.BytesIO(blob)),r
            del data
            if only_first_archive:return


def features(q,a,row):
    assert all(digest(v)==row['array_hashes'][k] for k,v in a.items())
    assert digest(a)==row['array_digest']
    assert not a['ownership'].any() and not a['stochastic'].any()
    assert a['end_flags'][-1].tolist()==[0,1,0]
    B.full_audit(q.p,row,a)
    x=torch.tensor(a['states'])
    with torch.no_grad():
        n=q.normalized(x);h,_=q.core(q.encoder(n)[:,None]);h=h[:,0]
        target=torch.tensor(a['locations']);loc=q.location(h,n)
        assert float((loc-target).abs().max())<1e-10
        return dict(id=row['job_id'],f=torch.cat([h,n],-1),logits=q.head(h),target=target,
                    source_sha256=row['sha256'],history_sha256=digest(a),physics=row['physics'])


def load(p,guard):
    q=L.frozen(p,p['initial']['path']);data=[]
    rows=selected()
    assert {r['job_id'] for r in rows}==set(p['before_success_ids'])
    for a,r in arrays(rows,guard):
        guard();data.append(features(q,a,r))
    assert len(data)==384 and {g['id'] for g in data}==set(p['before_success_ids'])
    return data
