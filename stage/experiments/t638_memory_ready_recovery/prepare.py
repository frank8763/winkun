"""Bounded real-byte RAM broker checks; no native session or optimiser."""
from pathlib import Path
import ast,io,json,os,sys,time
import t638_memory as Memory
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;DOC=ROOT/'stage/docs/day_20261118/t638'
OUT=ROOT/'stage/runs/t638_memory_ready_recovery'
sys.path.insert(0,str(ROOT/'stage/experiments/t637_joint_coverage_recovery'))
import t637_data as D


def main():
    assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True)
    p=D.read(ROOT/'stage/experiments/t637_joint_coverage_recovery/protocol.json')
    for k in ['authorization_approved','user_authorization','registration_epoch','absolute_native_stop','absolute_learning_stop','absolute_all_stop']:
        p.pop(k,None)
    p['inputs']={f:h for f,h in p['inputs'].items() if '/day_20261118/t637/' not in f}
    p.update(id='T638_RAM_READY_RECOVERY',out=str(OUT),preload_seconds=900,raw_RAM_cap_bytes=128*2**20,
         authorization_approved=False,all_seconds=7200,max_commands=431,
         preceding_T637_bookkeeping_commands=6,combined_bookkeeping_max=437)
    rows=D.selected();sources=[]
    for row in rows:
        path=Path(row['path']);source=dict(id=row['job_id'],path=str(path.relative_to(ROOT)),sha256=row['sha256'])
        if path.exists():source['bytes']=path.stat().st_size
        else:
            marker=D.read(str(path)+'.t629-remote.json')
            source.update(bytes=marker['bytes'],receipt=marker['receipt'],manifest=marker['manifest'])
            p['inputs'][marker['receipt']]=D.sha(marker['receipt']);p['inputs'][marker['manifest']]=D.sha(marker['manifest'])
        sources.append(source)
    assert len(sources)==384 and sum(r['bytes'] for r in sources)<p['raw_RAM_cap_bytes']
    p['memory_sources']=sources
    # Keep the original six reserved command entries and failed producer intact.
    # This proposal requests a separately approved new bounded window; its
    # scientific command/task sequence is exactly the original431 sequence.
    local=next(r for r in sources if (ROOT/r['path']).exists())
    remote=next(r for r in reversed(sources) if 'receipt' in r)
    start=time.time();archive=Memory.fetch(remote['receipt'],180)
    import tarfile
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as tar:
        remote_blob=Memory.member(tar.extractfile(remote['path']).read(),remote)
    local_blob=Memory.member((ROOT/local['path']).read_bytes(),local)
    raw={local['path']:local_blob,remote['path']:remote_blob};sample=[local,remote]
    negative=False
    try:Memory.member(remote_blob[:-1]+bytes([remote_blob[-1]^1]),remote)
    except AssertionError:negative=True
    assert negative
    path=ROOT/'stage/runs/.t638-ram-preflight.sock';assert not path.exists()
    server=Memory.Server(path,raw,sample,time.time()+180)
    q=D.L.frozen(p,p['initial']['path']);initial=D.digest(q.state_dict());byid={r['job_id']:r for r in rows};features=[]
    try:
        for phase in ['main','audit']:
            stage=[]
            for source in sample:
                blob=Memory.get(path,source,phase);a=D.E.N.S.arrays(io.BytesIO(blob));g=D.features(q,a,byid[source['id']])
                stage.append(dict(id=source['id'],sha256=Memory.digest(blob),features_sha256=D.digest(g['f']),history_sha256=g['history_sha256']))
            features.append(stage)
        assert features[0]==features[1]
        for phase in ['main','audit']:
            try:Memory.get(path,remote,phase);raise RuntimeError('Duplicate read unexpectedly admitted')
            except AssertionError:pass
        assert len(server.calls)==4 and server.error is None
    finally:server.close()
    assert not path.exists() and D.digest(q.state_dict())==initial
    # Science code is unchanged; only the data-source function is replaced.
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    p['inputs'][str(ROOT/'stage/docs/day_20261118/t637/PREFLIGHT.json')]=D.sha(ROOT/'stage/docs/day_20261118/t637/PREFLIGHT.json')
    p['inputs'][str(DOC/'PLAN.md')]=D.sha(DOC/'PLAN.md')
    p['inputs'].update({str(f):D.sha(f) for f in HERE.glob('*.py')})
    worst=D.B.used(ROOT/'stage/runs')+p['max_bytes']+131072
    assert worst<2**30
    D.save(HERE/'protocol.json',p)
    D.save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',actual_RAM_roundtrips=4,
        independent_feature_reconstruction_exact=True,sample=features[0],corrupt_bytes_rejected=negative,
        duplicate_read_rejected=True,socket_removed=True,model_unchanged=True,remote_one_archive_every_member_SHA_verified=True,
        raw_RAM_all384_bound=sum(r['bytes'] for r in sources),raw_RAM_cap=p['raw_RAM_cap_bytes'],
        full384_preload_still_required_before_science=True,temporary_disk_cache_bytes=0,
        inherited_scientific_GPU_preflight=str(ROOT/'stage/docs/day_20261118/t637/PREFLIGHT.json'),
        new_GPU_optimizers=0,global_storage_worst=worst,seconds=time.time()-start))
    D.save(DOC/'BUDGET.json',{k:p[k] for k in ['max_physics','max_commands','max_bytes','all_seconds','preload_seconds','main_seconds','audit_seconds','raw_RAM_cap_bytes','new_Actor_Adam','shadow_Actor_Adam','independent_historical_Actor_Adam','preceding_T637_bookkeeping_commands','combined_bookkeeping_max']})
    print('T638_PREPARED_NO_SCIENCE',sum(r['bytes'] for r in sources),worst,flush=True)


if __name__=='__main__':main()
