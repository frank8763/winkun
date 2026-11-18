"""Identical T637 science, supplied by immutable RAM rather than GitHub."""
from pathlib import Path
import io,sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stage/experiments/t637_joint_coverage_recovery'))
import t637_runtime as R
import t638_memory as Memory
OUT=ROOT/'stage/runs/t638_memory_ready_recovery';DOC=ROOT/'stage/docs/day_20261118/t638'
R.HERE=HERE;R.OUT=OUT;R.DOC=DOC


def load(p,guard):
    phase=sys.argv[1];assert phase in ['main','audit']
    ready=R.read(OUT/'DATA_READY.json');assert ready['state']=='ALL384_RAW_BYTES_READY_AND_SHA_VERIFIED'
    q=R.L.frozen(p,p['initial']['path']);by={r['path']:r for r in p['memory_sources']};data=[]
    for row in R.D.selected():
        guard();source=by[str(Path(row['path']).relative_to(ROOT))]
        blob=Memory.get(OUT/'raw.sock',source,phase)
        a=R.E.N.S.arrays(io.BytesIO(blob));data.append(R.D.features(q,a,row))
    assert len(data)==384 and {g['id'] for g in data}==set(p['before_success_ids'])
    return data


if __name__=='__main__':
    R.D.load=load
    def no_download(*args,**kwargs):raise RuntimeError('Science must use preverified RAM; remote data fetching is disabled')
    R.D.remote_bytes=no_download
    # Disable external network even before historical data reconstruction.
    R.E.N.offline()
    {'main':R.main,'audit':R.audit}[sys.argv[1]]()
