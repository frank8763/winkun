"""Reproduce the real duplicate-path overwrite, verify unique-runtime remedy."""
from pathlib import Path
import gzip
import hashlib
import json
import sys
import tempfile
import time
from unittest import mock
ROOT=Path(__file__).resolve().parents[3]
DOC=ROOT/'stage/docs/day_20261118/t634'
sys.path.insert(0,str(ROOT/'stage/experiments/t631_explored_recovery'))
import t631_runtime as R
from unique_runtime import Runtime


def main():
    old=ROOT/'stage/runs/t633_conditional_recovery_pg'
    with gzip.open(old/'cases.jsonl.gz','rt') as f:rows=[json.loads(z) for z in f if z.strip()]
    bad=[r for r in rows if hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()!=r['sha256']]
    assert len(bad)==1
    duplicate=next(r for r in rows if r['path']==bad[0]['path'] and r['sha256']!=bad[0]['sha256'])
    assert duplicate['job_id']==bad[0]['job_id']
    # Use actual full-array returns of this same job from two real trajectories.
    source=R.read(ROOT/'stage/runs/t631_explored_recovery/summary.json')['result']['exploration'][:2]
    aa=[R.R.B.R.E.N.S.arrays(r['path']) for r in source]
    p=R.read(ROOT/'stage/runs/t633_conditional_recovery_pg/protocol.json')
    cfg=R.read(p['actor_configuration']);cfg.update(p);p=cfg
    R.setup(p);B=R.R.B
    findings={}
    DOC.mkdir(parents=True,exist_ok=True)
    for name,cls in [('original',B.Runtime),('fixed',Runtime)]:
        with tempfile.TemporaryDirectory(prefix=name+'_',dir=DOC) as tmp:
            folder=Path(tmp)
            with mock.patch.object(B,'OUT',folder),mock.patch.object(B,'verify',lambda *args:None),mock.patch.object(B.N.N,'inspect',lambda *args:None):
                rt=cls(dict(p,absolute_native_stop=time.time()+300));index=0
                class Pool:
                    def request(self,requests):
                        nonlocal index
                        assert len(requests)==1
                        result=(aa[index],dict(source[index]));index+=1;return [result]
                rt.pool=Pool();model=p['reference_model'];job=source[0]['request']['job']
                first=rt.request([job],model,'FINAL49')[0][1]
                second=rt.request([job],model,'FINAL49')[0][1]
                findings[name]=dict(same_path=first['path']==second['path'],
                    first_bytes_retained=B.sha(first['path'])==first['sha256'],
                    second_bytes_retained=B.sha(second['path'])==second['sha256'])
    assert findings['original']==dict(same_path=True,first_bytes_retained=False,second_bytes_retained=True)
    assert findings['fixed']==dict(same_path=False,first_bytes_retained=True,second_bytes_retained=True)
    report=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',real_T633_raw_file_overwrite_reproduced=True,
                permanently_missing_original_compact_record=bad[0],current_record=duplicate,
                original_failure_preserved=True,regression=findings,
                fixture_inspector_disabled_for_storage_seam_only=True,native_steps=0,optimizer_updates=0)
    (DOC/'STORAGE_REGRESSION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(state=report['state'],regression=findings,overwritten_original_files=1)))


if __name__=='__main__':main()
