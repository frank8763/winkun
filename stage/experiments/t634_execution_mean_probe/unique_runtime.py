"""Persist each occurrence separately; repeated job IDs cannot overwrite raw."""
from pathlib import Path
import copy
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t626_multitask_closed_loop'))
import t626_runtime as B


class Runtime(B.Runtime):
    def request(self,jobs,model,role,seeds=None,references=None):
        values=[]
        for off in range(0,len(jobs),3):
            self.guard();js=jobs[off:off+3]
            assert sum(self.counts)+sum(j['case']['step_cap'] for j in js)<=self.p['max_physics']
            assert self.commands+len(js)+3<=self.p['max_commands']
            self.commands+=len(js)
            req=[dict(model=model['path'],job=j,seed=None if seeds is None else seeds[off+i],query=False,
                 full_path=str(B.OUT/'batch_cache'/f'episode_{len(self.rows)+i:03d}.npz')
                 if role=='FRESH_TRAIN4' else None) for i,j in enumerate(js)]
            returned=self.pool.request(req)
            for i,((a,r),request,j) in enumerate(zip(returned,req,js)):
                r=B.R.Base.envelope(j,r,a,self.p['field_limit_m'])
                r.update(role=role,seed=request['seed'],request=request,
                         loaded_model_sha256=model['sha256'],model_sha256=model['sha256'],full=True)
                assert r['policy_hash']==model['policy_hash']
                if references is not None:
                    old=references[off+i]
                    assert r['array_hashes']==old['array_hashes']
                    assert r['integration_sha256']==old['integration_sha256'] and r['field_detail']==old['field_detail']
                    r.update(reference_path=old['path'],exact=True);self.replays.append(r)
                    B.append('replays.jsonl',r)
                else:
                    if role=='FINAL49':
                        path=B.OUT/'evaluation'/f"{len(self.rows):04d}_{r['job_id']}.npz"
                        path.parent.mkdir(exist_ok=True);blob=B.R.Base.pack(a)
                        assert len(blob)<=self.p['compact_bytes']
                        with path.open('xb') as f:f.write(blob)
                        r.update(path=str(path),sha256=B.sha(path),full=False,full_arrays_generated_but_not_saved=True)
                    else:
                        path=Path(request['full_path']);assert path.exists() and path.stat().st_size<=self.p['fresh_bytes']
                        r.update(path=str(path),sha256=B.sha(path))
                    self.rows.append(r);B.append('cases.jsonl',r)
                try:B.N.N.inspect(self.p,[(a,r)])
                except BaseException:
                    with (B.OUT/f'contract_failure_{len(self.rows)}_{len(self.replays)}.npz').open('xb') as f:
                        f.write(B.arrays_blob(a))
                    raise
                values.append((a,r))
        return values
