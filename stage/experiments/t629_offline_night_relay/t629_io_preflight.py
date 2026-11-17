"""Real Runtime/cache/exact/metadata path, using only preexisting payloads."""
from pathlib import Path
import tempfile,copy,json,unittest.mock as mock,time,gzip,ast
import t629_runtime as R
M=R.M

def main():
 p=M.plan();R.setup(p,M.DOC);source=M.read(M.ROOT/'stage/runs/t608_current_policy_memory/rl_01/summary.json')['result']['training_rows'];arrays=[R.B.R.E.N.S.arrays(r['path']) for r in source]
 with tempfile.TemporaryDirectory(prefix='t629_io_fixture_',dir=M.DOC) as tmp:
  folder=Path(tmp)
  with mock.patch.object(R.B,'OUT',folder):
   rt=R.B.Runtime(p)
   class VirtualPool:
    def request(self,requests):
     out=[]
     for req in requests:
      i=next(k for k,row in enumerate(source) if row['seed']==req['seed']);a=arrays[i];row=copy.deepcopy(source[i]);row.pop('request',None)
      if req['full_path']:
       path=Path(req['full_path']);path.parent.mkdir(exist_ok=True);path.write_bytes(R.B.arrays_blob(a))
      out.append((a,row))
     return out
   rt.pool=VirtualPool()
   for off in [0,2]:
    jobs=[r['request']['job'] for r in source[off:off+2]];seeds=[r['seed'] for r in source[off:off+2]];vv=rt.request(jobs,p['student_initial'],'FRESH_TRAIN4',seeds=seeds);rows=[r for a,r in vv];assert all(Path(r['path']).exists() for r in rows);rt.request(jobs,p['student_initial'],'EXACT_TRAIN2',seeds=seeds,references=rows);assert all(Path(r['path']).exists() and M.sha(r['path'])==r['sha256'] for r in rows)
   assert len(rt.rows)==len(rt.replays)==4 and len(list((folder/'batch_cache').glob('*.npz')))==4
   manifest=[dict(path=str(f),sha256=M.sha(f),bytes=f.stat().st_size) for f in folder.rglob('*') if f.is_file()];assert all(M.sha(row['path'])==row['sha256'] for row in manifest)
   test=object.__new__(R.B.N.Pool);test.p=dict(p,out=str(folder));test.pending=[dict(model=p['student_initial']['path'],job=source[0]['request']['job'])];test.serial=999;bare=copy.deepcopy(source[0]);bare.pop('request',None)
   with mock.patch.object(R.B.N.N.Pool,'receive',return_value=(arrays[0],bare)):R.B.N.Pool.receive(test,0)
   with gzip.open(folder/'worker_return_999_0.json.gz','rt') as f:saved=json.load(f)
   assert 'request' not in saved['native_result'] and saved['request']['model']==p['student_initial']['path']
 with gzip.open(M.ROOT/'stage/runs/t626_multitask_closed_loop/cases.jsonl.gz','rt') as f:cases=[json.loads(x) for x in f if x.strip()]
 final=[r for r in cases if r['role']=='FINAL49'];metrics=R.metrics(p,final);assert metrics['success']==34 and len(metrics['lost_prior'])==len(metrics['gained_prior'])==2 and not R.accepted(metrics)
 # Lightweight driver and cold storage cannot initiate science through imports.
 driver=ast.parse((M.HERE/'t629_driver.py').read_text());assert all(not isinstance(node,ast.ImportFrom) or 't629' not in (node.module or '') for node in driver.body)
 report=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',actual_Runtime_full_cache_and_same_seed_exact=True,all_raw_retained_locally_without_network=True,original_Pool_receive_before_envelope=True,local_manifest_byte_SHA=True,total34_same_but_lost2_rejected=True,no_remote_science_requirement=True,historical_payloads_only_not_new_data=True,new_native=0,new_optimizer=0,epoch=time.time());M.save(M.DOC/'IO_PREFLIGHT.json',report);print('T629_OFFLINE_REAL_IO_NO_EVICTION_PREFLIGHT_PASS')
if __name__=='__main__':main()
