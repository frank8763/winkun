"""Real request/cache/archive/eviction/exact path with historical payloads."""
from pathlib import Path
import tempfile,copy,json,unittest.mock as mock,time,gzip
import t627_runtime as X
R=X.B
def main():
 p=X.M.plan();X.setup(p);source=R.read(R.ROOT/'stage/runs/t608_current_policy_memory/rl_01/summary.json')['result']['training_rows'];arrays=[R.R.E.N.S.arrays(r['path']) for r in source];jobs=[r['request']['job'] for r in source];seeds=[r['seed'] for r in source];checks=[]
 with tempfile.TemporaryDirectory(prefix='t627_fixture_',dir=R.DOC) as tmp:
  folder=Path(tmp)
  with mock.patch.object(R,'OUT',folder):
   rt=R.Runtime(p)
   class VirtualPool:
    index=0
    def request(self,requests):
     result=[]
     for request in requests:
      i=self.index%4;self.index+=1;a=arrays[i];row=copy.deepcopy(source[i]);row.pop('request',None)
      if request['full_path']:
       f=Path(request['full_path']);f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(R.arrays_blob(a))
      result.append((a,row))
     return result
   rt.pool=VirtualPool();vv=rt.request(jobs,p['student_initial'],'FRESH_TRAIN4',seeds=seeds);refs=[r for a,r in vv];assert len(refs)==4 and all(Path(r['path']).exists() for r in refs)
   captured=[]
   def fake_upload(p,action,data,tag):
    files,manifest=R.S.unpack(data);assert len(files)==4;captured.append(data);return json.dumps(dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',fixture_only=True,bytes=len(data),sha256=R.S.digest(data),manifest=manifest)).encode()
   with mock.patch.object(R,'stream',side_effect=fake_upload):archived=R.batch_archive(p,0,vv)
   assert len(archived)==4 and all(not Path(r['path']).exists() for r in refs);assert len(R.read_rows('cache_evictions.jsonl'))==4
   rt.request(jobs,p['student_initial'],'EXACT_TRAIN4',seeds=seeds,references=refs);assert len(rt.replays)==4 and all(r['exact'] for r in rt.replays);assert len(R.read_rows('cases.jsonl'))==len(R.read_rows('replays.jsonl'))==4
   # Exercise actual Pool.receive before the Runtime envelope has request.
   test=object.__new__(R.N.Pool);test.p=dict(p,out=str(folder));test.pending=[dict(model=p['student_initial']['path'],job=jobs[0])];test.serial=999;bare=copy.deepcopy(source[0]);bare.pop('request',None)
   with mock.patch.object(R.N.N.Pool,'receive',return_value=(arrays[0],bare)):R.N.Pool.receive(test,0)
   with gzip.open(folder/'worker_return_999_0.json.gz','rt') as f:saved=json.load(f)
   assert 'request' not in saved['native_result'] and saved['request']['model']==p['student_initial']['path']
   checks=['full_original_saved_before_envelope','real_full_history_native_inspect','four_member_RAM_archive_exact','local_cache_evict_only_after_receipt_SHA','same_seed_four_exact_native_payload_hashes','lossless_gzip_metadata_rows','actual_Pool_receive_without_request','raw_request_result_saved_before_derivation']
 R.save(R.DOC/'IO_PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',checks=checks,only_existing_historical_payloads_in_virtual_pool=True,virtual_physics_not_new_native=True,actual_GitHub_transport_verified_separately='STREAM_PREFLIGHT.json',new_native=0,new_optimizer=0,epoch=time.time()));print('T627_REAL_IO_CHAIN_PREFLIGHT_PASS',len(checks))
if __name__=='__main__':main()
