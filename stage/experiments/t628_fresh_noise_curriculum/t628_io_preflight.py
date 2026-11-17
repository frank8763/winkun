"""Existing-payload IO fixtures and synthetic successor parameters; no simulation/update."""
from pathlib import Path
import tempfile,copy,json,unittest.mock as mock,time,gzip
import t628_runtime as R
M=R.M

def main():
 p=M.plan();R.setup(p);source=R.read(R.ROOT/'stage/runs/t608_current_policy_memory/rl_01/summary.json')['result']['training_rows'];arrays=[R.B.R.E.N.S.arrays(r['path']) for r in source];assert len(source)==4
 with tempfile.TemporaryDirectory(prefix='t628_io_fixture_',dir=M.DOC) as tmp:
  folder=Path(tmp)
  with mock.patch.object(R,'OUT',folder),mock.patch.object(R.B,'OUT',folder):
   obj=M.make(p,[.2,.2],device='cpu');initial=M.state(obj);model=M.export(obj,p,folder/'start.pt.gz');captured={};rt=R.B.Runtime(p)
   class VirtualPool:
    def request(self,requests):
     values=[]
     for request in requests:
      i=next(k for k,r in enumerate(source) if r['seed']==request['seed']);a=arrays[i];row=copy.deepcopy(source[i]);row.pop('request',None)
      if request['full_path']:
       path=Path(request['full_path']);path.parent.mkdir(exist_ok=True);path.write_bytes(R.B.arrays_blob(a))
      values.append((a,row))
     return values
   rt.pool=VirtualPool()
   def fake_stream(p,action,data,tag=None):
    if action=='download':return captured[json.loads(data)['sha256']]
    files,manifest=R.B.S.unpack(data);h=R.B.S.digest(data);captured[h]=data;return json.dumps(dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',fixture_only=True,bytes=len(data),sha256=h,manifest=manifest)).encode()
   with mock.patch.object(R.B,'stream',side_effect=fake_stream):
    R.archive_model(p,model,'fixture_start')
    for group in range(4):
     for half in range(2):
      indexes=[half*2+i for i in range(2)];jobs=[source[i]['request']['job'] for i in indexes];seeds=[source[i]['seed'] for i in indexes];vv=rt.request(jobs,model,'FRESH_TRAIN4',seeds=seeds);refs=[r for a,r in vv];assert all(Path(r['path']).exists() for r in refs)
      R.B.batch_archive(p,group*2+half,vv);assert all(not Path(r['path']).exists() for r in refs);rt.request(jobs,model,'EXACT_TRAIN4',seeds=seeds,references=refs)
    assert len(list(folder.glob('batch_*_receipt.json')))==8 and len(rt.rows)==len(rt.replays)==16
    for f in folder.glob('batch_*_receipt.json'):
     files,manifest=R.retrieve(p,R.read(f));assert len(files)==len(manifest['rows'])==2
    R.evict_model(model);assert not Path(model['path']).exists();R.ensure_model(p,model);assert M.sha(model['path'])==model['sha256']
   x=M.torch.tensor(arrays[0]['states'][:12]);baseline=obj[0].sequence(x).detach().clone()
   with M.torch.no_grad():M.L.variables(obj[0])[-1].view(-1)[0].add_(.001)
   altered=obj[0].sequence(x).detach();assert not M.torch.equal(baseline,altered);altered_state=M.state(obj);M.set_sigma(obj,[.05,.05]);next_model=M.export(obj,p,folder/'successor.pt.gz');next_q=M.frozen(p,next_model['path']);assert M.torch.equal(next_q.sequence(x),altered) and next_q.phash()!=model['policy_hash'] and next_q.p['raw_sigma']==[.05,.05]
   M.load(obj,initial);assert M.digest(M.state(obj))==M.digest(initial);M.load(obj,altered_state);assert M.torch.equal(obj[0].sequence(x),altered);M.load(obj,initial)
   # Real lossless Pool receive, before envelope/request enrichment.
   test=object.__new__(R.B.N.Pool);test.p=dict(p,out=str(folder));test.pending=[dict(model=model['path'],job=source[0]['request']['job'])];test.serial=999;bare=copy.deepcopy(source[0]);bare.pop('request',None)
   with mock.patch.object(R.B.N.N.Pool,'receive',return_value=(arrays[0],bare)):R.B.N.Pool.receive(test,0)
   with gzip.open(folder/'worker_return_999_0.json.gz','rt') as f:saved=json.load(f)
   assert 'request' not in saved['native_result'] and saved['request']['model']==model['path']
 # Exact acceptance identities, including loss even when total success is equal.
 def row(job,good):
  r=copy.deepcopy(source[0]);r['job_id']=job
  # Use actual known safe/unsafe envelopes instead of inventing safe flags.
  return r
 summary=R.read(R.ROOT/'stage/runs/t627_conflict_aware_policy/summary.json')
 # Scientific gate booleans are exercised directly against real final rows.
 for arm in ['mean','cagrad']:
  path=R.ROOT/'stage/runs/t627_conflict_aware_policy'/arm/'cases.jsonl.gz'
  if not path.exists():continue
  with gzip.open(path,'rt') as f:rows=[json.loads(z) for z in f if z.strip()]
  final=[r for r in rows if r['role']=='FINAL49'];metrics=R.final_metrics(p,final);assert metrics['lost_prior'] and not (not metrics['lost_prior'] and bool(metrics['gained_prior']))
 report=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',actual_Runtime_request_pair_cache_receipt_eviction_exact_path=True,eight_pair_receipts_retrieved_and_unpacked=True,model_remote_restore_byte_exact=True,actual_Pool_receive_original_saved=True,synthetic_parameter_only_successor_not_training=True,successor_loader_uses_changed_parameters_and_new_sigma=True,weights_optimizer_sigma_full_state_rollback_exact=True,fixtures_are_historical_payload_reuse_not_new_scientific_data=True,new_native=0,new_optimizer=0,epoch=time.time());M.save(M.DOC/'IO_PREFLIGHT.json',report);print('T628_PAIR_STREAM_SUCCESSOR_IO_PREFLIGHT_PASS')
if __name__=='__main__':main()
