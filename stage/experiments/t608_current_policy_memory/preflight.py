"""Real-Actor trace schema regression and memory bounds, zero native/optimizer."""
from pathlib import Path
import sys,copy,types,ast,math,unittest.mock as mock,time
import t608_native as N
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;DOC=ROOT/'stage/docs/day_20261117/t608'
L=N.L;torch,np=L.torch,L.np

def main():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 for f in HERE.glob('*.py'):ast.parse(f.read_text())
 p=L.read(ROOT/'stage/runs/t606r_offline_retention_recovery/protocol.json');cfg=L.read(p['actor_configuration']);cfg.update(p)
 best=L.read(ROOT/'stage/runs/t606r_offline_retention_recovery/rl_00/summary.json')['result']['model'];rows=[]
 source=L.G.E.N.S.arrays(ROOT/'stage/runs/t606r_offline_retention_recovery/rl_00/episodes/full_0.npz');x=source['states'][:64];original_failure_reproduced=False
 for name,model in [('T550',p['base_model']),('T608_initial33',best)]:
  q=L.frozen(cfg,model['path']);q.sample=False;q.reset_noise();before=q.phash();h=None
  with torch.no_grad():
   for xx in x:_,loc,h,_=q.step(torch.tensor(xx[None]),h)
  s=types.SimpleNamespace(q=q);base={'states':x.copy()}
  with mock.patch.object(N.Legacy.R,'payload',side_effect=lambda *args,**kw:copy.deepcopy(base)):
   if name=='T550':
    try:N.Legacy.payload(s)
    except KeyError as e:assert e.args==('extra_hidden_before',);original_failure_reproduced=True
    else:raise AssertionError('Expected original T606R schema failure')
   a=N.payload(s)
  present='extra_hidden_before' in a;assert present==(name!='T550') and q.phash()==before
  if present:
   assert np.array_equal(a['extra_hidden_before'],np.asarray([z['extra_hidden_before'] for z in q.tape]));assert a['extra_hidden_before'].shape==(len(x),32)
   assert np.array_equal(a['last_extra_hidden_after'],q.extra_hidden[0,0].numpy())
  rows.append(dict(model=name,policy_hash=before,trace_decisions=len(x),extra_hidden_present=present))
 assert original_failure_reproduced
 d=math.ceil(max(j['case']['step_cap'] for j in cfg['evaluation_jobs'])/100)+1
 shape=dict(states=np.zeros((d,120),np.float64),locations=np.zeros((d,2),np.float64),actions=np.zeros((d,2),np.float32),ownership=np.zeros(d,np.uint8),native_counts=np.zeros(d,np.int64))
 import io
 b=io.BytesIO();np.savez(b,**shape);raw=len(b.getvalue());upper=raw+raw//1000+32768;assert upper<2**20
 j=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',original_failure_reproduced=True,architecture_schema_regression=rows,real_model_forward_only_no_native=True,whole_model_weights_unchanged=True,memory_rows_per_episode_upper=d,memory_uncompressed=raw,memory_deflate_safe_upper=upper,memory_file_cap=2**20,old_model_missing_fields_not_filled_with_fake_zeros=True,new_native=0,new_Adam=0,scope='Actual Actor.step tapes and real payload append seam, delegated base payload fixture. Actual spawned dual-model native path must still pass before learning.')
 N.P.frozen=L.frozen
 old=next(r for r in N.G.rows(ROOT/'stage/runs/t596_recurrent_joint_online') if r.get('full') and r['request']['model']==p['base_model']['path'])
 fresh=L.read(ROOT/'stage/runs/t606r_offline_retention_recovery/rl_00/summary.json')['result']['training_rows'][0];histories=[]
 for label,row in [('T550_full_native_stochastic_history',old),('T598_full_native_stochastic_history',fresh)]:
  assert L.sha(row['path'])==row['sha256'];arrays=N.E.N.S.arrays(row['path']);N.inspect(cfg,[(arrays,row)]);histories.append(dict(label=label,path=row['path'],sha256=row['sha256'],full_history_inspection=True))
 j['actual_saved_native_histories_both_architectures']=histories
 L.save(DOC/'PREFLIGHT.json',j);print('T608_DUAL_ACTOR_SCHEMA_AND_REAL_NATIVE_HISTORY_REGRESSION_PASS',upper,flush=True)
if __name__=='__main__':main()
