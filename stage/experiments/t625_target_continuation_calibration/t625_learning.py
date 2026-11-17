from pathlib import Path
import os,sys,io,gzip,time
REQUESTED_CUDA=os.environ.get('CUDA_VISIBLE_DEVICES','')
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t622_gpu_critic_calibration'))
import t622_learning as B
os.environ['CUDA_VISIBLE_DEVICES']=REQUESTED_CUDA
L,torch,np=B.L,B.torch,B.np
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t625_target_continuation_calibration';DOC=ROOT/'stage/docs/day_20261117/t625'
def config():
 p=L.read(HERE/'protocol.json');c=L.read(p['actor_configuration']);c.update(p);return c
def setup(p):
 B.setup();real=sys.modules['t493_runtime']._real_affinity;real(0,set(p['stage_cpus']));os.sched_setaffinity=real;assert sorted(os.sched_getaffinity(0))==p['stage_cpus']
def episodes(rows,p):
 result=[];k=p['branch_decision']
 for row in rows:
  e=L.episode(row,p);a=L.G.E.N.S.arrays(row['path']);n=e['n'];assert not a['stochastic'].any() and n>k+1 and row['policy_hash']==p['student_initial']['policy_hash']
  assert not a['ownership'][k+1:].any()
  assert np.array_equal(a['actions'][k+1:],a['locations'][k+1:].clip(-1,1).astype(np.float32))
  ret=e['r'].clone();acc=0.
  for i in range(n-1,k-1,-1):acc=e['r'][i]+e['discount'][i]*e['bootstrap'][i]*acc;ret[i]=acc
  result.append(dict(e,ret=ret,k=k,delta=row['request']['job']['steering_delta']))
 assert sorted(e['delta'] for e in result)==[-.1,0.,.1]
 # The full causal prefix is input only; forced-prefix returns are never labels.
 return result
def data(p,test=False):
 rows=L.read(OUT/'collect/ROWS.json') if test else L.read(p['training_reference_rows']);return episodes(rows,p)
def make(p):
 q,opt=B.make(p);s=torch.load(io.BytesIO(gzip.decompress(Path(p['critic_checkpoint']).read_bytes())),map_location='cpu',weights_only=False);q.load_state_dict(s['state']['critics']);assert not opt.state
 return q,opt
def pack(es,p,rng):
 x=torch.nn.utils.rnn.pad_sequence([e['x'] for e in es]).cuda();pairs=[(int(k),i) for i,e in enumerate(es) for k in rng.integers(e['k']+1,e['n'],size=16)];t=torch.tensor([k for k,i in pairs],device='cuda');b=torch.tensor([i for k,i in pairs],device='cuda');u=torch.stack([es[i]['u'][k] for k,i in pairs]).cuda();y=torch.stack([es[i]['ret'][k] for k,i in pairs]).cuda();ks=torch.tensor([e['k'] for e in es],device='cuda');ii=torch.arange(len(es),device='cuda');bu=torch.stack([e['u'][e['k']] for e in es]).cuda();by=torch.stack([e['ret'][e['k']] for e in es]).cuda();return x,t,b,u,y,ks,ii,bu,by
def loss(obj,pack,p,arm):
 x,t,b,u,y,k,i,bu,by=pack;parameter=next(obj[0].parameters());reg=parameter.new_zeros(());ranking=parameter.new_zeros(());pairs=0
 for q in obj[0]:
  h=q.features(x);v=q.values(h[t,b],u);bv=q.values(h[k,i],bu);reg+=.5*torch.nn.functional.mse_loss(v,y)+.5*torch.nn.functional.mse_loss(bv,by)
  for a in range(3):
   for b in range(a+1,3):
    gap=by[a]-by[b]
    if float(gap.abs())>=p['rank_target_min_gap']:ranking+=torch.relu(p['rank_margin']-gap.sign()*(bv[a]-bv[b]));pairs+=1
 if pairs:ranking/=pairs
 return reg+(p['ranking_weight']*ranking if arm=='rank' else 0),dict(regression=float(reg.detach()),ranking=float(ranking.detach()),ranking_pairs=pairs)
def fit(obj,es,p,arm,guard,event):
 rng=np.random.default_rng(p['learner_seed']);trace=[]
 for step in range(p['updates_per_arm']):
  guard();obj[1].zero_grad(set_to_none=True);value,parts=loss(obj,pack(es,p,rng),p,arm);value.backward();norm=float(torch.nn.utils.clip_grad_norm_(obj[0].parameters(),1.,error_if_nonfinite=True));event('BEFORE',step);obj[1].step();event('AFTER',step);trace.append(dict(step=step,loss=float(value.detach()),gradient_norm=norm,**parts))
 return trace
def evaluate(obj,es,p):
 rows=[]
 with torch.no_grad():
  for e in es:
   x=e['x'][:,None].cuda();u=e['u'].cuda();pred=torch.stack([q.values(q.features(x)[:-1,0],u) for q in obj[0]]).cpu();k=e['k'];rows.append(dict(delta=e['delta'],MSE=float((pred[:,k:].mean(0)-e['ret'][k:]).square().mean()),actual=float(e['ret'][k]),predicted=float(pred[:,k].mean()),individual=pred[:,k].tolist(),path=e['path']))
 return dict(rows=rows,MSE=sum(r['MSE'] for r in rows)/3,actual_ranking=[r['delta'] for r in sorted(rows,key=lambda r:-r['actual'])],predicted_ranking=[r['delta'] for r in sorted(rows,key=lambda r:-r['predicted'])])
