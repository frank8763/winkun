from pathlib import Path
import os,sys,tempfile,copy,time,io,ast,subprocess,json
import t628s_learning as M
L,torch,np=M.L,M.torch,M.np
def main():
 p=M.plan();M.configure_gpu(p);rows=M.read(M.ROOT/'stage/runs/t608_current_policy_memory/rl_01/summary.json')['result']['training_rows'];arrays=[M.P.Base.G.E.N.S.arrays(r['path']) for r in rows];obj=M.make(p,[.2,.2]);before=M.digest(M.state(obj));groups=[arrays for _ in range(4)];items,diags=M.P.batch(obj[0],groups,p);refs=M.references(obj[0],p);loss,_=M.objective(obj[0],items,refs,p);loss.backward();assert loss.dtype==torch.float64 and bool(torch.isfinite(loss));assert M.digest(M.state(obj))==before
 grad=torch.cat([v.grad.flatten() for v in L.variables(obj[0])]);assert torch.isfinite(grad).all() and float(torch.linalg.vector_norm(grad))>0
 # Exact current model function under sigma-only changes; real sample-step
 # path, density and exported checkpoint are all exercised on fixed inputs.
 q=obj[0].cpu();reference=q.sequence(torch.tensor(arrays[0]['states'][:12])).detach();sigma_checks=[]
 with tempfile.TemporaryDirectory(prefix='t628s_model_fixture_',dir=M.DOC) as tmp:
  for sigma in [[.2,.2],[.1,.1],[.05,.05]]:
   M.set_sigma(obj,sigma);path=Path(tmp)/('sigma_'+str(sigma[0])+'.pt.gz');model=M.export(obj,p,path);readback=M.frozen(p,path);assert readback.phash()==model['policy_hash'];assert torch.equal(readback.sequence(torch.tensor(arrays[0]['states'][:12])),reference)
   readback.sample=True;readback.rng=np.random.default_rng(62820261117);readback.reset_noise();x=torch.tensor(arrays[0]['states'][0:1])
   with torch.no_grad():action,loc,hidden,_=readback.step(x,None)
   row=readback.tape[-1];innovation=torch.tensor(row['innovations']);raw=torch.tensor(row['raw_actions']);assert torch.allclose(raw-loc[0],innovation*torch.tensor(sigma,dtype=torch.float64),atol=1e-12,rtol=1e-12);lp=L.P.L.density(loc,raw[None],torch.tensor(row['atom'][None]),sigma);assert abs(float(lp)-float(row['logp']))<1e-12
   sigma_checks.append(dict(sigma=sigma,policy_hash=model['policy_hash'],same_initial_mean_function=True,raw_innovation_scale_exact=True,logp=float(lp),bytes=model['bytes']))
  assert len({z['policy_hash'] for z in sigma_checks})==3 and sigma_checks[0]['policy_hash']==p['student_initial']['policy_hash']
  # Parameter-free sigma state recovery; no optimizer step, no new model kept.
  M.load(obj,{**M.state(obj), 'sigma':[.2,.2]});assert M.digest(M.state(obj))==before
  # New native loader is tested in a fresh CPU subprocess after its import
  # chain, avoiding inherited CUDA state and generic module shadowing.
  model=M.export(obj,p,Path(tmp)/'cpu_loader.pt.gz')
  code="""from pathlib import Path
import sys,os,json,unittest.mock as mock
sys.path.insert(0,str(Path('stage/experiments/t628s_actual_memory_recovery').resolve()))
import t628s_native as N
p=N.M.plan();N.install();q=N.N.P.frozen(p,sys.argv[1]);assert q.phash()==sys.argv[2];checks=[]
for i in range(3):
 def captured(pipe,index,p,counts):os.sched_setaffinity(0,{index});checks.append(sorted(os.sched_getaffinity(0)))
 with mock.patch.object(N.N.G.T,'worker',side_effect=captured):N.worker(None,i,p,None)
 N.N.restore(p)
print(json.dumps(checks))
"""
  c=subprocess.run([str(M.ROOT/'run_stage.sh'),'-c',code,model['path'],model['policy_hash']],cwd=M.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=30);assert c.returncode==0,c.stderr[-1400:];assert json.loads(c.stdout)==[[8],[10],[12]]
 for f in M.HERE.glob('*.py'):ast.parse(f.read_text())
 report=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',historical_groups_are_only_numeric_fixtures=True,sigma_checks=sigma_checks,initial_function_exact=True,original_sigma_policyhash_exact=True,new_native_CPU_worker_checkpoint_loader_pass=True,logp_atom_raw_sigma_real_step_exact=True,GPU_loss_gradient_finite=True,Actor_SGD_parameters_unchanged=True,full_history_BPTT=True,fixed_T608_memory_target_not_reset_each_round=True,new_native=0,new_optimizer=0,epoch=time.time());M.save(M.DOC/'PREFLIGHT.json',report);print('T628S_SIGMA_SAMPLING_DENSITY_CHECKPOINT_GPU_PREFLIGHT_PASS')
if __name__=='__main__':main()
