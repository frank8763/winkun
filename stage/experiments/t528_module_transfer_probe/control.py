"""One paired frozen-module diagnostic, zero optimizer updates."""
from pathlib import Path
import json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t528_runtime as R
G,P,torch,np,read,save,sha,digest=R.G,R.P,R.torch,R.np,R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/day_20261113'
def prepare():
 assert not OUT.exists();prior=ROOT/'stage/runs/t527_validated_recovery_learning';assert read(prior/'pilot/audit.json')['state']=='PASS'
 assert all(not Path('/proc',str(i)).exists() for i in read(prior/'SEALED.json')['scientific_pids']+[read(prior/'launch.json')['pid']])
 assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T527.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
 p=read(prior/'protocol.json');s=read(prior/'pilot/summary.json')['result'];assert not s['panel_pass'] and s['target']['success']==1
 now=time.time();assert now+540<p['absolute_all_stop'];target=p['probe_job'];old=next(j for j in p['old8_jobs'] if j['id']=='dev_task6_13');physics=4*(target['case']['step_cap']+old['case']['step_cap'])
 p.update(id='T528_PAIRED_FROZEN_MODULE_TRANSFERS',out=str(OUT),learned=s['model'],probe_jobs=[old,target],modes=['learned_backbone','learned_readout'],max_bytes=2*2**20,
  preflight_commands=0,pilot_commands=12,pilot_max_physics=physics,root_max_physics=physics,root_max_commands=12,root_max_main_seconds=300,absolute_native_stop=p['absolute_all_stop']-240,
  scientific_scope='Two frozen module recombinations encoderGRU vs headadapters, each oneActor two original tasks+exact.0Adam/nottraining/notfullfield/promotion. Samebuffers and immutable parents. Shared source results reused only as historical controls; actual newphysics fully charged.',
  authorization='User checkandcontinuewithin15:30window. After T527 learning improved target andlost oldskills, fixed paired module probe distinct fromnewtrainingorparamgrid.')
 p['protected_files']={p['initial']['path']:p['initial']['sha256'],p['learned']['path']:p['learned']['sha256']};p['inputs']={f:h for f,h in p['inputs'].items() if Path(f).suffix=='.py'}
 for f in [*HERE.glob('*.py'),DOC/'T528_PLAN.md',prior/'pilot/summary.json',prior/'pilot/audit.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T527.json']:p['inputs'][str(f)]=sha(f)
 p['inputs'].update(p['protected_files']);used=G.M.used();assert used+p['max_bytes']+65536<2**30
 OUT.mkdir();models={}
 for mode in p['modes']:models[mode]=R.export(R.hybrid(p,mode),OUT/(mode+'.pt.gz'))
 p['hybrid_models']=models
 for model in models.values():p['protected_files'][model['path']]=model['sha256'];p['inputs'][model['path']]=model['sha256']
 blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
 save(DOC/'T528_BUDGET.json',dict(max_physics=physics,max_commands=12,main_seconds=300,audit_seconds=60,backup_seconds=120,own_bytes=p['max_bytes'],global_reserved=used+p['max_bytes'],all_stop=p['absolute_all_stop'],Adam=0,CPU=[14,15],CPUQuota='100%',GPU=False));print('T528_REGISTERED_NOT_LAUNCHED',flush=True)
def preflight():
 p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'preflight.json').exists();a=G.E.N.S.arrays(p['keep_rows'][0]['path']);x=torch.tensor(a['states'][:12])
 for mode,model in p['hybrid_models'].items():
  q=R.hybrid(p,mode);loaded=R.A.frozen(p,model['path']);assert q.phash()==loaded.phash();q.sample=loaded.sample=False;h=k=None;values=[]
  with torch.no_grad():
   for z in x:
    u,loc,h,_=q.step(z[None],h);v,loc2,k,_=loaded.step(z[None],k);assert torch.equal(u,v) and torch.equal(h,k);values.append(loc[0])
   assert float((q.sequence(x)-torch.stack(values)).abs().max())<1e-10
 save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,Adam=0,commands=0,all_frozen_module_compositions_and_export_reload_exact=True,step_sequence_causal64carry_match=True,inference_only_models_NOT_resumable_Adam=True,native_worker_contract_reused_from_T527=True));print('T528_PREFLIGHT_PASS',flush=True)
def launch():
 p=read(OUT/'protocol.json');R.verify(p,True);pre=read(OUT/'preflight.json');assert pre['state']=='PASS' and not Path('/proc',str(pre['pid'])).exists() and not (OUT/'launch.json').exists()
 src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip() and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
 seconds=int(p['absolute_all_stop']-time.time());assert seconds>540
 subprocess.run(['systemd-run','--user','--unit=stage-t528-module-transfer-probe','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T528','--why=Bounded frozen module diagnostic nottraining','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],check=True);print('T528_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
