"""Assemble finite scope experiment, strict provenance and zero-update checks."""
from pathlib import Path
import ast,copy,gzip,hashlib,json,os,sys
import t652_runtime as T
ROOT,HERE,OUT,DOC=T.ROOT,T.HERE,T.OUT,T.DOC

def main():
    assert not OUT.exists()
    prior=ROOT/'stage/runs/t651_canonical_outcome_pg';p=T.read(prior/'protocol.json');s=T.read(prior/'pg/summary.json')
    assert T.read(prior/'pg/audit.json')['state']=='PASS' and T.read(prior/'SEALED.json')['all_scientific_processes_stopped']
    for k in ['registration_epoch','absolute_learning_stop','absolute_native_stop','absolute_all_stop','_root_sha256']:p.pop(k,None)
    rows=json.loads(gzip.decompress((prior/'pg/JOINT407.json.gz').read_bytes()))['rows'];by={r['job_id']:r for r in rows}
    jobs={j['id']:copy.deepcopy(j) for j in p['evaluation_jobs']};manifest=T.read(ROOT/'stage/runs/t646_offline_night/OFFLINE_INPUTS.json');old={r['id']:r for r in manifest['members']}
    lost=s['final407']['lost_prior'];assert len(lost)==6
    for key in lost:
        r=old[key];assert T.sha(r['offline_path'])==r['sha256'];a=T.R.E.N.S.arrays(r['offline_path'])
        contract=dict(scene_sha256=bytes(a['scene_sha256']).hex(),initial120_sha256=hashlib.sha256(a['states'][0].tobytes()).hexdigest(),integration_sha256=hashlib.sha256(a['initial_native_integration'].tobytes()).hexdigest(),goal_xy=a['task_goal_xy'].tolist())
        jobs[key]['initial_contract']=contract
    screen=p['screen_jobs']+[jobs[k] for k in lost];keys=p['key_jobs']+[jobs[k] for k in lost]
    assert len(screen)==len({j['id'] for j in screen})==36 and len(keys)==len({j['id'] for j in keys})==20
    reconstruction=[jobs[k] for k in lost];recon_worst=2*sum(j['case']['step_cap'] for j in reconstruction)
    evaluate_worst=sum(j['case']['step_cap'] for j in p['evaluation_jobs']+keys)
    p.update(id='T652_FIXED_BODY_OUTCOME_PG',out=str(OUT),arms=['pg'],authorization_approved=False,
        reconstruction_jobs=reconstruction,reconstruction_references=[by[k] for k in lost],reconstruction_model=s['proposal'],
        screen_jobs=screen,key_jobs=keys,old_screen_ids=p['old_screen_ids']+lost,
        total_max_physics=12500000,total_max_commands=460,total_main_seconds=3000,Actor_each_cap=8,
        reconstruction_physics=recon_worst,reconstruction_main_seconds=450,reconstruction_audit_seconds=240,
        reconstruction_bytes=48*2**20,child_bytes=128*2**20,max_bytes=192*2**20,global_bytes=10*2**30,
        main_seconds=2400,audit_seconds=600,learning_seconds=600,all_seconds=7200,backup_seconds=600,source_seconds=240,
        max_physics=12100000,max_commands=433,fit_steps=8,fit_lr=.00003,preflight_optimizer=0,new_Critic=0,new_Actor_trainable=11906)
    assert recon_worst+evaluate_worst<=p['total_max_physics'] and 18+433<=p['total_max_commands']
    assert p['reconstruction_main_seconds']+p['main_seconds']<=p['total_main_seconds']
    assert 450+240+2400+600+600+240+120<7200
    own_upper=12*p['fresh_bytes']+2*p['full_bytes']+20*p['fresh_bytes']+p['model_bytes']+407*p['compact_bytes']+20*2**20
    assert own_upper<=p['max_bytes']
    assert T.read(DOC/'SCOPE_PREFLIGHT.json')['state']=='PASS_ZERO_NATIVE_ZERO_OPTIMIZER'
    # Fresh data loader is the original T651 path, not a silently copied or
    # remapped learner dataset. Manifest equality is tested in actual main/audit.
    p['scope_reference']=dict(model=s['proposal'],source=str(prior),full407=s['final407'],all_param_Adam=T.read(prior/'pg/LEARNING.json')['update']['Actor_Adam'])
    p['inputs'].update({str(f):T.sha(f) for f in HERE.glob('*.py')})
    p['inputs'].update({str(DOC/'PLAN.md'):T.sha(DOC/'PLAN.md'),str(DOC/'SCOPE_PREFLIGHT.json'):T.sha(DOC/'SCOPE_PREFLIGHT.json'),str(DOC/'PRIMARY_NEXT_REVIEW.md'):T.sha(DOC/'PRIMARY_NEXT_REVIEW.md'),str(prior/'pg/summary.json'):T.sha(prior/'pg/summary.json'),str(prior/'pg/audit.json'):T.sha(prior/'pg/audit.json'),str(prior/'pg/JOINT407.json.gz'):T.sha(prior/'pg/JOINT407.json.gz')})
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=T.sha(Path(f).resolve())
    for r in [*p['reconstruction_references'],s['proposal']]:p['inputs'][r['path']]=r['sha256']
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    for f,h in p['inputs'].items():assert T.sha(f)==h,f
    T.save(HERE/'protocol.json',p)
    T.save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',actual_scope_preflight=str(DOC/'SCOPE_PREFLIGHT.json'),
       six_reconstruction_reference_hashes_locked=True,original76_data_reused_without_new_independence=True,screen36_key20_ids_unique=True,
       native_worst=recon_worst+evaluate_worst,commands_worst=451,storage_worst=own_upper,all_Actor_steps_new=0,
       only11906_existing_adapters_trainable=True,source_locks=len(p['inputs']),new_optimizer=0,new_native=0))
    T.save(DOC/'BUDGET.json',{k:p[k] for k in ['total_max_physics','total_max_commands','total_main_seconds','Actor_each_cap','max_bytes','global_bytes','reconstruction_physics','reconstruction_main_seconds','reconstruction_audit_seconds','main_seconds','audit_seconds','learning_seconds','all_seconds','backup_seconds','source_seconds']})
    print('T652_PREFLIGHT_PASS_ZERO_NATIVE_ZERO_OPTIMIZER',recon_worst+evaluate_worst,own_upper,flush=True)

if __name__=='__main__':main()
