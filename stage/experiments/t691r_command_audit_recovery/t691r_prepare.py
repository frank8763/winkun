"""Red/green command accounting on original evidence, zero repeated updates."""
from pathlib import Path
import ast,copy,json,os,sys,time
import t691r_runtime as S
ROOT,HERE,OUT,DOC=S.ROOT,S.HERE,S.ROOTOUT,S.ROOTDOC
L,T,torch,np=S.L,S.T,S.torch,S.np
read,sha,digest=S.read,S.sha,S.digest
from status_io import save

def main():
    start=time.time();os.sched_setaffinity(0,{14});assert not OUT.exists()
    src=ROOT/'stage/runs/t691_induced_state_rehearsal';old=read(src/'protocol.json');rep=read(src/'FINAL_REPORT.json');seal=read(src/'SEALED.json')
    assert rep['state']=='FAILED' and len(rep['children'])==1 and rep['children'][0]['Actor_calls']==dict(Actor_live=64,Actor_shadow=64,Actor_independent=0)
    assert all(not Path('/proc',str(i)).exists() for i in seal['scientific_pids']+[rep['helper_pid'],read(src/'DRIVER_STARTED.json')['pid']])
    s=read(src/'nominal/summary.json');calls=[json.loads(x) for x in (src/'nominal/optimizer_calls.jsonl').read_text().splitlines()]
    assert s['state']=='COMPLETE' and s['physics']==0 and s['commands']==6 and s['fit']['Actor_Adam']==64 and not s['fit']['accepted']
    assert not (src/'nominal/cases.jsonl.gz').exists() and not (src/'nominal/worker_generations.json').exists()
    for who in ['live','shadow']:assert [(r[1],r[3]) for r in calls if r[0]==who]==[(k,stage) for k in range(64) for stage in ['BEFORE','AFTER']]
    assert not any(r[0]=='independent_history' for r in calls)
    oldsource=(ROOT/'stage/experiments/t691_induced_state_rehearsal/t691_runtime.py').read_text()
    assert "(len(allrows)+6 if allrows else 3)==s['commands']" in oldsource
    old_rule=lambda n:n+6 if n else 3
    assert old_rule(0)!=s['commands'] and 0+6==s['commands']
    assert all(old_rule(n)==n+6 for n in [1,55,110,462])
    p=copy.deepcopy(old);p.pop('registration_epoch',None);p.update(id='T691R_COMMAND_ACCOUNTING_AUDIT_RECOVERY',authorization_approved=False,out=str(OUT),recovery_parent=str(src),
        remaining_nominal_audit_seconds=old['child_audit_seconds']-rep['children'][0]['audit_seconds'],
        night_main_seconds=old['child_main_seconds'],audit_total_seconds=old['audit_total_seconds']-rep['children'][0]['audit_seconds'],
        night_max_physics=old['child_max_physics'],night_max_commands=old['child_max_commands'],new_optimizer=64,
        max_bytes=old['max_bytes']-T.R.used(src),original_nominal_child_stop=read(src/'nominal/protocol.json')['absolute_all_stop'])
    assert p['absolute_all_stop']==1795262168.7778456 and time.time()+p['night_main_seconds']+p['audit_total_seconds']+120<p['absolute_all_stop']
    a=read(DOC.parent/'AUTHORIZATION.json');assert p['max_bytes']>p['child_bytes']+p['final_storage_reserve']
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);T.setup(dict(p,out=str(OUT/'nominal'),arm='nominal'))
    q,opt=L.make(p);before=digest(L.optimizer_state(q,opt));state=L.M.P.checkpoint(s['proposal']['path'])['joint_state']
    assert sha(s['proposal']['path'])==s['proposal']['sha256']
    L.load(q,opt,dict(actor=state['actor'],actor_opt=state['actor_opt']));assert digest(L.optimizer_state(q,opt))==read(src/'nominal/LEARNING.json')['trained_hash']
    q2,opt2=L.make(p);L.load(q,opt,L.optimizer_state(q2,opt2));assert digest(L.optimizer_state(q,opt))==before
    assert T.N.P.frozen==T.N.L.frozen and T.R.N.worker==T.N.worker
    for f,h in old['inputs'].items():assert sha(f)==h,f
    resources=S.Wait.Raw.resources();assert T.R.used(ROOT/'stage/runs')+p['max_bytes']<p['global_bytes']
    pre=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',original_no_rows_6_commands_red_reproduced=True,corrected6_management_rule_green=True,
        nonempty_rule_unchanged=True,original64live64shadow_preserved_not_reexecuted=True,original_independent_zero=True,
        actual_checkpoint_optimizer_full_load_and_rollback_exact=True,all_prior_processes_gone=True,real_loader_canonical_worker_bound=True,
        original_root_deadline_unchanged=p['absolute_all_stop'],remaining_nominal_audit=p['remaining_nominal_audit_seconds'],new_native=0,new_optimizer=0,resources=resources)
    save(DOC/'PREFLIGHT.json',pre)
    deps={Path(f) for f in old['inputs']}|set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC.parent/'t691/INTERRUPTION_REVIEW.json'}
    deps|={src/f for f in ['protocol.json','FINAL_REPORT.json','SEALED.json','DRIVER_STARTED.json','nominal/protocol.json','nominal/summary.json','nominal/LEARNING.json','nominal/ROLLBACK.json','nominal/NO_NATIVE.json','nominal/optimizer_calls.jsonl','nominal/audit.log']}
    deps.add(Path(s['proposal']['path']))
    for module in list(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):deps.add(Path(f).resolve())
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    p['inputs']={str(f):sha(f) for f in sorted(deps)};p['preflight_seconds']=time.time()-start
    assert rep['children'][0]['science_seconds']+p['preflight_seconds']+p['night_main_seconds']<old['night_main_seconds']
    save(HERE/'protocol.json',p);save(DOC/'BUDGET.json',{k:p[k] for k in ['new_optimizer','night_main_seconds','audit_total_seconds','remaining_nominal_audit_seconds','night_max_physics','night_max_commands','max_bytes','absolute_all_stop','preflight_seconds']})
    print(json.dumps(pre),flush=True)

if __name__=='__main__':main()
