"""Real-history suffix sampler/density reproducibility; no native or optimizer."""
from pathlib import Path
import ast,sys,time
import t644_runtime as T
R=T.R;ROOT,HERE,DOC,OUT=T.ROOT,T.HERE,T.DOC,T.OUT


def main():
    assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True)
    p=R.read(ROOT/'stage/experiments/t643_old_policy_recovery/protocol.json')
    for key in ['registration_epoch','absolute_native_stop','absolute_learning_stop','absolute_all_stop','user_authorization']:p.pop(key,None)
    old=ROOT/'stage/runs/t643_old_policy_recovery';s=R.read(old/'summary.json')
    assert s['state']=='COMPLETE' and not s['result']['qualified'] and R.read(old/'audit.json')['state']=='PASS'
    p.update(id='T644_FROZEN_SAMPLED_ACTUAL_RECOVERY',out=str(OUT),authorization_approved=False,
        exploration_seeds=[[20261118644,i] for i in range(8)],max_physics=640000,max_commands=22,max_bytes=26*2**20,
        main_seconds=600,audit_seconds=300,transport_seconds=120,source_seconds=90,all_seconds=1800,
        new_optimizer=0,new_Actor_Adam=0,shadow_Actor_Adam=0,independent_historical_Actor_Adam=0)
    assert p['raw_sigma']==[.2,.2] and 16*p['handoff_job']['case']['step_cap']==p['max_physics']
    R.B.setup(p);a=R.E.N.S.arrays(p['source_row']['path']);k=p['first_divergence']['decision'];assert k==7
    x=R.torch.tensor(a['states'][:k+5]);signatures=[];models=[]
    for seed in p['exploration_seeds'][:2]:
        duplicate=[]
        for repeat in range(2):
            q=R.L.frozen(p,p['initial']['path']);q.rng=R.np.random.default_rng(R.np.random.SeedSequence(seed));q.reset_noise();h=None;before=R.digest(q.state_dict())
            with R.torch.no_grad():
                for i,z in enumerate(x):
                    q.sample=i>=k;u,loc,h,_=q.step(z[None],h)
                    row=q.tape[-1]
                    if i<k:assert not row['innovations'].any() and not row['stochastic']
                    else:
                        assert row['stochastic'] and row['innovations'].any()
                        logp=R.L.P.L.density(loc,R.torch.tensor(row['raw_actions'][None]),R.torch.tensor(row['atom'][None]),p['raw_sigma'])
                        assert abs(float(logp[0])-float(row['logp']))<1e-12
            assert R.digest(q.state_dict())==before and h is not None and bool(h.abs().max()>0)
            duplicate.append(R.digest(q.tape));models.append(q.phash())
        assert duplicate[0]==duplicate[1];signatures.append(duplicate[0])
    assert signatures[0]!=signatures[1] and len(set(models))==1
    assert not R.torch.cuda.is_initialized()
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    files=set(HERE.glob('*.py'))|{DOC/'PLAN.md',old/'protocol.json',old/'summary.json',old/'audit.json',old/'SEALED.json'}
    for module in list(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):files.add(Path(f).resolve())
    p['inputs'].update({str(f):R.sha(f) for f in files})
    parent=R.read(ROOT/'stage/runs/t642_tail_retention/protocol.json')
    prior=[R.read(ROOT/'stage/runs'/name/'summary.json') for name in ['t642_tail_retention','t643_old_policy_recovery']]
    physics=sum(z['physics'] for z in prior)+p['max_physics'];commands=sum(z['commands'] for z in prior)+p['max_commands']
    assert physics<=parent['max_physics'] and commands<=parent['max_commands']
    reserved=sum(R.used(ROOT/'stage/runs'/name) for name in ['t642_tail_retention','t643_old_policy_recovery'])+p['max_bytes']
    assert reserved<=parent['max_bytes']
    glob=R.used(ROOT/'stage/runs')+p['max_bytes']+131072;assert glob<2**30
    assert time.time()+p['main_seconds']+p['audit_seconds']+p['transport_seconds']+p['source_seconds']+120<parent['absolute_all_stop']
    R.save(HERE/'protocol.json',p)
    R.save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',actual_source_SHA=p['source_row']['sha256'],
         same_seed_full_history_sampler_reproducible=True,different_seeds_different_innovations=True,
         prefix_deterministic_suffix_stochastic=True,actual_clipped_atom_log_density_checked=True,teacher_weights_unchanged=True,
         own_carry_nonzero=True,preflight_not_new_physical_trajectory=True,combined_physics=physics,combined_commands=commands,
         combined_reserved_bytes=reserved,parent_bytes=parent['max_bytes'],global_worst=glob,new_optimizer=0))
    R.save(DOC/'BUDGET.json',{key:p[key] for key in ['max_physics','max_commands','max_bytes','main_seconds','audit_seconds','transport_seconds','source_seconds','new_optimizer']})
    print('T644_PREFLIGHT_ZERO_NATIVE_ZERO_OPTIMIZER_PASS',physics,commands,glob,flush=True)


if __name__=='__main__':main()
