"""All407 tasks, predefined balanced chunks, no repeated seed rescue."""
from pathlib import Path
import ast,copy,json,os,sys,time
import t672_runtime as S
T=S.T;ROOT,HERE,OUT,DOC=S.ROOT,S.HERE,S.OUT,S.DOC
read,sha,digest=T.read,T.sha,T.digest
from status_io import save


def main():
    assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True)
    src=ROOT/'stage/runs/t671_joint_outcome_feasibility';summary=read(src/'summary.json')
    assert read(src/'audit.json')['state']=='PASS' and not summary['full407']['accepted']
    assert summary['full407']['lost_prior']==['dev_task6_13','dev_task6_25','dev_task7_17']
    assert all(not Path('/proc',str(pid)).exists() for pid in read(src/'SEALED.json')['scientific_pids'])
    assert not Path('/proc',str(read(src/'DRIVER_STARTED.json')['pid'])).exists()
    old=read(src/'protocol.json');p=copy.deepcopy(old);p.pop('registration_epoch',None)
    auth=read(DOC.parent/'AUTHORIZATION.json');assert auth['approved'];now=time.time()
    fields={j['id'] for j in p['field_jobs']};other=sorted([j for j in p['evaluation_jobs'] if j['id'] not in fields],key=lambda j:j['id'])
    assert len(other)==383 and len(fields)==24
    configs=[]
    for index in range(6):
        jobs=other[index::6]
        configs.append(dict(name=f'nonfield_{index:02d}',task_jobs=jobs,replicates=4))
    ff=sorted(p['field_jobs'],key=lambda j:j['id'])
    for index in range(2):configs.append(dict(name=f'field_{index:02d}',task_jobs=ff[index::2],replicates=8))
    seeds=set();all_ids=[]
    for index,c in enumerate(configs):
        jobs=[j for j in c['task_jobs'] for _ in range(c['replicates'])]
        c.update(kind='coverage',arm=c['name'],frozen_jobs=jobs,frozen_seeds=[[67220261120,index,i] for i in range(len(jobs))],
            frozen_full=True,actor_cap=0,value_cap=0,main_seconds=6000,audit_seconds=600,
            max_physics=2*sum(j['case']['step_cap'] for j in jobs),max_commands=2*len(jobs)+6,
            child_bytes=len(jobs)*(4*2**20+32768)+16*2**20)
        all_ids.extend(j['id'] for j in c['task_jobs'])
        for seed in c['frozen_seeds']:assert tuple(seed) not in seeds;seeds.add(tuple(seed))
    assert len(all_ids)==len(set(all_ids))==407 and len(seeds)==1724
    for c in configs:c.pop('task_jobs')
    p.update(id='T672_SYSTEMATIC_FULL407_ONPOLICY_COLLECTION',authorization_approved=False,out=str(OUT),
        batch_configurations=configs,learner_device='cpu',raw_sigma=[.05,.05],initial=old['initial'],
        max_bytes=7*2**30,global_bytes=10*2**30,final_storage_reserve=16*2**20,
        night_max_physics=sum(c['max_physics'] for c in configs),night_max_commands=sum(c['max_commands'] for c in configs),
        night_main_seconds=48000,audit_total_seconds=4800,Actor_each_cap=0,Value_each_cap=0,
        absolute_learning_stop=min(now+16*3600,auth['absolute_learning_stop']),
        absolute_native_stop=min(now+16*3600-900,auth['absolute_native_stop']),
        absolute_all_stop=min(now+16*3600,auth['absolute_all_stop']),finalization_seconds=60,
        weekend_authorization=str(DOC.parent/'AUTHORIZATION.json'),new_Actor=0,new_Value=0,new_optimizer=0)
    worst=sum(c['child_bytes'] for c in configs)+32*2**20
    assert worst<=p['max_bytes']
    assert T.R.used(ROOT/'stage/runs')+p['max_bytes']<p['global_bytes']
    assert p['night_max_physics']<=auth['max_new_physics'] and p['night_max_commands']<=auth['max_new_commands']
    os.sched_setaffinity(0,{14});T.torch.set_num_threads(1);T.setup(p)
    q=T.M.L.frozen(p,p['initial']['path']);assert q.phash()==p['initial']['policy_hash']
    assert T.R.N.Pool.__init__.__globals__['worker'] is T.N.worker
    # Real inherited child-plan merge/worker/transport paths, no native call.
    for c in configs:
        child=dict(p);child.update(c);child['out']=str(OUT/c['name'])
        assert len(child['frozen_seeds'])==len(child['frozen_jobs'])
        assert 2*len(child['frozen_jobs'])+6==child['max_commands']
        assert child['main_seconds']+child['audit_seconds']<=7200
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    resource=S.Wait.Raw.resources()
    pre=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',all407_unique_task_ids=True,protected384_all_included=True,
        task_weighting_not_claimed_by_sample_counts=True,nonfield383_four_each=True,field24_eight_each=True,
        fresh_units=1724,exact_units=1724,fixed_unique_seed_units=True,all8_child_plan_shapes_checked=True,
        canonical_worker_and_same_policy_loader=True,storage_worst=worst,resources=resource,new_native=0,new_optimizer=0)
    save(DOC/'PREFLIGHT.json',pre)
    deps={Path(f) for f in old['inputs']}|set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC.parent/'AUTHORIZATION.json',src/'summary.json',src/'audit.json'}
    for f,h in old['inputs'].items():assert sha(f)==h,f
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):deps.add(Path(f).resolve())
    p['inputs']={str(f):sha(f) for f in sorted(deps)};(HERE/'protocol.json').write_text(json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n')
    save(DOC/'BUDGET.json',{k:p[k] for k in ['night_max_physics','night_max_commands','night_main_seconds','audit_total_seconds','max_bytes','global_bytes','absolute_native_stop','absolute_all_stop']}|dict(batch_budgets=[{k:c[k] for k in ['name','max_physics','max_commands','child_bytes','main_seconds','audit_seconds']} for c in configs]))
    print(json.dumps(pre))


if __name__=='__main__':main()
