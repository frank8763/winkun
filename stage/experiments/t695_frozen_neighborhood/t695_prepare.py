"""Real frozen-loader/native-audit entry contracts; no physics or optimizer."""
from pathlib import Path
import ast,copy,importlib.util,inspect,json,os,resource,sys,time,unittest.mock as mock
import t695_runtime as S
A,N,T,torch,np=S.A,S.N,S.T,S.torch,S.np
ROOT,HERE,OUT,DOC=S.ROOT,S.HERE,S.OUT,S.DOC
read,sha,digest=S.read,S.sha,S.digest
from status_io import save


def reject(call):
    try:call()
    except AssertionError:return
    raise AssertionError('Corrupt input accepted')


def signature(job):
    return tuple(job['fullfield_start_pose']),tuple(job['fullfield_goal_xy'])


def main():
    started=time.time();os.sched_setaffinity(0,{14});assert not OUT.exists()
    assert not (DOC/'PREFLIGHT.json').exists(),'Never repeat completed contracts'
    review=DOC.parent/'T695_FROZEN_NEIGHBORHOOD_PRIMARY_REVIEW.md';assert review.exists()
    source=ROOT/'stage/runs/t694_full_history_compatibility';p=copy.deepcopy(read(source/'protocol.json'));p.pop('inputs',None);p.pop('registration_epoch',None)
    retained=read(DOC.parent/'RETAINED_MODEL.json');assert retained['accepted_gate']['success']==390 and retained['remote_raw_backup']['named_model_verified']
    assert read(ROOT/'stage/runs/t694r_audit_recovery/audit.json')['state']=='PASS'
    auth=read(DOC.parent/'AUTHORIZATION.json');ids=retained['accepted_gate']['gained'];assert len(ids)==6
    old={j['id']:j for j in p['evaluation_jobs']};jobs=[]
    for key in ids:
        for axis,sign in [(0,-1),(0,1),(1,-1),(1,1)]:
            j=copy.deepcopy(old[key]);j.pop('initial_contract');j['fullfield_start_pose'][axis]+=sign*.01
            direction=('x' if axis==0 else 'y')+('minus' if sign<0 else 'plus')
            j.update(id=key+'_T695_'+direction,source_reset_id=key+'_T695_'+direction,neighborhood_parent_id=key,neighborhood_direction=direction)
            jobs.append(j)
    assert len(jobs)==len(set(signature(j) for j in jobs))==24
    # Conservatively classify exposure by exact pose/goal, including other
    # models/protocols. Do not change the fixed points to obtain novelty.
    wanted={signature(j):j['id'] for j in jobs};matches={j['id']:set() for j in jobs};scanned=0
    def walk(value,path):
        if isinstance(value,dict):
            if 'fullfield_start_pose' in value and 'fullfield_goal_xy' in value:
                try:sig=signature(value)
                except TypeError:sig=None
                if sig in wanted:matches[wanted[sig]].add(str(path))
            for v in value.values():
                if isinstance(v,(dict,list)):walk(v,path)
        elif isinstance(value,list):
            for v in value:
                if isinstance(v,(dict,list)):walk(v,path)
    for parent in [ROOT/'stage/experiments',ROOT/'stage/runs']:
        for f in parent.glob('*/protocol.json'):
            if f.is_relative_to(HERE) or f.is_relative_to(OUT):continue
            walk(read(f),f);scanned+=1
    exposure=dict(method='Exact fullfield_start_pose and fixedgoal equality across existing Stage root protocol metadata; not all old trajectories or original chat history',scanned_protocols=scanned,matches={k:sorted(v) for k,v in matches.items()},new_to_scanned_protocols=[k for k,v in matches.items() if not v],no_point_replacement=True)
    save(DOC/'EXPOSURE.json',exposure)
    p.update(id='T695_FROZEN_T694_T501_FIXED_1CM_XY_NEIGHBORHOOD',authorization_approved=False,out=str(OUT),arm='frozen_neighborhood',neighborhood_jobs=jobs,parent_ids=ids,
        models=[dict(name='T501',model=retained['original_frozen_parent']),dict(name='T694',model=retained['model'])],retained=retained['model'],
        max_physics=4*sum(j['case']['step_cap'] for j in jobs),max_commands=128,new_Actor=0,new_Value=0,new_optimizer=0,
        max_bytes=256*2**20,child_bytes=256*2**20,main_seconds=2400,audit_seconds=600,root_seconds=3300,learning_seconds=0,finalization_seconds=60,
        absolute_learning_stop=auth['absolute_learning_stop'],absolute_native_stop=time.time()+2700,absolute_all_stop=time.time()+3300,global_bytes=10*2**30)
    assert p['max_physics']==3801552 and p['max_physics']/2500+120<p['main_seconds']
    storage_upper=48*p['full_bytes']+48*p['compact_bytes']+3*p['full_bytes']+24*2**20
    assert storage_upper<p['max_bytes'] and T.R.used(ROOT/'stage/runs')+p['max_bytes']<auth['archive_trigger_bytes']
    S.Wait.Raw.resources();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);S.setup(p)
    def forbidden(*args,**kwargs):raise AssertionError('No native or optimizer in preflight')
    torch.optim.Adam.step=forbidden
    import mujoco
    mujoco.mj_step=forbidden
    checks=[]
    for item in p['models']:
        m=item['model'];assert sha(m['path'])==m['sha256'];q=A.frozen(p,m['path']);assert q.phash()==m['policy_hash'] and all(not v.requires_grad for v in q.parameters())
        before=digest(q.state_dict())
        if item['name']=='T694':
            summary=read(source/'evaluation/summary.json');r=next(r for r in summary['screen'] if r['job_id']==ids[0])
        else:
            row=p['protected_rows'][0];a=T.R.E.N.S.arrays(row['source_path'])
            r=dict(job_id=row['task_id'],request=dict(model=m['path'],job=old[row['task_id']]),seed=None,teacher=False,path=row['source_path'],physics=int(a['native_counts'].sum()),array_hashes={k:digest(v) for k,v in a.items()})
        aa=T.R.E.N.S.arrays(r['path']) if 'path' in r else a
        inspector=N.inspect if item['name']=='T694' else N.ORIGINAL_INSPECT
        inspector(p,aa,r)
        hidden=None
        with torch.no_grad():
            for i,x in enumerate(aa['states']):
                expected=np.zeros(64) if hidden is None else hidden[0,0].numpy()
                assert np.array_equal(expected,aa['hidden_before'][i])
                command,mu,hidden,_=q.step(torch.tensor(x[None]),hidden)
                assert A.same_float32(command[0].numpy(),aa['actions'][i])
        assert digest(q.state_dict())==before
        bad=dict(aa,ownership=aa['ownership'].copy());bad['ownership'][0]=1;reject(lambda:inspector(p,bad,r))
        if item['name']=='T694':
            broken=copy.deepcopy(r);broken['request']['job']['fullfield_start_pose'][0]+=.5;reject(lambda:N.inspect(p,aa,broken))
        checks.append(dict(name=item['name'],policy_hash=q.phash(),source_path=r['path'],source_sha256=sha(r['path']),decisions=len(aa['states'])))
    seen=[];original=os.sched_setaffinity
    def witness(pipe,index,pc,counts):
        assert N.G.T.Session is N.Session and N.P.frozen is A.frozen;seen.append(index)
    try:
        with mock.patch.object(N.G.T,'worker',witness):N.worker(None,0,p,None)
    finally:os.sched_setaffinity=original
    assert seen==[0] and T.R.N.Pool.__init__.__globals__['worker'] is N.worker
    constructor=inspect.getsource(N.C.Session.__init__);assert 'self.steps==0' in constructor and 'self.hidden.any()' in constructor and 'self.scene==expected' in constructor
    spec=importlib.util.spec_from_file_location('_T695_helper',HERE/'affinity_guard.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    assert helper.OUT==OUT and helper.CG.name=='stage-t695.service'
    assert 24*2*2+6+26==128
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;assert peak<auth['memory_bytes_cap']
    pre=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',actual_frozen_models_batch1_owncarry_commands=True,standard_fullhistory_native_audits=True,corrupt_ownership_and_initialpose_rejected=True,realworker_canonicalfactory_loader_binding=True,new_scene_validity_checked_formal_before_each_firstphysical=True,new_native=0,new_optimizer=0,new_Actor=0,new_Value=0,checks=checks,storage_upper_bytes=storage_upper,peak_rss_bytes=peak,elapsed_seconds=time.time()-started)
    save(DOC/'PREFLIGHT.json',pre)
    # Preserve all inherited concrete runtime/source dependencies, omitting
    # obsolete data manifests that are never read by this frozen evaluation.
    prior=read(source/'protocol.json');deps={Path(f) for f in prior['inputs'] if str(f).endswith('.py')}
    deps|=set(HERE.glob('*.py'))|{review,DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC/'EXPOSURE.json',DOC.parent/'RETAINED_MODEL.json',DOC.parent/'T694_RETAINED_BACKUP.json',source/'protocol.json'}
    deps|={Path(v['model']['path']) for v in p['models']}|{Path(j['exact_snapshot']) for j in jobs}
    deps|={Path(v['source_path']) for v in checks}
    for module in list(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):deps.add(Path(f).resolve())
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    p['inputs']={str(f):sha(f) for f in sorted(deps)};p['preflight_seconds']=180+time.time()-started+5
    (HERE/'protocol.json').write_text(json.dumps(p,separators=(',',':'))+'\n');assert (HERE/'protocol.json').stat().st_size<2**21
    save(DOC/'BUDGET.json',{k:p[k] for k in ['max_physics','max_commands','max_bytes','main_seconds','audit_seconds','root_seconds','preflight_seconds']})
    print(json.dumps(pre),flush=True)


if __name__=='__main__':main()
