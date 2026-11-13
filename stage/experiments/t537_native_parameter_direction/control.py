"""Bounded registration for the user-requested next native measurement."""
from pathlib import Path
import copy,os,sys,time
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t537_runtime as R
ROOT,OUT,DOC,read,save,sha,digest=R.ROOT,R.OUT,R.DOC,R.read,R.save,R.sha,R.digest
def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t536_compact_native_restore';p0=read(prior/'protocol.json');s=read(prior/'summary.json')
    assert s['state']=='COMPLETE' and s['result']['restore_gate'] and read(prior/'audit.json')['state']=='PASS'
    assert all(not Path('/proc',str(i)).exists() for i in read(prior/'SEALED.json')['scientific_pids']+[read(prior/'launch.json')['pid']])
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T536.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    end=p0['absolute_all_stop'];assert time.time()+1380<end
    config=read(p0['actor_configuration']);parent=config['neural_teacher'];candidate=config['initial']
    q=R.A.frozen(config,parent['path']);names={k for k,v in q.named_parameters()};old=R.P.checkpoint(parent['path'])['joint_state']['actor'];new=R.P.checkpoint(candidate['path'])['joint_state']['actor'];d={k:new[k]-old[k] for k in sorted(names)}
    p={k:p0[k] for k in ['source_protocol','actor_configuration','protected_files']}
    windows=[]
    for wi,(w,row) in enumerate(zip(p0['windows'],s['result']['windows'])):
        item=copy.deepcopy(w);item.update(snapshot_path=row['snapshot_path'],snapshot_sha256=row['snapshot_sha256'],reference_suffix_path=row['payload_path'],reference_suffix_sha256=row['payload_sha256']);windows.append(item)
    p.update(id='T537_NATIVE_CONDITIONAL_PARAMETER_DIRECTION',windows=windows,holds=5,epsilon=.01,alphas=[0.,.01,-.01,.005,-.005],effect_floor=1e-7,relative_scale_tolerance=.25,max_first_command_change=.02,direction_parent=parent,direction_candidate=candidate,direction_sha256=digest(d),max_bytes=2**20,window_bytes=128*1024,main_seconds=480,audit_seconds=240,transport_seconds=240,source_seconds=120,absolute_all_stop=end,absolute_native_stop=end-900,absolute_learning_stop=end-1800,max_physics=20000,max_commands=45,scope='User latest next-step authorization after successful T536. Only fixed learned parameter direction T527minusT507, eps.01/.005 signed around each original source policy,4snapshots,5branches plus exactrepeat=40one-second branches. All perturbation models RAM-only,0optimizer/teacher/route/planner. Native future120/history/64carry generated anew. Hidden recomputed under perturbed weights on the same historical prefix: conditional experiment,not original-start full-episode gradient. No model promotion or automatic learning.',authorization='User explicitly 下一步开始吧 after T536 completion. New independent20k native budget, not T536-only73k continuation. Existing20:00deadline/CPU/GPU/globaldisk constraints preserved.')
    p['inputs']=dict(p0['inputs'])
    for w in windows:
        p['inputs'][w['snapshot_path']]=w['snapshot_sha256'];p['inputs'][w['reference_suffix_path']]=w['reference_suffix_sha256']
    for f in [*HERE.glob('*.py'),DOC/'T537_PLAN.md',prior/'protocol.json',prior/'summary.json',prior/'audit.json']:p['inputs'][str(f)]=sha(f)
    assert R.used(ROOT/'stage/runs')+p['max_bytes']+65536<2**30
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);R.verify(p,True);R.direction(p)
    save(DOC/'T537_BUDGET.json',dict(new_physics=20000,commands=45,new_Actor_Adam=0,new_Value_Adam=0,main_seconds=480,audit_seconds=240,backup_seconds=240,source_seconds=120,own_max_bytes=2**20,global_reserved=R.used(ROOT/'stage/runs')-R.used(OUT)+2**20,cumulative_T534_T536_T537_max=131900,absolute_native_stop=p['absolute_native_stop'],absolute_all_stop=end,branches=40))
    print('T537_REGISTERED_NOT_LAUNCHED',flush=True)
def preflight():
    p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'preflight.json').exists();config=read(p['actor_configuration']);d=R.direction(p);checks=[]
    for window in p['windows']:
        a=R.E.N.S.arrays(window['row']['path']);q=R.A.frozen(config,window['row']['request']['model']);base=copy.deepcopy(q.state_dict());first=window['first'];actions=[]
        for alpha in p['alphas']:
            R.perturb(q,base,d,alpha);h=R.hidden(q,a['states'][:first])
            with R.torch.no_grad():u,loc,h,_=q.step(R.torch.tensor(a['states'][first:first+1]),h)
            if alpha==0.:assert R.np.array_equal(u[0].numpy(),a['actions'][first])
            change=float(R.np.max(R.np.abs(u[0].numpy()-a['actions'][first])));assert change<=p['max_first_command_change'];actions.append(change)
        R.perturb(q,base,d,0.);assert q.phash()==window['row']['policy_hash']
        checks.append(dict(id=window['id'],max_first_command_changes=actions,whole_parameter_restore_exact=True))
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),new_native=0,new_optimizer=0,conditional_hidden_and_first_command_bounds=checks,actual_native_direction_NOT_tested=True));print('T537_PREFLIGHT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
