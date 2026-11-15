"""Frozen BC checkpoint comparison. Every decision autonomous; no learning."""
from pathlib import Path
import fcntl,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t557_bc_stage_isolation';DOC=ROOT/'stage/docs/day_20261115/t557'
sys.path.insert(0,str(ROOT/'stage/experiments/t555_adam_direction_control'));import t555_runtime as Old
sys.path.insert(0,str(ROOT/'stage/experiments/t552_joint_robust_night'));import t552_actor as L
Base,F,G,E,P,np,torch=Old.Base,Old.F,Old.G,Old.E,Old.P,L.np,L.torch
read,save,sha,digest=L.read,L.save,L.sha,L.digest

def used(f):return sum(x.stat().st_size for x in Path(f).rglob('*') if x.is_file() and not x.is_symlink())
def verify(p,full=False):
    root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert time.time()<root['absolute_all_stop'] and used(OUT)<=root['max_bytes'] and used(ROOT/'stage/runs')-used(OUT)+root['max_bytes']+131072<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized() and set(os.sched_getaffinity(0))<={14,15}
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f

def configuration():
    root=read(OUT/'protocol.json');p=read(root['actor_configuration']);p.update(root);p['out']=str(OUT);return p

def setup():
    Old.OUT=OUT;Old.HERE=HERE;Old.DOC=DOC;Old.verify=verify;Old.setup(OUT)
    Base.ACTIVE=OUT;P.frozen=L.frozen;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

def compare(p,rows):
    results=[]
    for name,model in p['models'].items():
        measured={r['job_id']:r for r in rows if r['role']=='BC_ONLY_'+name};assert set(measured)==set(p['job_ids'])
        for ident in p['job_ids']:
            r=measured[ident];prior=p['baseline_rows'][ident];post=p['post_RL_rows'][name][ident]
            initial=E.N.S.arrays(r['path'])['initial_native_integration'];assert digest(initial)==prior['array_hashes']['initial_native_integration']==post['array_hashes']['initial_native_integration']
            safe=bool(r['success'] and not r['collision']);assert prior['success'] and not prior['collision']
            first='BC_ALREADY_FAILS' if not safe else 'BC_SAFE_POST_RL_FAILS' if not (post['success'] and not post['collision']) else 'BC_AND_POST_RL_SAFE'
            results.append(dict(arm=name,id=ident,baseline_safe=True,BC_safe=safe,BC_contact=r['collision'],BC_timeout=r['timeout'],BC_remaining_m=r['remaining_m'],BC_seconds=r['physics']*.002,post_RL_safe=bool(post['success'] and not post['collision']),post_RL_contact=post['collision'],post_RL_timeout=post['timeout'],stage=first,BC_raw_sha256=r['sha256']))
    return dict(rows=results,new_Adam=0,new_RL=0,new_teacher=0,not_full38_or_navigation_gate=True,limitations='Only these six consumed tasks, paired actual autonomous continuations. BC/RL attribution is conditional on respective frozen checkpoints; no unique algorithmic causal claim or unseen robustness.')

def main():
    p=configuration();setup();verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=Base.Runtime(p);state='FAILED';error=None;result=None
    try:
        for i,(name,model) in enumerate(p['models'].items()):
            if i:rt.commands+=2
            rt.pool=Old.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
            values=rt.request(p['jobs'],model,'BC_ONLY_'+name,full=True);rows=[r for a,r in values]
            rt.request(p['jobs'],model,'EXACT_BC_ONLY_'+name,full=True,reference=rows)
            rt.pool.close();rt.pool=None;rt.commands+=2
        result=compare(p,rt.rows);save(OUT/'comparison.json',result);state='COMPLETE'
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,no_restart=True,raw_preserved=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(OUT/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error

def audit():
    p=configuration();setup();verify(p,True);G.stopped(OUT);assert not (OUT/'audit.json').exists();s=read(OUT/'summary.json');assert s['state']=='COMPLETE'
    rows=G.rows(OUT);reps=G.rows(OUT,'replays.jsonl');assert len(rows)==len(reps)==12;by={r['path']:r for r in rows}
    for r in rows:
        a=E.N.S.arrays(r['path']);assert sha(r['path'])==r['sha256'] and int(a['native_counts'].sum())==r['physics'];assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
        assert not a['ownership'].any() and not a['stochastic'].any();F.M.audit_full_history(p,r,a)
    for r in reps:
        src=by[r['reference_path']];assert r['exact'] and r['array_hashes']==src['array_hashes'] and r['integration_sha256']==src['integration_sha256']
    assert compare(p,rows)==s['result']==read(OUT/'comparison.json');assert s['physics']==sum(r['physics'] for r in rows+reps)<=p['max_physics'];assert s['commands']==32<=p['max_commands']
    for m in p['models'].values():assert sha(m['path'])==m['sha256']
    save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,new_optimizer=0,all12_full_history_and12_native_exact=True,models_unchanged=True,physics=s['physics'],commands=s['commands'],no_candidate_selected=True))
    print('T557_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
