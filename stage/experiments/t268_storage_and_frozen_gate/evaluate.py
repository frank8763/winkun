"""Independent frozen-model missing-case acceptance, with full worst-case reserve."""
from pathlib import Path
import gzip
import hashlib
import io
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t268_storage_and_frozen_gate'
sys.path.insert(0,str(ROOT/'stage/experiments/t259_closed_loop_recovery'))
from closed_loop_learning import L,S,C,policy,np,torch
sys.path.insert(0,str(ROOT/'stage/experiments/t263_adaptive_goal_course'))
from course_worker import worker


class Stop(Exception):pass


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def readrows(path):
    p=Path(path);b=p.read_bytes()
    return list(map(json.loads,(gzip.decompress(b) if p.suffix=='.gz' else b).splitlines()))


def preflight(plan):
    assert not (OUT/'preflight.json').exists()
    for name,h in plan['input_hashes'].items():assert sha(name)==h,name
    p=policy(plan).eval().requires_grad_(False)
    assert S.digest(p.state_dict())==plan['policy_hash']
    old=readrows(plan['reused_old_rows']);assert len(old)==224
    assert all(r['policy_hash']==plan['policy_hash'] and r['complete'] and r['success'] and not r['collision'] for r in old)
    for r in old:assert sha(r['path'])==r['sha256']
    reused={r['source_reset_id'] for r in old}
    missing={r['id'] for r in plan['missing_old_jobs']}
    assert len(reused)==224 and len(missing)==51 and not reused&missing
    assert reused|missing==set(plan['old275_ids'])
    assert len(plan['development_jobs'])==54 and len(plan['heldout_jobs'])==16
    bound=plan['max_episode_records']*plan['episode_bytes']+plan['max_record_rows']*plan['record_bytes']+plan['metadata_bytes']
    assert bound<=plan['final_bytes']
    used=S.used(ROOT/'stage/runs');own=S.used(OUT)
    assert used-own+plan['final_bytes']<plan['global_bytes']
    assert shutil.disk_usage(ROOT).free>=plan['min_free_bytes']
    C.save(OUT/'preflight.json',dict(state='PASS',physics_steps=0,optimizer_calls=0,
        policy_hash=plan['policy_hash'],reused_224_same_policy_and_payload_hashes_verified=True,
        reused_results_not_new_physical_repeats=True,missing_old=51,six_development=54,
        full_conditional_worst_case_output_bytes=bound,registered_output_bytes=plan['final_bytes'],
        global_if_full=used-own+plan['final_bytes'],protocol_sha256=sha(HERE/'protocol.json')))


def main():
    plan=json.loads((HERE/'protocol.json').read_text())
    if '--preflight' in sys.argv:return preflight(plan)
    assert not (OUT/'status.json').exists() and not (OUT/'summary.json').exists(),'No duplicate/restart'
    assert json.loads((OUT/'preflight.json').read_text())['protocol_sha256']==sha(HERE/'protocol.json')
    for name,h in plan['input_hashes'].items():assert sha(name)==h,name
    start=time.time();deadline=start+plan['wall_seconds'];physics=commands=inflight=0
    processes=[];pipes=[];reason='COMPLETE';gate=False
    def status(state):C.save(OUT/'status.json',dict(state=state,pid=os.getpid(),physics_steps=physics,
        commands=commands,optimizer_calls=0,updated_epoch=time.time(),elapsed_s=time.time()-start))
    def append(name,row):
        text=json.dumps(row)+'\n';assert len(text.encode())<=plan['record_bytes']
        with (OUT/name).open('a') as f:f.write(text)
    def guard(n,steps):
        if time.time()>=deadline:raise Stop('WALL')
        if commands+n>plan['max_commands'] or physics+steps>plan['max_physics']:raise Stop('COUNTS')
        # Every possible future payload and row was reserved before launch.
        if S.used(OUT)>=plan['final_bytes'] or S.used(ROOT/'stage/runs')>=plan['global_bytes'] or shutil.disk_usage(ROOT).free<plan['min_free_bytes']:raise Stop('STORAGE')
    def receive(pipe):
        while not pipe.poll(.5):
            if time.time()>=deadline:raise Stop('RESPONSE_WALL')
        item=pipe.recv();assert 'error' not in item,item
        return item
    def jobs(rows,prefix):return [dict(r,id=prefix+'_'+r['id'],source_reset_id=r['id'],mode='deterministic',seed=0,memory_on=False,model_seed=2026103501) for r in rows]
    def batch(jj,name,geometry=False):
        nonlocal physics,commands,inflight
        result=[]
        for off in range(0,len(jj),4):
            part=jj[off:off+4];guard(len(part),0 if geometry else sum(j['case']['step_cap'] for j in part))
            inflight=0 if geometry else sum(j['case']['step_cap'] for j in part)
            for pipe,j in zip(pipes,part):pipe.send(dict(j,geometry_only=True) if geometry else dict(j,policy_np=state,policy_hash=plan['policy_hash']))
            rr=[receive(pipe) for pipe in pipes[:len(part)]];commands+=len(rr)
            if not geometry:physics+=sum(i['row']['physics_steps'] for i in rr)
            inflight=0
            for j,item in zip(part,rr):
                if geometry:row=item
                else:
                    row=item['row'];a=item['arrays'];buf=io.BytesIO()
                    np.savez_compressed(buf,**{k:v for k,v in a.items() if k not in ['states','next_states']})
                    payload=buf.getvalue();assert len(payload)<=plan['episode_bytes'],'Declared episode byte contract exceeded'
                    dest=OUT/'episodes'/(j['id']+'.npz');dest.parent.mkdir(exist_ok=True);assert not dest.exists()
                    dest.write_bytes(payload);row.update(path=str(dest),sha256=sha(dest))
                append(name,row);result.append(row)
            status(name)
        return result
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        p=policy(plan).eval().requires_grad_(False);assert S.digest(p.state_dict())==plan['policy_hash'];state=S.state_np(p)
        status('STARTING');ctx=mp.get_context('spawn')
        for k in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(k+1)
            w=ctx.Process(target=worker,args=(b,k,str(OUT),plan,deadline));w.start();b.close();processes.append(w);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';assert all(receive(p)['ready'] for p in pipes)
        C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[w.pid for w in processes],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
        gg=batch(jobs(plan['development_jobs']+plan['heldout_jobs'],'geometry'),'geometry.jsonl',True)
        assert len(gg)==70 and all(r['valid'] for r in gg),'Invalid preregistered initial state; no replacement'
        old=batch(jobs(plan['missing_old_jobs'],'oldmissing'),'old_missing.jsonl')
        dev=batch(jobs(plan['development_jobs'],'dev6'),'development.jsonl')
        safe=lambda r:r['complete'] and r['success'] and not r['collision']
        if len(old)==51 and all(map(safe,old)) and len(dev)==54 and all(map(safe,dev)):
            held=batch(jobs(plan['heldout_jobs'],'fresh'),'heldout.jsonl')
            if len(held)==16 and all(map(safe,held)):
                # Reconstruct a fresh Actor from the immutable saved checkpoint.
                q=policy(plan).eval().requires_grad_(False);assert S.digest(q.state_dict())==plan['policy_hash'];state=S.state_np(q)
                refs={r['source_reset_id']:r for r in readrows(plan['reused_old_rows'])+old+dev+held}
                vr=batch(jobs(plan['development_jobs']+plan['heldout_jobs']+plan['old_centers'],'verify'),'verification.jsonl')
                for r in vr:
                    reference=refs[r['source_reset_id']]
                    assert r['array_hashes']==reference['array_hashes'] and r['integration_sha256']==reference['integration_sha256']
                gate=len(vr)==78 and all(map(safe,vr))
                if gate:batch(jobs(plan['next_distance_jobs'],'frozen7'),'next_distance.jsonl')
    except Stop as e:reason=str(e)
    except BaseException:
        reason='FAILED';C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physics_steps=physics,commands=commands,inflight_physics_upper_bound=inflight))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except OSError:pass
        for w in processes:
            w.join(3)
            if w.is_alive():w.kill();w.join(3)
    safe=lambda r:r['complete'] and r['success'] and not r['collision']
    raw={n:readrows(OUT/n) if (OUT/n).exists() else [] for n in ['old_missing.jsonl','development.jsonl','heldout.jsonl','verification.jsonl','next_distance.jsonl']}
    score=lambda rr:dict(n=len(rr),safe=sum(map(safe,rr)),contact=sum(r['collision'] for r in rr),timeout=sum(r['timeout'] for r in rr),censored=sum(not r['complete'] for r in rr))
    result=dict(state='FAILED' if reason=='FAILED' else 'COMPLETE',stop_reason=reason,
        physics_steps=physics,commands=commands,optimizer_calls=0,new_navigation_training_episodes=0,
        reused_old_224=224,reused_not_independent_physical_repetition=True,
        old_missing=score(raw['old_missing.jsonl']),old_combined_n=224+len(raw['old_missing.jsonl']),old_combined_safe=224+sum(map(safe,raw['old_missing.jsonl'])),
        six_development=score(raw['development.jsonl']),six_by_side={side:score([r for r in raw['development.jsonl'] if r['job']['case']['side']==side]) for side in ['LEFT','RIGHT']},
        heldout=score(raw['heldout.jsonl']),reload=score(raw['verification.jsonl']),next_distance=raw['next_distance.jsonl'],six_m_gate=gate,
        all_workers_stopped=all(not w.is_alive() for w in processes),elapsed_s=time.time()-start,
        inflight_physics_upper_bound=inflight,checkpoint_sha256=sha(plan['checkpoint']),policy_hash=plan['policy_hash'],
        own_bytes=S.used(OUT),global_bytes=S.used(ROOT/'stage/runs'),automatic_next_training=False)
    C.save(OUT/'summary.json',result);C.save(OUT/'status.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
