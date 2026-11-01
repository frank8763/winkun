"""One registered paired frozen exploration probe; no optimizer or promotion."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from t271_runtime import *
import io,json,math,multiprocessing as mp,os,shutil,subprocess,time,traceback
OUT=ROOT/'stage/runs/t271_frozen_feature_exploration'


def main():
    assert not (OUT/'status.json').exists(),'No duplicate/restart or exploration grid'
    plan=json.loads((HERE/'protocol.json').read_text());start=time.time();deadline=start+plan['wall_seconds']
    physics=commands=inflight=0;workers=[];pipes=[];reason='COMPLETE'
    def rows(name):
        f=OUT/name;return list(map(json.loads,f.read_text().splitlines())) if f.exists() else []
    def status(label):C.save(OUT/'status.json',dict(state=label,pid=os.getpid(),physics_steps=physics,commands=commands,optimizer_calls=0,updated_epoch=time.time(),elapsed_s=time.time()-start))
    def append(name,row):
        text=json.dumps(row)+'\n';assert len(text.encode())<=plan['record_bytes']
        with (OUT/name).open('a') as f:f.write(text)
    def guard(n,steps):
        assert time.time()<deadline,'WALL'
        assert commands+n<=plan['max_commands'] and physics+steps<=plan['max_physics'],'COUNTS'
        assert S.used(OUT)<plan['final_bytes'] and S.used(ROOT/'stage/runs')<plan['global_bytes'],'STORAGE'
        assert shutil.disk_usage(ROOT).free>=plan['min_free_bytes']
    def receive(pipe):
        while not pipe.poll(.5):
            assert time.time()<deadline,'RESPONSE_WALL'
        item=pipe.recv();assert 'error' not in item,item;return item
    def batch(jobs,name,replay=False,geometry=False):
        nonlocal physics,commands,inflight
        result=[]
        for off in range(0,len(jobs),4):
            jj=jobs[off:off+4];mult=2 if replay else 1
            guard(len(jj)*mult,0 if geometry else sum(j['case']['step_cap'] for j in jj)*mult)
            for pipe,j in zip(pipes,jj):pipe.send(dict(j,geometry_only=True) if geometry else j)
            inflight=0 if geometry else sum(j['case']['step_cap'] for j in jj)
            first=[receive(pipe) for pipe in pipes[:len(jj)]];commands+=len(first)
            if not geometry:physics+=sum(i['row']['physics_steps'] for i in first)
            inflight=0
            for j,item in zip(jj,first):
                if geometry:row=item
                else:
                    row=item['row'];a=item['arrays'];full=j['mode']=='feature_copula'
                    kept=a if full else {k:v for k,v in a.items() if k not in ['states','next_states']}
                    buffer=io.BytesIO();np.savez_compressed(buffer,**kept);data=buffer.getvalue()
                    assert len(data)<=plan['full_episode_bytes' if full else 'compact_episode_bytes'],'Registered payload bound exceeded'
                    path=OUT/'episodes'/(j['id']+'.npz');path.parent.mkdir(exist_ok=True);assert not path.exists();path.write_bytes(data)
                    row.update(path=str(path),sha256=C.sha(path),full120_states_saved=full)
                append(name,row);result.append(item)
            if replay:
                for pipe,j in zip(pipes,jj):pipe.send(j)
                inflight=sum(j['case']['step_cap'] for j in jj)
                again=[receive(pipe) for pipe in pipes[:len(jj)]];commands+=len(again);physics+=sum(i['row']['physics_steps'] for i in again);inflight=0
                for a,b in zip(first,again):
                    assert a['row']['array_hashes']==b['row']['array_hashes'] and a['row']['integration_sha256']==b['row']['integration_sha256']
                    assert all(np.array_equal(v,b['arrays'][k]) for k,v in a['arrays'].items())
                    append('replays.jsonl',b['row'])
            status(name)
        return result
    try:
        for path,h in plan['input_hashes'].items():assert C.sha(path)==h,path
        assert json.loads((OUT/'preflight.json').read_text())['state']=='PASS'
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);status('STARTING')
        ctx=mp.get_context('spawn')
        for k in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(k+1);w=ctx.Process(target=worker,args=(b,k,str(OUT),plan,deadline));w.start();b.close();workers.append(w);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';assert all(receive(p)['ready'] for p in pipes)
        C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[w.pid for w in workers],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
        gg=batch(plan['geometry_jobs'],'geometry.jsonl',geometry=True);assert len(gg)==4 and all(r['valid'] for r in gg)
        cc=batch(plan['control_jobs'],'controls.jsonl')
        for item in cc:
            row=item['row'];ref=plan['control_references'][row['source_reset_id']]
            assert row['array_hashes']==ref['array_hashes'] and row['integration_sha256']==ref['integration_sha256']
        # Interleaved matched source-reset/seed pairs; no adaptive task/arm selection.
        batch(plan['probe_jobs'],'exploration.jsonl',replay=True)
    except BaseException:
        reason='FAILED';C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physics_steps=physics,commands=commands,inflight_physics_upper_bound=inflight))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except OSError:pass
        for w in workers:
            w.join(3)
            if w.is_alive():w.kill();w.join(3)
    rr=rows('exploration.jsonl');safe=lambda r:r['complete'] and r['success'] and not r['collision']
    score=lambda rs:dict(n=len(rs),safe=sum(map(safe,rs)),contact=sum(r['collision'] for r in rs),timeout=sum(r['timeout'] for r in rs),censored=sum(not r['complete'] for r in rs))
    result=dict(state=reason,physics_steps=physics,commands=commands,optimizer_calls=0,new_navigation_RL_updates=0,
        unique_stochastic_records=len(rr),exact_replays=len(rows('replays.jsonl')),
        by_arm={arm:score([r for r in rr if r['mode']==arm]) for arm in ['conditional_AR','feature_copula']},
        by_side_arm={arm:{side:score([r for r in rr if r['mode']==arm and r['job']['case']['side']==side]) for side in ['LEFT','RIGHT']} for arm in ['conditional_AR','feature_copula']},
        controls=len(rows('controls.jsonl')),all_workers_stopped=all(not w.is_alive() for w in workers),
        inflight_physics_upper_bound=inflight,elapsed_s=time.time()-start,own_bytes=S.used(OUT),global_bytes=S.used(ROOT/'stage/runs'),
        actor_hash_unchanged=plan['policy_hash'],six_m_deterministic_gate=False,automatic_next_training=False,
        interpretation='Frozen exploration feasibility, not learned navigation, independent training seeds, statistical superiority, or six-metre stability.')
    C.save(OUT/'summary.json',result);C.save(OUT/'status.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
