from coverage_support import *
import multiprocessing as mp,shutil,fcntl
class Stop(Exception):pass
def main():
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'status.json').exists(),'No duplicate T225'
    plan=json.loads((HERE/'protocol.json').read_text());start=time.monotonic();deadline=start+plan['wall_seconds'];wall=time.time()+plan['wall_seconds'];physics=commands=replays=0;workers=[];pipes=[];rows=[];reason=None;recipe_updates=0;export=None
    def status(state):C.save(OUT/'status.json',dict(state=state,physics_steps=physics,commands=commands,episodes=len(rows),replays=replays,navigation_updates=0,recipe_reconstruction_updates=recipe_updates,elapsed_s=time.monotonic()-start))
    def guard(n=0,count=0,pending=0):
        if time.monotonic()+5>=deadline:raise Stop('WALL_RESERVE')
        if commands+n>plan['max_commands'] or physics+count>plan['max_physics']:raise Stop('COUNT_RESERVE')
        if used(OUT)+pending>=plan['final_bytes'] or used(ROOT/'stage/runs')+pending>=plan['global_bytes'] or shutil.disk_usage(ROOT).free<40*1024**3:raise Stop('STORAGE_RESERVE')
    def receive(pipe):
        while not pipe.poll(.5):
            if time.monotonic()>deadline:raise Stop('RESPONSE_DEADLINE')
        r=pipe.recv();assert 'error' not in r,r;return r
    try:
        for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);status('RECONSTRUCTING_CANDIDATE')
        candidate=L.reconstruct();recipe_updates=3000;assert digest(candidate.state_dict())==plan['candidate_hash']
        parent=S.policy(torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['policy']);policies={a:state_np(p) for a,p in [('parent',parent),('candidate',candidate)]};hashes={a:digest(p) for a,p in policies.items()}
        export=S.save_gzip(OUT/'T221_reconstructed_Actor.pt.gz',dict(policy=candidate.state_dict(),source='T221exactrecipe',content_hash=plan['candidate_hash']),lambda n:guard(pending=n))
        C.save(OUT/'contracts.json',dict(state='PASS',reconstructed_policy_hash=hashes['candidate'],recipe_updates=3000,physics_steps=0,navigation_updates=0,export=export))
        ctx=mp.get_context('spawn')
        for i in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(i+1);w=ctx.Process(target=worker,args=(b,i,wall));w.start();b.close();pipes.append(a);workers.append(w)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(p) for p in pipes];assert all(r.get('ready') for r in ready);C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[w.pid for w in workers]))
        for off in range(0,len(plan['jobs']),4):
            batch=plan['jobs'][off:off+4];guard(len(batch)*2,sum(j['case']['step_cap'] for j in batch)*2,len(batch)*256*1024+16384)
            for pipe,j in zip(pipes,batch):pipe.send(dict(j,policy_np=policies[j['arm']],policy_hash=hashes[j['arm']]))
            first=[receive(p) for p in pipes[:len(batch)]];physics+=sum(r['row']['physics_steps'] for r in first);commands+=len(first)
            # Store every success and failure before checking independent regeneration.
            for job,item in zip(batch,first):
                arrays=item['arrays'];path=OUT/(job['id']+'.npz');assert not path.exists();np.savez_compressed(path,**arrays);assert path.stat().st_size<256*1024
                row={**item['row'],'path':str(path),'sha256':C.sha(path),'job':job,'policy_hash':hashes[job['arm']]};rows.append(row)
                with (OUT/'episodes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            for pipe,j in zip(pipes,batch):pipe.send(dict(j,policy_np=policies[j['arm']],policy_hash=hashes[j['arm']]))
            second=[receive(p) for p in pipes[:len(batch)]];physics+=sum(r['row']['physics_steps'] for r in second);commands+=len(second)
            for a,b in zip(first,second):
                assert a['row']==b['row'] and all(np.array_equal(v,b['arrays'][k]) for k,v in a['arrays'].items());replays+=1
                with (OUT/'replays.jsonl').open('a') as f:f.write(json.dumps(b['row'])+'\n')
            status('PAIRED_FROZEN_COVERAGE')
        reason='BUDGET_COMPLETE'
    except Stop as e:reason=str(e)
    except BaseException:reason='FAILED';C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physics_steps=physics,commands=commands))
    finally:
        for p in pipes:
            try:p.send(None)
            except OSError:pass
        for w in workers:
            w.join(3)
            if w.is_alive():w.kill();w.join(3)
        assert all(not w.is_alive() for w in workers)
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    safe=lambda r:r['complete'] and r['success'] and not r['collision'];scores={}
    for arm in ['parent','candidate']:
        for reset in ['LEFT_06','LEFT_21']:
            rr=[r for r in rows if r['arm']==arm and r['id'].startswith(reset)];scores[arm+'_'+reset]=dict(n=len(rr),safeGOAL=sum(safe(r) for r in rr),CONTACT=sum(r['collision'] for r in rr))
    gate=reason=='BUDGET_COMPLETE' and replays==len(rows) and all(any(safe(r) and r['id'].startswith(reset) for r in rows) for reset in ['LEFT_06','LEFT_21']) and sum(safe(r) for r in rows if r['job']['role']=='right_retention_source')==4
    summary=dict(state='FAILED' if reason=='FAILED' else 'COMPLETE',stop_reason=reason,scores=scores,success_source_gate=gate,episodes=len(rows),replays=replays,physics_steps=physics,commands=commands,navigation_updates=0,recipe_reconstruction_updates=recipe_updates,export=export,all_workers_stopped=True,automatic_promotion=False,stable_5m=False,elapsed_s=time.monotonic()-start,own_bytes=used(OUT),runs_MiB=used(ROOT/'stage/runs')/1024**2)
    C.save(OUT/'summary.json',summary);C.save(OUT/'status.json',summary);print(json.dumps(summary),flush=True)
    assert physics<=plan['max_physics'] and commands<=plan['max_commands'] and used(OUT)<plan['final_bytes'] and used(ROOT/'stage/runs')<plan['global_bytes']
if __name__=='__main__':main()
