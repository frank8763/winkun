"""Frozen real-prefix recovery diagnostic; zero optimizer and no teleport."""
from pathlib import Path
import fcntl,json,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stage/experiments/t502_fullfield_probe'))
import t502_runtime as F
G,P,E,M,torch,np=F.G,F.P,F.E,F.M,F.torch,F.np
read,save,sha,digest=G.read,G.save,G.sha,G.digest
OUT=ROOT/'stage/runs/t512_actual_state_recovery'

def verify(p,full=False):
    root=read(OUT/'protocol.json')
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert M.used(OUT)<=root['max_bytes'] and M.used()-M.used(OUT)+root['max_bytes']<=2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and set(os.sched_getaffinity(0))<={14,15} and not torch.cuda.is_initialized()
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f
F.OUT=OUT;F.HERE=HERE;F.verify=verify
F.R.OUT=OUT;F.R.HERE=HERE;F.R.verify=verify

def frozen(p,path):
    q=F.L.frozen(p,path);original=q.step
    def step(x,hidden=None):
        requested=q.sample
        q.sample=requested and len(q.tape)>=p['recovery_start_decision']
        result=original(x,hidden)
        q.sample=requested
        return result
    q.step=step
    return q

def worker(pipe,index,p,counts):
    G.T.Session=F.FieldSession;G.B.native_setup=F.field_setup;P.frozen=frozen
    G.cpu_worker(pipe,index,p,counts)

class Pool(F.Pool):
    def __init__(self,p,counts,guard):
        self.p=p;self.guard,self.children,self.pipes=guard,[],[]
        ctx=G.R.M.mp.get_context('spawn')
        try:
            for i in range(2):
                a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts))
                c.start();b.close();self.children.append(c);self.pipes.append(a)
            self.ready=[self.receive(i) for i in range(2)]
            assert all(v['state']=='READY' for v in self.ready)
            f=Path(p['out'])/'worker_generations.json'
            previous=read(f) if f.exists() else []
            previous.append(dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time(),CPU=[14,15],GPU=False))
            save(f,previous)
        except BaseException:self.close();raise

PREFIX_KEYS=['states','actions','native_counts','next_states','poses','next_poses','end_flags','rewards','history_poses','history_goals','history_times','hidden_before','locations']
def prefix_contract(p,a,baseline,stochastic):
    k=p['recovery_start_decision'];assert len(a['states'])>k
    for key in PREFIX_KEYS:assert np.array_equal(a[key][:k],baseline[key][:k]),key
    for key in ['states','poses','history_poses','history_goals','history_times','hidden_before']:
        assert np.array_equal(a[key][k],baseline[key][k]),key
    assert not a['ownership'].any() and not a['stochastic'][:k].any()
    assert bool(a['stochastic'][k:].all())==stochastic
    assert int(a['native_counts'][:k].sum())==p['prefix_physics_steps']
    if not stochastic:assert not a['stochastic'].any()

def main(phase):
    assert phase=='probe';folder=OUT/phase;p=F.R.child_plan(folder)
    F.R.setup(folder);verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=G.B.Runtime(p)
    state,error,result='FAILED',None,None
    try:
        rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        control=rt.request([p['probe_job']],p['initial'],'FROZEN_DETERMINISTIC_CONTROL',full=True)
        a,r=control[0];old=p['baseline_row']
        assert r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
        assert not r['success'] and r['timeout'] and not r['collision']
        branches=rt.request([p['probe_job']]*4,p['initial'],'FOUR_FRESH_CONDITIONAL_RECOVERIES',seeds=p['recovery_seeds'],full=True)
        for aa,rr in branches:prefix_contract(p,aa,a,True)
        vals=control+branches;rows=[rr for aa,rr in vals]
        rt.request([p['probe_job']]*5,p['initial'],'EXACT_CONTROL_AND_FOUR_RECOVERIES',seeds=[None]+p['recovery_seeds'],full=True,references=rows)
        details=[F.detail(p,rr,aa) for aa,rr in vals];save(folder/'diagnosis.json',details)
        successful=[rr for (aa,rr),dd in zip(branches,details[1:]) if dd['field_qualified_success']]
        result=dict(control=G.R.M.tally([r]),conditional_recoveries=G.R.M.tally([rr for aa,rr in branches]),qualified_recovery_successes=len(successful),successful_recovery_rows=successful,
                    recovery_start_decision=p['recovery_start_decision'],prefix_seconds=p['prefix_physics_steps']*.002,
                    Actor_Adam=0,Value_Adam=0,new_policy_gradient_updates=0,model=p['initial'],
                    interpretation='Four different suffix innovation streams from one identical actual student prefix. Not four independent full episodes, not new deterministic success, no fullfield24 retest.',
                    next_learning_data_eligible=bool(successful),fullfield24=None,old275=None,six54=None,eight54=None,fresh16=None,reload78=None)
        state='COMPLETE'
    except BaseException:error=traceback.format_exc();save(folder/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(folder/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result))
        fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error

def audit(phase):
    folder=OUT/phase;p=F.R.child_plan(folder);F.R.setup(folder);verify(p,True)
    s=read(folder/'summary.json');assert s['state']=='COMPLETE' and not (folder/'audit.json').exists();G.stopped(folder)
    P.frozen=frozen;rows,rep=G.rows(folder),G.rows(folder,'replays.jsonl');assert len(rows)==len(rep)==5
    baseline=E.N.S.arrays(p['baseline_row']['path']);details=[]
    for i,r in enumerate(rows):
        assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path'])
        assert digest(a)==r['array_digest'] and all(digest(v)==r['array_hashes'][k] for k,v in a.items())
        assert int(a['native_counts'].sum())==r['physics'] and r['loaded_model_sha256']==p['initial']['sha256']
        M.audit_full_history(p,r,a);prefix_contract(p,a,baseline,i>0)
        assert not r['administrative_censor'] and not r['teacher']
        details.append(F.detail(p,r,a))
    assert rows[0]['array_hashes']==p['baseline_row']['array_hashes']
    by={r['path']:r for r in rows}
    for r in rep:
        old=by[r['reference_path']]
        assert r['exact'] and r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
    result=s['result'];assert details==read(folder/'diagnosis.json')
    assert result['conditional_recoveries']==G.R.M.tally(rows[1:])
    assert result['qualified_recovery_successes']==sum(d['field_qualified_success'] for d in details[1:])
    assert s['physics']==sum(r['physics'] for r in rows+rep)<=p['max_physics'] and s['commands']==14
    assert sha(p['initial']['path'])==p['initial']['sha256'] and not (folder/'optimizer_calls.jsonl').exists()
    save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),native=0,Adam=0,full_history_density_reward_hidden_arrays_and_integrations_exact=True,
         actual_prefix_identical_clock_hidden_history_not_reset=True,five_exact_replays_NOT_independent=True,model_unchanged=True,physics=s['physics'],commands=s['commands']))
    print('T512_ACTUAL_STATE_RECOVERY_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])
