import os,time,fcntl,traceback,json,subprocess,shutil
from pathlib import Path
import t622_learning as M
L,torch=M.L,M.torch

def verify(p):
    assert p['authorization_approved'] and L.sha(M.HERE/'protocol.json')==L.sha(M.OUT/'protocol.json')
    assert time.time()<p['absolute_all_stop'] and set(os.sched_getaffinity(0))<=set(p['stage_cpus'])
    assert sum(f.stat().st_size for f in M.ROOT.joinpath('stage/runs').rglob('*') if f.is_file())<2**30
    assert shutil.disk_usage(M.ROOT).free>=40*2**30
    for f,h in p['inputs'].items():assert L.sha(f)==h,f

def main(arm):
    p=M.config();verify(p);M.setup();folder=M.OUT/arm;train,test=M.data(p);obj=M.make(p);before=M.state(obj);shadow=M.make(p);started=time.time();events=[];last_gpu_check=0.
    def guard():
        nonlocal last_gpu_check
        now=time.time();assert now<min(started+p['learning_seconds'],p['absolute_learning_stop'])
        if now-last_gpu_check>5:
            jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip().splitlines();assert all(int(x.strip())==os.getpid() for x in jobs if x.strip()),'Other GPU compute appeared; yield resources';last_gpu_check=now
    def event(who):
        def emit(stage,k):
            events.append([who,stage,k])
            with (folder/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([who,stage,k])+'\n');f.flush();os.fsync(f.fileno())
            L.save(folder/'status.json',dict(state=who,step=k,stage=stage,epoch=time.time()))
        return emit
    initial=M.evaluate(obj[0],test,arm);u=M.fit(obj,train,p,arm,guard,event('live'));v=M.fit(shadow,train,p,arm,guard,event('shadow'));assert u==v and L.digest(M.state(obj))==L.digest(M.state(shadow))
    final=M.evaluate(obj[0],test,arm);checkpoint=M.write_checkpoint(obj,p,folder/'critic.pt.gz')
    L.save(folder/'summary.json',dict(state='COMPLETE',arm=arm,initial=initial,final=final,trace=u,optimizer_events=events,checkpoint=checkpoint,actor_updates=0,new_native=0,physics=0,commands=0,elapsed_s=time.time()-started,actor_policy_unchanged=True,not_policy_Q_or_gradient_certification=True));print('CRITIC_CALIBRATION_COMPLETE',arm,flush=True)

def audit(arm):
    p=M.config();verify(p);M.setup();folder=M.OUT/arm;s=L.read(folder/'summary.json');train,test=M.data(p);obj=M.make(p);events=[]
    def event(stage,k):
        events.append(['live',stage,k])
        with (folder/'audit_optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([stage,k])+'\n');f.flush()
    u=M.fit(obj,train,p,arm,lambda:None,event);assert u==s['trace'] and L.digest(M.state(obj))==s['checkpoint']['state_sha'];assert L.sha(s['checkpoint']['path'])==s['checkpoint']['sha256'];assert [z for z in s['optimizer_events'] if z[0]=='live']==events
    assert [json.loads(z) for z in (folder/'optimizer_calls.jsonl').read_text().splitlines()]==s['optimizer_events']
    assert M.evaluate(obj[0],test,arm)==s['final'];L.save(folder/'audit.json',dict(state='PASS',independent_critic_Adam_reconstruction=p['updates_per_arm'],GPU_strict_state_and_trace_identity=True,new_native=0,actor_updates=0))

if __name__=='__main__':
    try:globals()[__import__('sys').argv[1]](__import__('sys').argv[2])
    except BaseException:
        L.save(M.OUT/__import__('sys').argv[2]/'failure.json',dict(error=traceback.format_exc(),raw_preserved=True));raise
