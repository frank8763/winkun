"""Frozen native failure localization, then own-policy pooled-credit experiment."""
from pathlib import Path
import sys,copy,json,os,time,fcntl,shutil,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t478_learning as L
B,P,E,M,R,torch,np=L.B,L.P,L.E,L.M,L.R,L.torch,L.np
read,save,sha,digest=L.read,L.save,L.sha,L.digest
OUT=ROOT/'stage/runs/t478_wide8_pooled_credit';ACTIVE=OUT

def verify(p,full=False):
    root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json');assert M.used(OUT)<=root['max_bytes'] and M.used()-M.used(OUT)+root['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f

class Session(B.Session):
    def result(self):
        import mujoco
        r=super().result();pairs=[]
        for i in range(int(self.env.data.ncon)):
            c=self.env.data.contact[i];pairs.append(dict(geom1=mujoco.mj_id2name(self.env.model,mujoco.mjtObj.mjOBJ_GEOM,int(c.geom1)),geom2=mujoco.mj_id2name(self.env.model,mujoco.mjtObj.mjOBJ_GEOM,int(c.geom2)),distance=float(c.dist)))
        r['terminal_contacts']=pairs;return r

def worker(pipe,index,p,counts):B.Session=Session;B.worker(pipe,index,p,counts)
class Pool(B.Pool):
    def __init__(self,p,counts,guard):
        self.guard,self.children,self.pipes=guard,[],[];ctx=R.M.mp.get_context('spawn')
        try:
            for i in range(2):
                a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
            self.ready=[self.receive(i) for i in range(2)];assert all(v['state']=='READY' for v in self.ready);save(Path(p['out'])/'worker_generations.json',[dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time())])
        except BaseException:self.close();raise

def setup(folder):
    global ACTIVE
    ACTIVE=Path(folder);B.OUT=OUT;B.ACTIVE=ACTIVE;B.verify=verify;M.OUT=ACTIVE;M.verify=verify;M.setup_runtime()
def safe(rows):return sorted(r['job_id'] for r in rows if r['success'] and not r['collision'])
def events(which):
    def f(stage,k):
        with (ACTIVE/'optimizer_calls.jsonl').open('a') as h:h.write(json.dumps([which,stage,k])+'\n')
    return f

def detail(a,r):
    d=np.linalg.norm(a['poses'][:,:2]-a['task_goal_xy'],axis=1);dn=np.linalg.norm(a['next_poses'][:,:2]-a['task_goal_xy'],axis=1);dt=a['native_counts'].astype(float)*.002;speed=np.linalg.norm(a['next_poses'][:,:2]-a['poses'][:,:2],axis=1)/dt;n=min(100,len(d))
    return dict(id=r['job_id'],success=r['success'],collision=r['collision'],timeout=r['timeout'],seconds=float(dt.sum()),remaining_m=r['remaining_m'],min_decision_distance_m=float(min(d.min(),dn.min())),last20s_min_distance_m=float(min(d[-n:].min(),dn[-n:].min())),last20s_path_m=float((speed[-n:]*dt[-n:]).sum()),last20s_mean_speed_m_s=float(np.mean(speed[-n:])),last20s_action0_mean=float(a['actions'][-n:,0].mean()),last20s_action1_mean=float(a['actions'][-n:,1].mean()),terminal_contacts=r['terminal_contacts'])

def main(mode):
    folder=OUT/mode;p=read(folder/'protocol.json');setup(folder);verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=B.Runtime(p);state='FAILED';error=None;rounds=[];selected=p['initial_result'];probe=None;obj=None
    try:
        rt.pool=Pool(dict(p,native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        if mode=='probe':
            values=rt.request(p['probe_jobs'],p['initial'],'FROZEN_FAILURE_LOCALIZATION',full=True)
            for a,r in values:
                cached=p['probe_cached'][r['job_id']];assert r['array_digest']==cached['array_digest'] and r['array_hashes']==cached['array_hashes'] and r['integration_sha256']==cached['integration_sha256'];r['historical_exact_repeat_NOT_independent']=True
            rt.request(p['probe_jobs'],p['initial'],'EXACT_PROBE',full=True,references=[r for a,r in values]);probe=[detail(a,r) for a,r in values];save(folder/'diagnosis.json',probe)
        else:
            ps=read(OUT/'probe/summary.json');assert read(OUT/'probe/audit.json')['state']=='PASS';pr=[json.loads(x) for x in (OUT/'probe/cases.jsonl').read_text().splitlines()];own=[r for r in pr if r['job_id'] in p['probe_success_reference_ids']];assert len(own)==2 and len(safe(own))==2
            refs=L.references(p,own);save(folder/'reference_provenance.json',refs['provenance']);obj=L.make(p);shadow=L.make(p);current=p['initial'];keep=set(p['initial_success_ids']);current8=p['initial_result']['wide8']['success'];rejects=0
            for rnd in range(2):
                rt.guard(True)
                if M.used(OUT)+12*2**20>read(OUT/'protocol.json')['max_bytes']:save(folder/'capacity_stop.json',dict(round=rnd,reason='COMPLETE_NEXT_ROUND_WORST_DATA_MODEL383_AND_METADATA_NOT_AVAILABLE',used=M.used(OUT)));break
                seeds=[[p['seed'],rnd,i] for i in range(4)];samples=rt.request(p['training_jobs'],current,f'TRAIN_{rnd}',seeds=seeds,full=True);rt.request(p['training_jobs'],current,f'EXACT_{rnd}',seeds=seeds,full=True,references=[r for a,r in samples]);before=P.state(*obj)
                aa=L.fit(obj,[a for a,r in samples],p,refs,rnd,lambda:rt.guard(True),events('live'));bb=L.fit(shadow,[a for a,r in samples],p,refs,rnd,lambda:rt.guard(True),events('shadow'));assert aa==bb and digest(P.state(*obj))==digest(P.state(*shadow));proposal=P.export(obj,folder/f'proposal_{rnd}.pt.gz',p['model_bytes'])
                values=rt.request(p['wide8_jobs']+p['six_jobs']+p['old275_jobs'],proposal,f'FULL383_{rnd}');raw=[r for a,r in values];success=set(safe(raw));candidate=dict(model=proposal,success_ids=sorted(success),wide8=R.M.tally(raw[:54]),wide6=R.M.tally(raw[54:108]),old275=R.M.tally(raw[108:]),lost_ids=sorted(keep-success))
                accepted=keep<=success and candidate['wide8']['success']>current8
                if accepted:selected=candidate;current=proposal;keep=success;current8=candidate['wide8']['success'];rejects=0
                else:P.load(obj,before);P.load(shadow,before);assert digest(P.state(*obj))==digest(before)==digest(P.state(*shadow));rejects+=1
                rounds.append(dict(round=rnd,samples=[r for a,r in samples],update=aa,proposal=proposal,candidate=candidate,accepted=accepted,retained=current,rollback_hash=digest(P.state(*obj))));save(folder/'rounds.json',rounds)
                if rejects>=2:break
        state='COMPLETE'
    except BaseException:error=traceback.format_exc();save(folder/'failure.json',dict(error=error,no_restart=True,raw_preserved=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(folder/'summary.json',dict(state=state,pid=os.getpid(),mode=mode,error=error,physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,rounds=rounds,selected=selected,probe=probe,fresh16=None,reload78=None,eight_stability_gate=False));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error

def audit(mode):
    folder=OUT/mode;p=read(folder/'protocol.json');setup(folder);verify(p,True);s=read(folder/'summary.json');assert s['state']=='COMPLETE' and not (folder/'audit.json').exists();ids=[s['pid']]+[v for g in read(folder/'worker_generations.json') for v in [g['parent']]+g['workers']];assert all(not Path('/proc',str(v)).exists() for v in ids)
    rows=[json.loads(x) for x in (folder/'cases.jsonl').read_text().splitlines()];replays=[json.loads(x) for x in (folder/'replays.jsonl').read_text().splitlines()] if (folder/'replays.jsonl').exists() else [];bypath={r['path']:r for r in rows}
    for r in rows:
        assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert all(digest(v)==r['array_hashes'][k] for k,v in a.items()) and int(a['native_counts'].sum())==r['physics'];assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1 and a['end_flags'][-1].tolist()==[int(r['collision']),int(r['success']),int(r['timeout'])]
        if r['full']:M.audit_full_history(p,r,a)
        if mode=='probe':assert detail(a,r)==next(d for d in s['probe'] if d['id']==r['job_id']) and r['array_digest']==p['probe_cached'][r['job_id']]['array_digest']
    for r in replays:
        prev=bypath[r['reference_path']];assert r['array_digest']==prev['array_digest'] and r['array_hashes']==prev['array_hashes'] and r['integration_sha256']==prev['integration_sha256'] and r['exact']
    n=0
    if mode=='learn':
        own=[r for r in map(json.loads,(OUT/'probe/cases.jsonl').read_text().splitlines()) if r['job_id'] in p['probe_success_reference_ids']];refs=L.references(p,own);assert refs['provenance']==read(folder/'reference_provenance.json');obj=L.make(p);keep=set(p['initial_success_ids']);count8=p['initial_result']['wide8']['success'];selected=p['initial_result']
        for row in s['rounds']:
            before=P.state(*obj);result=L.fit(obj,[E.N.S.arrays(r['path']) for r in row['samples']],p,refs,row['round'],lambda:None,lambda *_:None);assert result==row['update'] and digest(P.state(*obj))==row['proposal']['joint_hash'];assert sha(row['proposal']['path'])==row['proposal']['sha256'];n+=2*result['Actor_Adam'];cc=[r for r in rows if r['role']==f"FULL383_{row['round']}"];assert len(cc)==383;success=set(safe(cc));accepted=keep<=success and sum(r['success'] for r in cc[:54])>count8;assert accepted==row['accepted']
            assert row['candidate']['wide8']==R.M.tally(cc[:54]) and row['candidate']['wide6']==R.M.tally(cc[54:108]) and row['candidate']['old275']==R.M.tally(cc[108:])
            if accepted:keep=success;count8=row['candidate']['wide8']['success'];selected=row['candidate']
            else:P.load(obj,before)
            assert digest(P.state(*obj))==row['rollback_hash']
        assert selected==s['selected']
    assert s['physics']==sum(r['physics'] for r in rows+replays) and s['commands']==len(rows)+len(replays)+4
    save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),native=0,historical_discarded_Actor_and_Value_Adam=n,physics=s['physics'],commands=s['commands'],full_actual_training_arrays_density_hidden_rewards_and_joint_Adam_rebuilt=True,all335_success_ID_acceptance_checked=True,compact383_NOT_full120_reconstructed=True,selected=s['selected']))
if __name__=='__main__':
    try:globals()[sys.argv[1]](*sys.argv[2:])
    except BaseException:
        if OUT.exists():save(OUT/f'error_{sys.argv[1]}_{os.getpid()}.json',dict(error=traceback.format_exc(),epoch=time.time()))
        raise
