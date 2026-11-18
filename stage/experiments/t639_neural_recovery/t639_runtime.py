"""One locked neural teacher, nominal control and one exact real-state recovery."""
from pathlib import Path
import ast,copy,fcntl,gzip,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t639_neural_recovery';DOC=ROOT/'stage/docs/day_20261118/t639'
sys.path.insert(0,str(ROOT/'stage/experiments/t631_explored_recovery'))
import t631_runtime as Base
N=Base.N;R=Base.R;M=Base.M
read,save,sha,digest,torch,np=M.read,M.save,M.sha,M.digest,M.torch,M.np


def plan():
    p=read(HERE/'protocol.json');config=read(p['actor_configuration']);config.update(p);return config


def verify(p,*args):
    assert p['authorization_approved'] and time.time()<p['absolute_all_stop']
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert R.used(OUT)<=p['max_bytes'] and R.used(ROOT/'stage/runs')-R.used(OUT)+p['max_bytes']+131072<2**30
    assert R.shutil.disk_usage(ROOT).free>=40*2**30
    assert not torch.cuda.is_initialized()
    for f,h in p['inputs'].items():assert sha(f)==h,f


def setup(p):
    Base.HERE=HERE;Base.OUT=OUT;Base.DOC=DOC;Base.verify=verify;Base.setup(p)


class Runtime(R.B.Runtime):
    def request(self,job,role,reference=None):
        self.guard();assert self.commands+1+3<=self.p['max_commands']
        assert sum(self.counts)+job['case']['step_cap']<=self.p['max_physics']
        self.commands+=1;path=OUT/'episodes'/f'{len(self.rows):04d}_{role}.npz'
        assert not path.exists()
        req=dict(model=self.p['reference_model']['path'],job=copy.deepcopy(job),seed=None,query=False,full_path=str(path))
        a,row=self.pool.request([req])[0]
        row=R.B.R.Base.envelope(job,row,a,self.p['field_limit_m'])
        row.update(request=req,role=role,seed=None,path=str(path),sha256=sha(path),full=True,
                   loaded_model_sha256=self.p['reference_model']['sha256'])
        assert path.stat().st_size<=self.p['full_bytes']
        self.rows.append(row);R.B.append('cases.jsonl',row)
        if reference:
            assert row['array_hashes']==reference['array_hashes'] and row['integration_sha256']==reference['integration_sha256']
            assert row['field_detail']==reference['field_detail']
        N.inspect(self.p,[(a,row)])
        return a,row


def main():
    p=plan();verify(p);setup(p)
    assert read(ROOT/'stage/runs/t638_memory_ready_recovery/SEALED.json')['all_scientific_processes_stopped']
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'launch.json').exists();save(OUT/'launch.json',dict(pid=os.getpid(),epoch=time.time()))
    rt=Runtime(p);error=None;result=dict(nominal=None,recovery=None,qualified=False,learning=None,full407=None)
    try:
        rt.pool=R.B.N.Pool(dict(p,out=str(OUT),phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        a,control=rt.request(p['nominal_job'],'nominal');old=p['nominal_reference']
        assert control['array_hashes']==old['array_hashes'] and control['integration_sha256']==old['integration_sha256']
        rt.request(p['nominal_job'],'exact_nominal',control);result['nominal']=control
        assert R.safe(control),'Previously successful teacher nominal control did not reproduce'
        a,recovery=rt.request(p['handoff_job'],'neural_recovery')
        rt.request(p['handoff_job'],'exact_neural_recovery',recovery)
        result.update(recovery=recovery,qualified=R.safe(recovery))
        save(OUT/'RECOVERY_ADMISSION.json',dict(passed=result['qualified'],teacher=p['reference_model'],
             source_T501=p['source_row'],recovery=recovery,not_original_start_student_success=True))
    except BaseException:
        error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=3
        save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,
             physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,result=result,new_optimizer=0))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error


def audit():
    p=plan();verify(p);setup(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE'
    assert not Path('/proc',str(s['pid'])).exists()
    for g in read(OUT/'worker_generations.json'):
        assert all(not Path('/proc',str(i)).exists() for i in [g['parent'],*g['workers']])
    rows=R.B.read_rows('cases.jsonl');assert len(rows)==4 and len({r['path'] for r in rows})==4
    for row in rows:
        assert sha(row['path'])==row['sha256'];a=R.B.R.E.N.S.arrays(row['path'])
        assert all(digest(v)==row['array_hashes'][key] for key,v in a.items());N.inspect(p,[(a,row)])
    for i in [0,2]:
        assert rows[i]['array_hashes']==rows[i+1]['array_hashes'] and rows[i]['integration_sha256']==rows[i+1]['integration_sha256']
    assert rows[0]['array_hashes']==p['nominal_reference']['array_hashes']
    assert s['result']['qualified']==R.safe(rows[2]) and R.safe(rows[0])
    assert s['physics']==sum(r['physics'] for r in rows)<=p['max_physics']
    assert s['commands']==10==len(rows)+6
    assert sha(p['reference_model']['path'])==p['reference_model']['sha256']
    save(OUT/'audit.json',dict(state='PASS',new_optimizer=0,new_native=0,
        all_four_full_history_density_reward_and64plus32plus32_memory_reconstructed=True,
        same_actual_T501_prefix_and_boundary_checked=True,all_unique_files_and_two_exact_replays=True,
        nominal_teacher_matches_original_T629=True,qualified=s['result']['qualified']))


if __name__=='__main__':globals()[sys.argv[1]]()
