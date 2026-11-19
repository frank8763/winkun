"""Finite frozen repair validation under T649 remaining bounds, no new learner."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,os,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t650_reset_contract_repair';DOC=ROOT/'stage/docs/day_20261119/t650'
sys.path.insert(0,str(ROOT/'stage/experiments/t645_iterative_recovery_relay'))
import control as Resource

def read(f):return json.loads(Path(f).read_text())
def save(f,v):Path(f).write_text(json.dumps(v,indent=2)+'\n')
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def prepare():
    assert not OUT.exists();old=ROOT/'stage/runs/t649_fresh_outcome_pg'
    assert read(old/'SEALED.json')['all_scientific_processes_stopped']
    assert all(not Path('/proc',str(i)).exists() for i in read(old/'SEALED.json')['scientific_pids'])
    prior=read(old/'protocol.json');ledger=read(old/'LEDGER.json')
    assert read(DOC/'FACTORY_REGRESSION.json')['state']=='PASS_RED_OLD_GREEN_REPAIRED_FACTORY'
    p=read(ROOT/'stage/experiments/t636_joint_best_reproduction/protocol.json')
    rows=[json.loads(v) for v in (ROOT/'stage/runs/t636_joint_best_reproduction/cases.jsonl').read_text().splitlines()]
    refs=[r for r in rows if r['role']=='fresh'];assert len(refs)==6
    import numpy as np
    jobs=[]
    for row in refs:
        assert sha(row['path'])==row['sha256'];a=np.load(row['path'],allow_pickle=False)
        job=dict(row['request']['job']);job['initial_contract']=dict(scene_sha256=bytes(a['scene_sha256']).hex(),
            initial120_sha256=hashlib.sha256(a['states'][0].tobytes()).hexdigest(),
            integration_sha256=hashlib.sha256(a['initial_native_integration'].tobytes()).hexdigest(),goal_xy=a['task_goal_xy'].tolist())
        jobs.append(job)
    p.update(id='T650_CANONICAL_RESET_REPAIR',out=str(OUT),jobs=jobs,references=refs,authorization_approved=True,
        max_physics=466672,max_commands=18,max_bytes=64*2**20,full_bytes=4*2**20,fresh_bytes=3*2**20,compact_bytes=32768,
        main_seconds=900,audit_seconds=300,backup_seconds=300,source_seconds=180,
        absolute_native_stop=prior['absolute_native_stop'],absolute_all_stop=prior['absolute_all_stop'],
        authorization_scope='User latest check-and-continue; necessary frozen baseline repair within existing T649 remaining physics/commands/time/storage, no new optimizer',
        previous_T649_ledger=ledger,new_Actor=0,new_Critic=0,new_optimizer=0)
    assert ledger['physics']+p['max_physics']<=prior['total_max_physics']
    assert ledger['commands']+p['max_commands']<=prior['total_max_commands']
    assert ledger['main_seconds']+p['main_seconds']<=prior['total_main_seconds']
    old_bytes=sum(f.stat().st_size for f in old.rglob('*') if f.is_file())
    assert old_bytes+p['max_bytes']<prior['max_bytes']
    assert time.time()+900+300+300+180+60<p['absolute_all_stop'] and time.time()+900<p['absolute_native_stop']
    p['inputs'].update({str(f):sha(f) for f in HERE.glob('*.py')})
    p['inputs'].update({str(DOC/'FACTORY_REGRESSION.json'):sha(DOC/'FACTORY_REGRESSION.json'),str(DOC/'FINDINGS.md'):sha(DOC/'FINDINGS.md'),str(old/'LEDGER.json'):sha(old/'LEDGER.json')})
    for row in refs:p['inputs'][row['path']]=row['sha256']
    # All historical locks must remain exact; do not modify a prior producer.
    for f,h in p['inputs'].items():assert sha(f)==h,f
    Resource.resources();OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'BUDGET.json',dict(max_physics=p['max_physics'],max_commands=18,new_optimizer=0,max_bytes=p['max_bytes'],prior_physics=ledger['physics'],combined_physics_upper=ledger['physics']+p['max_physics'],combined_commands_upper=ledger['commands']+18,absolute_all_stop=p['absolute_all_stop']))
    save(DOC/'REGISTERED.json',dict(state='REGISTERED_NOT_LAUNCHED',epoch=time.time(),scope=p['authorization_scope']))
    print('T650_REGISTERED_FROZEN_REPAIR_EXISTING_REMAINING_BOUNDS')


def action(name,seconds,p,children):
    with (OUT/(name+'.log')).open('xb') as log:
        c=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'runtime.py'),name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        children.append(c.pid)
        try:c.wait(timeout=seconds);assert c.returncode==0,(name,c.returncode)
        except BaseException:
            try:os.killpg(c.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:c.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
            raise


def driver():
    p=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert not (OUT/'DRIVER_STARTED.json').exists();receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    Resource.resources();os.sched_setaffinity(0,{14});save(OUT/'DRIVER_STARTED.json',dict(pid=os.getpid(),epoch=time.time(),source_remote=receipt['remote_commit']))
    log=(OUT/'affinity.log').open('xb');helper=subprocess.Popen(['/usr/bin/python3',str(HERE/'affinity_guard.py')],cwd=ROOT,stdout=log,stderr=log)
    children=[];error=None
    try:action('main',p['main_seconds'],p,children);action('audit',p['audit_seconds'],p,children)
    except BaseException:error=traceback.format_exc()
    for f in [OUT/'worker_generations.json']:
        if f.exists():
            for g in read(f):children.extend([g['parent'],*g['workers']])
    children=sorted(set(children));gone=all(not Path('/proc',str(i)).exists() for i in children)
    save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=gone,scientific_pids=children,epoch=time.time()));assert gone
    report=dict(state='FAILED' if error else 'COMPLETE',error=error,summary=read(OUT/'summary.json') if (OUT/'summary.json').exists() else None,
        audit=read(OUT/'audit.json') if (OUT/'audit.json').exists() else None,remote='PENDING_LOCAL_RAW_RETAINED',new_optimizer=0)
    save(OUT/'FINAL_REPORT.json',report)
    if time.time()+p['backup_seconds']+p['source_seconds']+60<p['absolute_all_stop']:
        try:
            c=subprocess.run([str(ROOT/'run_stage.sh'),str(HERE/'transport.py')],cwd=ROOT,capture_output=True,timeout=p['backup_seconds'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
            assert c.returncode==0,c.stderr.decode(errors='replace')[-1000:];report['remote']='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
        except BaseException:report['backup_error']=traceback.format_exc()
    helper.terminate();helper.wait(timeout=3);log.close()
    save(OUT/'FINAL_REPORT_FINAL.json',report);save(DOC/'FINAL_REPORT.json',report)
    status=ROOT/'stage/docs/RESEARCH_STATUS.md';status.write_text('2026-11-19 T650冻结基线初始化修复验证结束：'+report['state']+'，remote='+report['remote']+'，0学习；完整六原轨迹/六exact/初始场景合同见day_20261119/t650/FINAL_REPORT。T637–T649相关旧wide场景保留及鲁棒性比较因初始化工厂绕过存在场景混淆，不能直接称遗忘或算法失败，原结果保留并读FINDINGS勘误。\n\n'+status.read_text())
    try:
        paths=[str(DOC.relative_to(ROOT)),'stage/docs/RESEARCH_STATUS.md']
        paths.extend(str(f.relative_to(ROOT)) for f in (ROOT/'stage/docs/long_distance_20261101').glob('*T650*.json'))
        subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True,capture_output=True)
        subprocess.run(['git','commit','-m','Report canonical reset regression repair and exact frozen trajectories'],cwd=ROOT,check=True,capture_output=True)
        subprocess.run([str(ROOT/'run_stage.sh'),str(ROOT/'stage/experiments/t629_offline_night_relay/t629_transport.py'),'source_sync'],cwd=ROOT,check=True,capture_output=True,timeout=p['source_seconds'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    except BaseException:save(DOC/'SOURCE_PENDING.json',dict(error=traceback.format_exc()))
    print(json.dumps(dict(state=report['state'],remote=report['remote'],error=error)),flush=True)


def launch():
    p=read(OUT/'protocol.json');assert not (OUT/'DRIVER_STARTED.json').exists()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    Resource.resources();left=int(p['absolute_all_stop']-time.time()-2)
    cmd=['systemd-run','--user','--unit=stage-t650','--collect','--property=Type=exec','--property=CPUQuota=400%','--property=CPUWeight=10','--property=Nice=19','--property=AllowedCPUs=8 10 12 14','--property=MemoryMax=4G','--property=MemorySwapMax=0',f'--property=RuntimeMaxSec={left}','--property=KillMode=control-group','--property=TimeoutStopSec=5',f'--property=WorkingDirectory={ROOT}',f'--property=StandardOutput=append:{OUT}/driver.log','--property=StandardError=inherit','--setenv=CUDA_VISIBLE_DEVICES=',
       'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage T650','--why=Frozen canonical reset repair',str(ROOT/'run_stage.sh'),str(HERE/'control.py'),'driver']
    c=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True);assert c.returncode==0,c.stderr;print(c.stderr.strip())

if __name__=='__main__':globals()[sys.argv[1]]()
