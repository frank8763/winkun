"""One network-independent, bounded Stage supervisor for 2026-11-07.

Only immutable registered mechanisms run here. Scientific rejection is a normal
result; shared numerical/evidence failures stop the chain without a retry.
"""
from pathlib import Path
from zoneinfo import ZoneInfo
import copy
import fcntl
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t410_day_offline_chain'
DOC=ROOT/'stage/docs/day_20261107'
ACTIVE=None


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(str(path)+'.partial')
    with temporary.open('x') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def used(directory):
    return sum(path.stat().st_size for path in Path(directory).rglob('*') if path.is_file())


def plan():
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    return read(OUT/'protocol.json')


def deadlines(p):
    tz=ZoneInfo('Asia/Shanghai')
    canonical={name:datetime(2026,11,7,hour,minute,tzinfo=tz).timestamp()
               for name,hour,minute in [('learning',18,45),('native',19,30),('all',20,0)]}
    selected={
        'learning':p.get('learning_deadline',p.get('learning_stop_epoch',canonical['learning'])),
        'native':p.get('native_deadline',p.get('native_stop_epoch',canonical['native'])),
        'all':p.get('all_work_deadline',p.get('all_stop_epoch',canonical['all']))}
    assert all(selected[key]<=canonical[key] for key in canonical)
    assert selected['learning']<selected['native']<selected['all']
    return selected


def offline():
    def prohibited(*args,**kwargs):
        raise RuntimeError('The offline Stage supervisor cannot use the network')
    socket.create_connection=prohibited
    socket.socket.connect=prohibited
    socket.socket.connect_ex=prohibited


def verify_inputs(p):
    mapping=p.get('inputs',p.get('input_hashes',{}))
    for file,digest in mapping.items():
        assert sha(file)==digest,('Registered input changed',file)


def stop():
    global ACTIVE
    if ACTIVE is not None and ACTIVE.poll() is None:
        try:
            os.killpg(ACTIVE.pid,signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            ACTIVE.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(ACTIVE.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            ACTIVE.wait(timeout=2)


def call(script,args,seconds,log):
    global ACTIVE
    assert seconds>0 and ACTIVE is None
    with Path(log).open('xb') as stream:
        ACTIVE=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(script),*map(str,args)],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
            env=dict(os.environ,STAGE_PROCESS_CPU='0'),start_new_session=True)
        try:
            code=ACTIVE.wait(timeout=seconds)
            assert code==0,('Registered child failed',str(script),args,code)
        except BaseException:
            stop();raise
        finally:
            ACTIVE=None


def native_stopped(directory):
    pids=set()
    for name in ['main_started.json','workers.json','summary.json']:
        file=directory/name
        if not file.exists():
            continue
        value=read(file)
        pids.update(value[key] for key in ['pid','parent'] if type(value.get(key)) is int)
        pids.update(value.get('workers',[]))
    for pid in pids:
        assert not Path('/proc',str(pid)).exists(),('Live or reused native PID',pid)
    return sorted(pids)


def manifest(directory):
    # Keep the producer/auditor's original local manifest unchanged.
    file=directory/'driver_manifest.json'
    assert not file.exists()
    files=[dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path))
           for path in sorted(directory.rglob('*')) if path.is_file() and path!=file]
    for row in files:
        assert sha(ROOT/row['path'])==row['sha256']
    save(file,dict(state='LOCAL_COMPLETE_MEMBER_SHA_VERIFIED_REMOTE_PENDING',files=files,
                   epoch=time.time(),network_required=False,remote_verified=False))
    return dict(path=str(file),sha256=sha(file),members=len(files),bytes=sum(row['bytes'] for row in files))


def selection_groups(p):
    joint=set(p['joint_ids'])
    if 'selection_groups' in p:
        groups={key:set(p['selection_groups'][key]) for key in ['left','old','right']}
    elif all(key in p for key in ['left5_jobs','old8_jobs','right_job']):
        groups=dict(left={job['id'] for job in p['left5_jobs']},old={job['id'] for job in p['old8_jobs']},right={p['right_job']['id']})
    else:
        old={name for name in joint if name.startswith('dev_task')}
        right={name for name in joint if 'RIGHT' in name}
        groups=dict(left=joint-old-right,old=old,right=right)
    assert [len(groups[key]) for key in ['left','old','right']]==[5,8,1]
    assert set().union(*groups.values())==joint and sum(map(len,groups.values()))==len(joint)==14
    return groups


def checked_selection(summary,p):
    model=summary['selected_model'];assert isinstance(model,dict)
    assert Path(model['path']).is_file() and sha(model['path'])==model['sha256']
    assert isinstance(model['policy_hash'],str) and isinstance(model['hashes'],dict)
    ids=summary['selected_success_ids'];assert len(ids)==len(set(ids))
    groups=selection_groups(p);score=[len(set(ids)&groups[key]) for key in ['left','old','right']]
    assert summary['selection_score']==score
    assert summary['selected_joint_all']==(set(p['joint_ids'])<=set(ids))
    return dict(model=model,success_ids=ids,score=score,joint_all=summary['selected_joint_all'])


def winner(records,allowed):
    eligible=[row for row in records if row['id'] in allowed and row['state']=='COMPLETE_LOCAL_AUDIT_PASS_REMOTE_PENDING']
    if not eligible:
        return None
    # Python max retains the first registered arm on an exact score tie.
    return max(eligible,key=lambda row:tuple(row['selection']['score']))


def driver():
    offline();p=plan();verify_inputs(p);absolute=deadlines(p)
    assert read(OUT/'preflight.json')['state']=='PASS'
    assert (OUT/'launch.json').exists() and not (OUT/'driver_started.json').exists()
    launch_record=read(OUT/'launch.json');start=time.time()
    # Every deadline can become stricter after launch, never later than 20:00.
    absolute['all']=min(absolute['all'],launch_record['epoch']+p.get('all_seconds',30000))
    assert start+10<absolute['all']
    root_budget=p.get('root_bytes',p.get('final_bytes',41*2**20))
    assert root_budget<=41*2**20
    limits=p.get('limits',p.get('global_limits',dict(physics=50_000_000,commands=2600,main_wall_s=20000)))
    actual=copy.deepcopy(p.get('prior_actual',dict(physics=0,commands=0,main_wall_s=0.)))
    actual.setdefault('physics',0);actual.setdefault('commands',0);actual.setdefault('main_wall_s',0.)
    save(OUT/'driver_started.json',dict(pid=os.getpid(),parent=os.getppid(),epoch=start,deadlines=absolute,network_required=False))
    def interrupted(signum,frame):
        raise TimeoutError('Offline Stage absolute stop '+str(signum))
    for signum in [signal.SIGTERM,signal.SIGINT,signal.SIGALRM]:
        signal.signal(signum,interrupted)
    signal.alarm(max(1,int(absolute['all']-start-10)))
    ledger=dict(state='RUNNING_OFFLINE_PREPROGRAMMED_CHAIN',pid=os.getpid(),start_epoch=start,
                deadlines=absolute,network_required=False,remote_backups_pending=True,limits=limits,
                actual=actual,completed=[],current=None,seven_eight_entered=False)
    selected=[];normal=False;state='STOPPED_WITH_ERROR_OR_CUTOFF'
    def publish():
        ledger['updated_epoch']=time.time()
        save(OUT/'day_ledger.json',ledger)
        save(DOC/'DAY_LEDGER.json',ledger)
        save(DOC/'DAY_STATUS.json',dict(state=ledger['state'],current=ledger['current'],updated_epoch=time.time(),pid=os.getpid(),remote_pending=True))
    def capacity():
        total=used(ROOT/'stage/runs');own=used(OUT);outside=total-own
        assert own<=root_budget and outside+root_budget<=2**30
        assert total<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
        return total,own,outside
    try:
        capacity();publish()
        specs=p['stages'];assert len({spec['id'] for spec in specs})==len(specs)
        assert len({spec['folder'] for spec in specs})==len(specs)
        for spec in specs:
            now=time.time();kind=spec.get('kind',spec['id']);validation=kind in ['validation','V'] or spec['id']=='V'
            main_seconds=spec.get('main_seconds',4500);audit_seconds=spec.get('audit_seconds',300 if validation else 900)
            learning_seconds=spec.get('learning_seconds',0 if validation else max(0,main_seconds-900))
            record=dict(id=spec['id'],folder=spec['folder'],kind=kind)
            if (now+main_seconds+audit_seconds+120>absolute['all'] or now+main_seconds>absolute['native'] or
                    (not validation and now+learning_seconds>absolute['learning'])):
                ledger['completed'].append(dict(record,state='NOT_STARTED_INSUFFICIENT_FULL_WALL_RESERVE'))
                publish();continue
            if any(actual[key]+reserve>limits[key] for key,reserve in [('physics',spec['max_physics']),('commands',spec['max_commands']),('main_wall_s',main_seconds)]):
                ledger['completed'].append(dict(record,state='NOT_STARTED_GLOBAL_BUDGET_RESERVE'));publish();continue
            capacity();verify_inputs(p)
            model=dict(path=p['initial_model'],sha256=sha(p['initial_model']))
            source_arm='FIXED_REGISTERED_PARENT'
            if spec['id']=='C' or kind in ['online','C']:
                chosen=winner(selected,['A','B'])
                if chosen is not None:
                    model=copy.deepcopy(chosen['selection']['model']);source_arm=chosen['id']
            if validation:
                chosen=winner(selected,['A','B','C'])
                # One final full diagnostic is useful even if the cheap joint
                # gate is incomplete. The validation producer alone controls
                # fresh16/reload78 after the same candidate's54+275 allGOAL.
                # No fresh outcome participates in selecting this model.
                if chosen is not None:
                    model=copy.deepcopy(chosen['selection']['model']);source_arm=chosen['id']
            assert sha(model['path'])==model['sha256']
            directory=OUT/spec['folder'];assert directory.parent==OUT and not directory.exists()
            directory.mkdir();(directory/'episodes').mkdir();(directory/'blocks').mkdir()
            runtime=dict(spec,initial_model=model['path'],initial_model_sha256=model['sha256'],
                         learning_stop_epoch=min(now+learning_seconds,absolute['learning']),
                         native_stop_epoch=min(now+main_seconds,absolute['native']),
                         own_bytes=spec['final_bytes'],out=str(directory),id=spec['id'],
                         root_protocol_sha256=sha(OUT/'protocol.json'),selected_source_arm=source_arm,
                         root_launch_sha256=sha(OUT/'launch.json'),network_required=False,
                         fresh_Adam_at_start=not validation,validation_only=validation)
            save(directory/'runtime.json',runtime)
            save(directory/'launch.json',dict(epoch=now,parent_driver_pid=os.getpid(),source_arm=source_arm,
                  initial_model=model['path'],initial_model_sha256=model['sha256'],runtime_sha256=sha(directory/'runtime.json'),
                  source_local_commit=launch_record['source_local_commit'],source_remote_commit=launch_record.get('source_remote_commit'),offline=True))
            ledger['current']=dict(record,directory=str(directory),phase='NATIVE',source_arm=source_arm,
                                   reserved=dict(physics=spec['max_physics'],commands=spec['max_commands'],main_seconds=main_seconds,audit_seconds=audit_seconds,bytes=spec['final_bytes']))
            publish()
            try:
                call(HERE/'run.py',[str(directory)],min(main_seconds+2,runtime['native_stop_epoch']-time.time()+2,absolute['all']-time.time()-10),directory/'main.log')
            except BaseException:
                save(directory/'entry_or_timeout_error.json',dict(error=traceback.format_exc(),restart=False,epoch=time.time()))
                raise
            summary=read(directory/'summary.json');native_stopped(directory)
            assert summary.get('all_workers_stopped',summary.get('allworkersstopped')) is True
            actual['physics']+=summary['physics'];actual['commands']+=summary['commands'];actual['main_wall_s']+=summary['elapsed_s']
            assert all(actual[key]<=limits[key] for key in ['physics','commands','main_wall_s'])
            capacity()
            if summary['state']!='COMPLETE':
                manifest(directory)
                raise RuntimeError('Shared scientific/evidence contract failed; no restart or new numbered retry')
            ledger['current']['phase']='AUDIT';publish()
            try:
                call(HERE/'audit.py',[str(directory)],min(audit_seconds,absolute['all']-time.time()-10),directory/'audit_console.log')
            except BaseException:
                save(directory/'audit_error.json',dict(error=traceback.format_exc(),native_restarted=False,epoch=time.time()))
                manifest(directory);raise
            audit=read(directory/'audit.json');assert audit['state']=='PASS';native_stopped(directory)
            local=manifest(directory);selection=checked_selection(summary,p)
            completed=dict(record,state='COMPLETE_LOCAL_AUDIT_PASS_REMOTE_PENDING',
                           directory=str(directory),summary_sha256=sha(directory/'summary.json'),
                           audit_sha256=sha(directory/'audit.json'),manifest=local,
                           physics=summary['physics'],commands=summary['commands'],elapsed_s=summary['elapsed_s'],
                           selection=selection,initial_model_source=source_arm,
                           stable_six=bool(summary.get('stable_wide_six',summary.get('stable_six',False))))
            ledger['completed'].append(completed);selected.append(completed);ledger['current']=None;publish()
        normal=True;state='COMPLETE_PREPROGRAMMED_SEQUENCE'
    except BaseException:
        stop();save(DOC/'DAY_ERROR.json',dict(error=traceback.format_exc(),epoch=time.time(),current=ledger['current'],training_restarted=False))
    finally:
        stop();signal.alarm(0)
        all_absent=True
        for completed in ledger['completed']:
            if 'directory' in completed:
                try:native_stopped(Path(completed['directory']))
                except BaseException:all_absent=False
        if ledger['current']:
            try:native_stopped(Path(ledger['current']['directory']))
            except BaseException:all_absent=False
        if not all_absent:state='STOPPED_NATIVE_EXIT_NOT_VERIFIED'
        ledger.update(state=state,finished_epoch=time.time(),all_owned_children_stopped=all_absent,
                      global_runs_bytes=used(ROOT/'stage/runs'),root_bytes=used(OUT),free_bytes=shutil.disk_usage(ROOT).free,
                      finite_sequence_completed=normal,remote_backups_pending=True,
                      early_finish_reason='All finite registered mechanisms completed or explicit admission gates stopped them; no infinite repetition to fill clock' if normal else None)
        publish();save(DOC/'FINAL_REPORT.json',ledger)
        lines=['# Stage offline day result','',f"State: {state}.",
               '','All reports are local. Remote archive verification is PENDING; no network service was needed.',
               'No seven/eight-metre or full-map training was implemented or executed.','']
        for row in ledger['completed']:
            lines.append(f"- {row['id']}: {row['state']}"+(f"; selection score {row['selection']['score']}; physics {row['physics']}; commands {row['commands']}" if 'selection' in row else ''))
        lines.extend(['','Original failures, models and partial evidence remain. Complete old275/wide54/fresh16/reload78 are required before six-metre acceptance.',''])
        (DOC/'FINAL_REPORT.md').write_text('\n'.join(lines))
        print(json.dumps(dict(state=state,completed=len(ledger['completed']),network_required=False,remote_pending=True)),flush=True)


def launch():
    p=plan();verify_inputs(p);absolute=deadlines(p)
    assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['state']=='REMOTE_HEAD_AND_FULL_ROOT_TREE_VERIFIED' and receipt['root_tree']==tree
    assert used(ROOT/'stage/runs')-used(OUT)+p.get('root_bytes',p.get('final_bytes',41*2**20))<=2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30
    epoch=time.time();seconds=min(int(absolute['all']-epoch)-3,p.get('all_seconds',30000))
    assert seconds>5400
    with (DOC/'DAY_CONSOLE.log').open('xb') as log:
        proc=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-offline-day',
            '--why=Bounded preprogrammed Stage offline mechanisms until20:00',
            'timeout','--signal=TERM','--kill-after=2s',str(seconds)+'s','taskset','-c','0',
            str(ROOT/'run_stage.sh'),str(Path(__file__)),'driver'],cwd=ROOT,
            env=dict(os.environ,STAGE_PROCESS_CPU='0'),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=proc.pid,epoch=epoch,source_local_commit=local,source_root_tree=tree,
                source_remote_commit=receipt['remote_commit'],protocol_sha256=sha(OUT/'protocol.json'),
                absolute_stop_epoch=absolute['all'],external_timeout_seconds=seconds,network_required=False,
                hard_stop_Beijing='2026-11-07T20:00:00+08:00')
    save(OUT/'launch.json',record);save(DOC/'DAY_LAUNCH.json',record)
    print(json.dumps(dict(state='ACTUALLY_LAUNCHED_OFFLINE_DAY_CHAIN',pid=proc.pid,deadline_epoch=absolute['all'],network_required=False)),flush=True)


if __name__=='__main__':
    assert sys.argv[1:] in (['launch'],['driver'])
    globals()[sys.argv[1]]()
