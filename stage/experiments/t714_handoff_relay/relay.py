"""Process/JSON-only handoff supervision; never fit, simulate, restart or sync."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,time,traceback
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'stage/runs/t714_success_data'
DOC=ROOT/'stage/docs/day_20261124/t714'
DEST=ROOT/'stage/docs/day_20261124/t714_handoff'
UNITS=['stage-t714.service','stage-t714-finalizer.service']
def read(p):return json.loads(p.read_text()) if p.exists() else {}
def save(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n');os.replace(tmp,p)
def units():
    result={}
    for unit in UNITS:
        x=subprocess.check_output(['systemctl','--user','show',unit,'-p','MainPID','-p','ActiveState'],text=True)
        result[unit]=dict(line.split('=',1) for line in x.splitlines() if '=' in line)
    return result
def stopped(state):return all(v.get('MainPID')=='0' and v.get('ActiveState') in ['inactive','failed'] for v in state.values())
def snapshot(p,last_progress):
    state=units();status=read(OUT/'status.json');results={}
    for suite,arm in p['schedule']:
        folder=OUT/(suite+'_'+arm);s=read(folder/'summary.json');a=read(folder/'audit.json')
        results[folder.name]=dict(state=s.get('state','PENDING'),audit=a.get('state','PENDING'),safe=s.get('success'),total=s.get('total'),old403_lost=len(s['lost403']) if s.get('lost403') is not None else None,collisions=len(s['collision_ids']) if s.get('collision_ids') is not None else None)
    return dict(epoch=time.time(),units=state,status=status,results=results,final_report=read(OUT/'FINAL_REPORT.json'),finalizer_done=read(DOC/'FINALIZER_DONE.json'),source=read(DOC/'FINAL_SOURCE.json'),primary_backup=read(DOC/'PRIMARY_BACKUP.json'),backup_deferred=read(DOC/'BACKUP_DEFERRED.json'),science_failure=read(OUT/'failure.json'),finalizer_failure=read(DOC/'FINALIZER_FAILURE.json'),last_observed_progress=last_progress,absolute_all_stop=p['absolute_all_stop'],no_model_forward=True,no_physics=True,no_optimizer=True,no_restart=True,no_automatic_next_science=True)
def report(s):
    local=datetime.datetime.fromtimestamp(s['epoch'],ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
    lines=['# T714 evening handoff','',f'Last checked {local} Beijing.', '',f"Science: {s['final_report'].get('state','ACTIVE')}; finalizer: {s['finalizer_done'].get('state','ACTIVE_OR_PENDING')}.",'','| Block | Science | Audit | Safe | Total | Old403 lost | Collisions |','|---|---|---|---:|---:|---:|---:|']
    for name,v in s['results'].items():lines.append('| '+' | '.join(str(x) for x in [name,v['state'],v['audit'],v['safe'],v['total'],v['old403_lost'],v['collisions']])+' |')
    lines+=['','Full paired conclusions: ../t714/PAIRED_RESULTS.json and ../t714/FINDINGS.md. Results without completed audits are provisional; missing is not zero.', '', 'T698 remains retained. No old run was restarted and no additional experiment was automatically selected. All model/raw/failure evidence is preserved. Today21:00 Beijing is the final stop.', '', 'Source receipt: '+json.dumps(s['source'],ensure_ascii=False), 'Backup receipt: '+json.dumps(s['primary_backup'] or s['backup_deferred'],ensure_ascii=False)]
    if s.get('alerts'):lines+=['','Alerts: '+json.dumps(s['alerts'],ensure_ascii=False)]
    (DEST/'HANDOFF.md').write_text('\n'.join(lines)+'\n')
def main():
    DEST.mkdir(exist_ok=True);p=read(OUT/'protocol.json');assert p and (OUT/'launch.json').exists()
    assert not (DEST/'STARTED.json').exists(),'Relay must not duplicate'
    save(DEST/'STARTED.json',dict(pid=os.getpid(),epoch=time.time(),units=UNITS,deadline=p['absolute_all_stop'],process_metadata_and_JSON_only=True))
    last=time.time();signature=None
    while True:
        now=time.time();s=snapshot(p,last);status=s['status']
        current=json.dumps([status,s['results'],s['finalizer_done'],s['source']],sort_keys=True)
        if current!=signature:last=now;signature=current
        s['last_observed_progress']=last;s['alerts']=[]
        if s['science_failure']:s['alerts'].append('Science failed; preserve all evidence; do not restart.')
        if s['finalizer_failure']:s['alerts'].append('Finalizer failed; science status is independent; preserve evidence; no automatic retry.')
        # Independent fits/audits may legitimately have no status updates until
        # their bounded action completes. Flag only after that action limit.
        limit=status.get('absolute_action_stop')
        if limit and now>limit+30 and not s['final_report']:s['alerts'].append('Action past recorded deadline; inspect registered cgroup, no deadline extension.')
        if now>=p['absolute_all_stop']:
            for unit,value in s['units'].items():
                if value.get('MainPID')!='0':subprocess.run(['systemctl','--user','stop',unit],check=False,timeout=15)
            s=snapshot(p,last);s['alerts']=['Final absolute stop reached; only registered T714 units were stopped.']
            save(DEST/'STATUS.json',s);report(s);save(DEST/'DONE.json',dict(state='ABSOLUTE_STOP',epoch=time.time(),units=s['units']));return
        save(DEST/'STATUS.json',s);report(s)
        if stopped(s['units']):
            save(DEST/'DONE.json',dict(state='ALL_REGISTERED_T714_UNITS_EXITED',epoch=time.time(),science=s['final_report'].get('state','MISSING'),finalizer_done=bool(s['finalizer_done']),failure=bool(s['science_failure'] or s['finalizer_failure']),no_next_science=True));return
        time.sleep(min(30,max(1,p['absolute_all_stop']-time.time())))
if __name__=='__main__':
    try:main()
    except BaseException:
        DEST.mkdir(exist_ok=True);save(DEST/'RELAY_FAILURE.json',dict(epoch=time.time(),error=traceback.format_exc(),no_scientific_mutation=True));raise
