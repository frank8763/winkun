"""One finite overnight relay. No recurring schedule or unbounded retries."""
from pathlib import Path
import json,os,signal,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[3]
DOC=ROOT/'stage/docs/overnight_20261123_24'
PY=ROOT/'run_stage.sh'


def read(p):return json.loads(Path(p).read_text())
def save(p,value):
    p=Path(p);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(p)


def inactive(unit):
    p=subprocess.run(['systemctl','--user','show',unit,'-p','MainPID','-p','ActiveState'],capture_output=True,text=True)
    return 'MainPID=0' in p.stdout and ('ActiveState=inactive' in p.stdout or 'ActiveState=failed' in p.stdout)


def wait_finished(code,folder,deadline):
    while time.time()<deadline:
        if inactive(f'stage-{code}.service') and inactive(f'stage-{code}-finalizer.service'):
            if (folder/'FINALIZER_DONE.json').exists() or (folder/'FINALIZER_FAILURE.json').exists():return
        time.sleep(5)
    raise TimeoutError(('Earlier root/relay deadline exhausted',code))


def command(args,log,seconds):
    # All scientific subprocesses are launched in their separately registered
    # cgroups. Preparation/sync are children of this bounded Stage relay only.
    with Path(log).open('xb') as f:
        child=subprocess.Popen(args,cwd=ROOT,stdout=f,stderr=f,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        try:return child.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise


def finalizer(code,h,d,absolute):
    assert not (d/'FINALIZER_STARTED.json').exists()
    cmd=['systemd-run','--user',f'--unit=stage-{code}-finalizer','--collect','--property=Type=exec','--property=AllowedCPUs=15','--property=CPUQuota=100%','--property=Nice=19','--property=MemoryMax=4G','--property=MemorySwapMax=0','--property=KillMode=control-group','--property=TimeoutStopSec=5',f'--property=RuntimeMaxSec={int(absolute-time.time()-2)}',f'--property=WorkingDirectory={ROOT}',f'--property=StandardOutput=append:{d}/finalizer.log','--property=StandardError=inherit','--setenv=CUDA_VISIBLE_DEVICES=',str(PY),str(h/'finalize.py'),'watch']
    c=subprocess.run(cmd,capture_output=True,text=True);assert c.returncode==0,c.stderr


def charge_preparation(code,elapsed,passed):
    ledger=ROOT/'stage/docs/weekend_20261120_23/LEDGER.json';l=read(ledger);key=code.upper()+'_PREPARATION'
    assert not any(j.get('id')==key for j in l['new_monday_experiments'])
    l['science_seconds']+=elapsed;assert l['science_seconds']<=180000
    l['new_monday_experiments'].append(dict(id=key,state='PASS' if passed else 'FAILED',science_seconds=elapsed,new_actual_model_learning=0,new_native=0))
    save(ledger,l)


def run_stage(code,name):
    auth=read(DOC/'AUTHORIZATION.json');h=ROOT/'stage/experiments'/name;d=DOC/code;out=ROOT/'stage/runs'/name
    assert not (d/'RELAY_STAGE_ENTERED.json').exists();save(d/'RELAY_STAGE_ENTERED.json',dict(epoch=time.time(),single_entry=True))
    # No job is admitted unless its complete worst path fits the remaining night.
    reserve=18800 if code=='t710' else 7*(1400+300+60)+1200
    assert time.time()+reserve<auth['absolute_all_stop']
    assert time.time()+(2400 if code=='t710' else 0)+30<auth['absolute_learning_stop']
    save(d/'PREPARATION_BOUND.json',dict(start=time.time(),absolute_stop=time.time()+180,max_seconds=180,new_actual_learning=0,new_native=0))
    started=time.time();passed=False
    try:
        value=command([str(PY),str(h/'prepare.py')],d/'preflight.log',180);assert value==0,('Preparation failed',code,value)
        assert read(d/'PREFLIGHT.json')['state']=='PASS';passed=True
    finally:charge_preparation(code,time.time()-started,passed)
    assert command([str(PY),str(h/'run.py'),'register'],d/'register.log',120)==0
    assert (out/'protocol.json').exists()
    # Full tracked metadata only. Scientific models/NPZ remain in runs/Release.
    paths=[str(h.relative_to(ROOT)),str(d.relative_to(ROOT)),str((DOC/'PRIMARY_REVIEW.md').relative_to(ROOT)),str((DOC/'PLAN.md').relative_to(ROOT)),str((DOC/'AUTHORIZATION.json').relative_to(ROOT)),str((ROOT/'stage/experiments/overnight_20261123_24').relative_to(ROOT)),'stage/docs/RESEARCH_STATUS.md','stage/docs/weekend_20261120_23/LEDGER.json']
    for prior in [ROOT/'stage/docs/day_20261123/t709',DOC/'t710']:
        for filename in ['TRANSPORT_LEDGER.json','FINAL_SOURCE.json','FINALIZER_DONE.json']:
            file=prior/filename
            if file.exists():paths.append(str(file.relative_to(ROOT)))
    subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True);subprocess.run(['git','commit','-m',f'Register {code.upper()} finite overnight Stage comparison'],cwd=ROOT,check=True)
    target=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    sync_start=time.time();result=command([str(PY),str(h/'run.py'),'sync'],d/'prelaunch_source.log',300)
    if result!=0:
        l=read(d/'TRANSPORT_LEDGER.json');l['charged_seconds']+=300;save(d/'TRANSPORT_LEDGER.json',l)
        raise RuntimeError(('Actual source exit nonzero, no scientific launch',code,result))
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==target
    save(d/'TRANSPORT_PENDING.json',dict(state='ACTUAL_EXIT0_EXACT_TREE_PENDING_FINAL_EXIT_CHARGE',remote_commit=receipt['remote_commit'],root_tree=target,charge_seconds=300,accounted=False,actual_shell_seconds=time.time()-sync_start))
    assert command([str(PY),str(h/'run.py'),'launch'],d/'launch.log',120)==0
    p=read(out/'protocol.json');finalizer(code,h,d,p['absolute_all_stop'])
    save(DOC/'STATE.json',dict(state='ACTIVE',stage=code.upper(),epoch=time.time(),absolute_all_stop=auth['absolute_all_stop'],notification='Only meaningful full results, real failure/stall or budget risk; routine healthy progress silent'))
    wait_finished(code,d,min(p['absolute_all_stop'],auth['absolute_all_stop']))
    return read(out/'FINAL_REPORT.json')


def main():
    a=read(DOC/'AUTHORIZATION.json');assert not (DOC/'RELAY_STARTED.json').exists()
    save(DOC/'RELAY_STARTED.json',dict(pid=os.getpid(),epoch=time.time(),one_shot=True,absolute_stop=a['absolute_all_stop']))
    wait_finished('t709',ROOT/'stage/docs/day_20261123/t709',min(read(ROOT/'stage/runs/t709_induced_replay/protocol.json')['absolute_all_stop'],a['absolute_all_stop']))
    # One complete result is required, not partial progress-based selection.
    prior=read(ROOT/'stage/docs/day_20261123/t709/PAIRED_RESULTS.json')
    save(DOC/'T709_HANDOFF.json',dict(state=prior['state'],results=prior.get('results'),pareto=prior.get('pareto'),epoch=time.time(),next_question='Matched whole-head optimization with unchanged deployment capacity; prior fixed-readout result remains a reference'))
    results={}
    for code,name in [('t710','t710_whole_head_replay'),('t711','t711_fresh_starts')]:
        results[code]=run_stage(code,name)
    save(DOC/'FINAL_REPORT.json',dict(state='COMPLETE',stages={k:v['state'] for k,v in results.items()},epoch=time.time(),retained='T698',all_meaningful_planned_work_finished=True,no_recurring_automation=True,no_more_science=True))
    save(DOC/'STATE.json',dict(state='STOPPED',epoch=time.time(),reason='Finite two-stage program completed, no clock-filling reruns'))


if __name__=='__main__':
    try:main()
    except BaseException:
        save(DOC/'RELAY_FAILURE.json',dict(state='FAILED',error=traceback.format_exc(),epoch=time.time(),no_duplicate_launch_or_automatic_retry=True,all_original_evidence_preserved=True));raise
