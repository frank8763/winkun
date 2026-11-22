"""Sequential source300 with explicit T699 reservation/preparation/process guard."""
import copy,signal,sys,time
from t699_common import *
sys.path.insert(0,str(ROOT/'stage/experiments/weekend_20261120_23'))
import archive_after_t695 as Old
ORIGINAL_STOPPED,ORIGINAL_READ=Old.stopped,Old.read

def stopped():
    # Existing guard owns all previous jobs. This adapter owns T699 explicitly,
    # including registered-but-never-launched, instead of inventing a SEALED file.
    def previous_only(path):
        v=ORIGINAL_READ(path)
        if Path(path)==WEEKEND/'LEDGER.json':
            v=copy.deepcopy(v);v['jobs']=[j for j in v['jobs'] if j['id']!='T699']
        return v
    try:
        Old.read=previous_only;ORIGINAL_STOPPED()
    finally:Old.read=ORIGINAL_READ
    pids=set()
    for marker in [DOC/'PREPARATION_STARTED.json',OUT/'DRIVER_STARTED.json',DOC/'AFFINITY_GUARD.json']:
        if marker.exists():pids.add(read(marker)['pid'])
    for unit in ['stage-t699','stage-t699-preparation','stage-t699-preparation-recovery']:
        s=subprocess.check_output(['systemctl','--user','show',unit+'.service','-p','ActiveState','-p','MainPID'],text=True)
        assert 'MainPID=0' in s and ('ActiveState=inactive' in s or 'ActiveState=failed' in s),(unit,s)
    if (OUT/'launch.json').exists():
        s=read(OUT/'SEALED.json');assert s['all_scientific_processes_stopped'] and s['helper_stopped'];pids.update(s['scientific_pids'])
    else:
        assert read(DOC/'REGISTERED.json')['state']=='REGISTERED_NOT_LAUNCHED'
        assert not (OUT/'DRIVER_STARTED.json').exists() and not (OUT/'SEALED.json').exists()
    assert all(not Path('/proc',str(pid)).exists() for pid in pids)

def main():
    started=time.time();stopped()
    p=read(OUT/'protocol.json');l=read(WEEKEND/'LEDGER.json')
    assert l['transport_seconds']+600<18000 and time.time()+300<1795395600
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T699 source300 original bound')));signal.alarm(295)
    Old.stopped=stopped;Old.Client.client()
    sys.path.insert(0,str(ROOT/'stage/experiments/t626_multitask_closed_loop'))
    import t626_sync_source
    t626_sync_source.main();stopped();signal.alarm(0)
    # Accounting is done by the caller after actual shell exit and tree receipt.
    print(json.dumps(dict(state='SOURCE_SHELL_COMPLETE',elapsed_seconds=time.time()-started)))

if __name__=='__main__':main()
