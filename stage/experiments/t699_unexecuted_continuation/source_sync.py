"""Source-only continuation guard extends all existing Stage process checks."""
from pathlib import Path
import copy,json,signal,sys,time,subprocess
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stage/experiments/t699_matched_value'))
import source_sync as Previous
import t699_common as C
sys.path.insert(0,str(HERE));import run as R
ORIGINAL_READ=Previous.Old.read

def stopped():
    def filtered(path):
        value=ORIGINAL_READ(path)
        if Path(path)==C.WEEKEND/'LEDGER.json':
            value=copy.deepcopy(value);value['jobs']=[j for j in value['jobs'] if j['id']!='T699_CONTINUATION']
        return value
    # The old guard's own previous-only adapter calls ORIGINAL_READ directly;
    # filter at that seam, while this adapter owns the new cgroup explicitly.
    old_read=Previous.ORIGINAL_READ
    try:
        Previous.ORIGINAL_READ=filtered;Previous.stopped()
    finally:Previous.ORIGINAL_READ=old_read;Previous.Old.read=ORIGINAL_READ
    for unit in ['stage-t699-continuation','stage-t699-diagnosis','stage-t699-diagnosis-green']:
        s=subprocess.check_output(['systemctl','--user','show',unit+'.service','-p','MainPID','-p','ActiveState'],text=True)
        assert 'MainPID=0' in s and ('ActiveState=inactive' in s or 'ActiveState=failed' in s),(unit,s)
    if (R.OUT/'launch.json').exists():
        seal=C.read(R.OUT/'SEALED.json');assert seal['all_scientific_processes_stopped'] and seal['helper_stopped']
        ids=seal['scientific_pids']+[C.read(R.OUT/'DRIVER_STARTED.json')['pid']]
        assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    else:
        assert C.read(R.DOC/'REGISTERED.json')['state']=='REGISTERED_NOT_LAUNCHED'
        assert not (R.OUT/'DRIVER_STARTED.json').exists() and not (R.OUT/'SEALED.json').exists()

def main():
    stopped();l=C.read(C.WEEKEND/'LEDGER.json')
    assert l['transport_seconds']+600<18000 and time.time()+300<1795395600
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Continuation source300')));signal.alarm(295)
    Previous.Old.stopped=stopped;Previous.Old.Client.client()
    sys.path.insert(0,str(ROOT/'stage/experiments/t626_multitask_closed_loop'))
    import t626_sync_source
    t626_sync_source.main();stopped();signal.alarm(0)

if __name__=='__main__':main()
