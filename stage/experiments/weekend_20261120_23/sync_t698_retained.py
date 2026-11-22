"""One guarded source snapshot after completed T698; no science or raw deletion."""
from pathlib import Path
import json,signal,sys,time
import archive_after_t695 as Guard
ROOT,DOC=Guard.ROOT,Guard.DOC
def main():
    Guard.stopped()
    auth=Guard.read(DOC/'AUTHORIZATION.json');ledger=Guard.read(DOC/'LEDGER.json')
    assert ledger['transport_seconds']+300<auth['max_transport_seconds']
    assert time.time()+300<auth['absolute_all_stop']
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T698 bounded source300s')))
    signal.alarm(295)
    Guard.Client.client()
    sys.path.insert(0,str(ROOT/'stage/experiments/t626_multitask_closed_loop'))
    import t626_sync_source
    t626_sync_source.main()
    Guard.stopped();signal.alarm(0)
if __name__=='__main__':main()
