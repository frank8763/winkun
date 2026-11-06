"""Entry-only storage recovery inside the original absolute time budget."""
from pathlib import Path
import importlib.util,json,os,signal,subprocess,sys,time,hashlib
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('stage395_fixed_storage_original',HERE/'storage.py')
S=importlib.util.module_from_spec(sp);sp.loader.exec_module(S)
OLD_OUT=ROOT/'stage/runs/t395_storage_reserve'
OUT=ROOT/'stage/runs/t395_storage_entry_recovery';S.OUT=OUT
DEADLINE=S.read(OLD_OUT/'launch.json')['epoch']+600

def verify(p):
    assert (HERE/'storage_protocol.json').read_bytes()==(OLD_OUT/'protocol.json').read_bytes()
    for f,h in {**p['protected_hashes'],**p['source_hashes']}.items():assert S.sha(Path(f).read_bytes())==h,f
    assert S.used()<=1024**3 and S.shutil.disk_usage(ROOT).free>=40*1024**3
S.verify=verify

def prepare():
    assert not OUT.exists();old=S.read(OLD_OUT/'launch.json');assert not Path('/proc',str(old['pid'])).exists()
    assert not (OLD_OUT/'summary.json').exists() and not (OLD_OUT/'encoding_ledger.json').exists()
    p=S.read(HERE/'storage_protocol.json');verify(p)
    for row in p['records']:assert S.sha((ROOT/row['old_record']['encoded_path']).read_bytes())==row['old_record']['encoded_sha256']
    assert time.time()+180<DEADLINE
    OUT.mkdir();S.save(OUT/'protocol.json',dict(id='T395_STORAGE_ENTRY_ONLY_RECOVERY',source_protocol=str(HERE/'storage_protocol.json'),
        source_protocol_SHA=S.sha((HERE/'storage_protocol.json').read_bytes()),absolute_deadline=DEADLINE,
        no_original_file_modified_by_failed_entry=True,original_native=0,original_optimizer=0))
    S.save(ROOT/'stage/docs/review_20261106/T395_STORAGE_ENTRY_FAILURE.json',dict(state='ZERO_MODIFICATION_ENTRY_FAILURE',
        old_launch=old,error=(OLD_OUT/'main.log').read_text(),original_files_SHA_all_unchanged=True,
        recovery='Unique entry wrapper with original600s absolute mutation deadline; no changed codec/files/budget'))

def launch():
    assert not (OUT/'launch.json').exists();verify(S.read(HERE/'storage_protocol.json'))
    receipt=S.read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True,cwd=ROOT).strip()
    remaining=int(DEADLINE-time.time())-2;assert remaining>120
    with (OUT/'main.log').open('xb') as log:
        q=subprocess.Popen(['timeout','--signal=TERM','--kill-after=1',str(remaining),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'storage_recovery.py'),'main'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    S.save(OUT/'launch.json',dict(pid=q.pid,epoch=time.time(),absolute_deadline=DEADLINE,source_remote=receipt['remote_commit']))
    print(q.pid)

def main():
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Original storage absolute deadline')))
    signal.alarm(max(1,int(DEADLINE-time.time())-2));S.main()

def audit():S.audit()

if __name__=='__main__':globals()[sys.argv[1]]()
