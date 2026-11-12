"""Supplement immutable zero-native registration/refusal files to raw backup."""
from pathlib import Path
import importlib.util,json,time,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t502_fullfield_probe'
spec=importlib.util.spec_from_file_location('_t502_complete_transport',HERE/'transport.py');T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)
def main():
    T.stopped();A=T.A;A.OLD.install();B=A.OLD.T;A.stopped=T.stopped
    files=sorted(f for f in OUT.rglob('*') if f.is_file() and f.parent!=OUT and OUT/'probe' not in f.parents and f.suffix not in ['.partial','.lock','.pyc'])
    release=B.B.api('GET','releases/tags/stage-eight-only-2026-11-11');r=A.archive('T502_registration_aux',files,release,time.time()+85,B)
    path=ROOT/'stage/docs/long_distance_20261101/release_receipt_T502.json';p=json.loads(path.read_text());previous=path.with_name('release_receipt_T502_scientific_original.json')
    if not previous.exists():shutil.copyfile(path,previous)
    if r['group'] not in p['child_receipts']:
        p['child_receipts'].append(r['group']);p['members']+=r['members'];p['bytes']+=r['bytes'];p['registration_static_refusal_and_preflight_files_also_verified']=True;p['verified_epoch']=time.time();B.B.write(path,p)
    print('T502_REGISTRATION_AUXILIARY_BACKUP_PASS',r['members'],flush=True)
if __name__=='__main__':main()
