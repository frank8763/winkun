"""Lossless references for duplicated, already archived T425 summary objects.

Metadata only: no model imports, simulator, optimizer, network or raw changes.
Original report bytes remain in Git history and are exactly reconstructible.
"""
from pathlib import Path
import hashlib,json,sys,subprocess

ROOT=Path(__file__).resolve().parents[3]
DOC=ROOT/'stage/docs/day_20261108'
OUT=ROOT/'stage/runs/t425_offline_day_chain'
REPORTS=['DAY_LEDGER.json','FINAL_REPORT.json']
KEY='lossless_summary_reference_encoding'

def sha(data):return hashlib.sha256(data).hexdigest()
def encode(value):return (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
def read(path):return json.loads(Path(path).read_bytes())

def restore_bytes(value):
    value=json.loads(json.dumps(value));contract=value.pop(KEY)
    assert contract['version']==1
    for record in value['records']:
        reference=record.get('summary')
        if reference is None:continue
        assert set(reference)=={'$exact_T425_summary_reference','sha256'}
        path=(ROOT/reference['$exact_T425_summary_reference']).resolve()
        assert path.is_relative_to(OUT.resolve()) and path.name=='summary.json'
        data=path.read_bytes();assert sha(data)==reference['sha256']
        record['summary']=json.loads(data)
    data=encode(value)
    assert len(data)==contract['original_bytes'] and sha(data)==contract['original_sha256']
    return data

def compact():
    receipt_path=DOC/'REPORT_ENCODING_RECEIPT.json';assert not receipt_path.exists()
    archive=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T425.json')
    assert archive['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    parent=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    pending=[];records=[]
    for name in REPORTS:
        path=DOC/name;original=path.read_bytes();value=json.loads(original)
        assert KEY not in value and encode(value)==original
        refs=[]
        for record in value['records']:
            summary=record.get('summary')
            if summary is None:continue
            target=Path(record['folder'])/'summary.json'
            assert target.resolve().is_relative_to(OUT.resolve())
            data=target.read_bytes();assert json.loads(data)==summary
            reference=dict(sha256=sha(data))
            reference['$exact_T425_summary_reference']=str(target.relative_to(ROOT))
            record['summary']=reference;refs.append(reference)
        value[KEY]=dict(version=1,original_sha256=sha(original),original_bytes=len(original),
            original_git_commit=parent,original_path=str(path.relative_to(ROOT)),
            verified_raw_archive_sha256=archive['sha256'],raw_summaries_unchanged=True)
        assert restore_bytes(value)==original
        compacted=encode(value);assert len(compacted)<2**20
        pending.append((path,compacted))
        records.append(dict(path=str(path.relative_to(ROOT)),original_sha256=sha(original),
            original_bytes=len(original),compact_sha256=sha(compacted),compact_bytes=len(compacted),
            exact_reconstruction_test=True,references=refs))
    for path,data in pending:path.write_bytes(data)
    receipt=dict(state='PASS_LOSSLESS_METADATA_ENCODING',records=records,
        saved_working_tree_bytes=sum(r['original_bytes']-r['compact_bytes'] for r in records),
        raw_files_changed=0,new_physics=0,new_Adam=0,
        restore='python stage/experiments/t425_report_encoding/codec.py verify; restore_bytes(read(report)) returns exact original bytes without overwriting files')
    receipt_path.write_bytes(encode(receipt));print(json.dumps(receipt,ensure_ascii=False))

def verify():
    receipt=read(DOC/'REPORT_ENCODING_RECEIPT.json')
    for record in receipt['records']:
        path=ROOT/record['path'];assert sha(path.read_bytes())==record['compact_sha256']
        assert sha(restore_bytes(read(path)))==record['original_sha256']
    print('T425_ORIGINAL_REPORT_BYTES_EXACTLY_RECONSTRUCTED_ZERO_NATIVE_ZERO_ADAM')

if __name__=='__main__':
    assert sys.argv[1:] in [['compact'],['verify']]
    (compact if sys.argv[1]=='compact' else verify)()
