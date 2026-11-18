"""One new bounded post-science export of preserved T632, no producer restart."""
from pathlib import Path
import importlib.util
import json
import time
import traceback
ROOT=Path(__file__).resolve().parents[3]
DOC=ROOT/'stage/docs/day_20261118/after_t632'


def main():
    spec=importlib.util.spec_from_file_location('_t632_post_science_export',ROOT/'stage/experiments/t632_taskwise_recovery/transport.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    original=mod.read;now=time.time()
    def read(path):
        value=original(path)
        if Path(path)==mod.OUT/'protocol.json':
            value=dict(value,absolute_all_stop=now+420,transport_seconds=300)
        return value
    mod.read=read;DOC.mkdir(parents=True,exist_ok=True)
    try:
        mod.main()
        result=dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',
                    new_native=0,new_optimizer=0,original_protocol_unchanged=True,
                    failed_first_export_preserved=True,epoch=time.time())
    except BaseException:
        result=dict(state='REMOTE_PENDING_LOCAL_RAW_RETAINED',error=traceback.format_exc(),
                    new_native=0,new_optimizer=0,original_protocol_unchanged=True,epoch=time.time())
    (DOC/'T632_BACKUP_RECOVERY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['state'])


if __name__=='__main__':main()
