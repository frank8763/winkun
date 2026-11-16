"""Namespace-only recovery after the original reader failed before any Adam.

Original run.py, launch, logs and locked T598 files remain unmodified.
"""
from pathlib import Path
import sys,json,hashlib,time,importlib.util
spec=importlib.util.spec_from_file_location('_stage_t599_reader',Path(__file__).with_name('run.py'))
M=importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def register():
    parent=M.OUT;child=parent/'adapter_001';assert not child.exists()
    error=(parent/'recovery.log').read_text()
    assert "has no attribute 'assess'" in error and not (parent/'result.json').exists()
    assert not Path('/proc',str(M.read(parent/'launch.json')['pid'])).exists()
    assert callable(M.R.R.assess)
    p=M.read(parent/'protocol.json')
    p.update(id=p['id']+'_READER_NAMESPACE_ADAPTER_001',
        previous_reader_failure='AttributeError before optimizer construction or historical updates; 0native0Adam, evidence preserved.',
        previous_historical_Adam=0)
    p['inputs'][str(Path(__file__).resolve())]=M.sha(__file__)
    p['inputs'][str(parent/'recovery.log')]=M.sha(parent/'recovery.log')
    child.mkdir();M.save(child/'protocol.json',p)
    M.save(M.DOC/'ADAPTER_001.json',dict(reason=p['previous_reader_failure'],only_change='resolve assess via existing t591 runtime namespace; same data/loops/128auditcap',new_native=0,new_learning=0,unused_historical_audit_Adam=128))
    print('NAMESPACE_ADAPTER_REGISTERED_ZERO_NATIVE_ZERO_ADAM',flush=True)


def main():
    M.OUT=M.OUT/'adapter_001'
    M.R.assess=M.R.R.assess
    M.main()


if __name__=='__main__':globals()[sys.argv[1]]()
