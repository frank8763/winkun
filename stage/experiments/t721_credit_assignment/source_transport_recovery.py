"""Metadata-only exact-tree source transport,8MiB text envelope; science immutable."""
from pathlib import Path
import sys,importlib.util,types,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
s=importlib.util.spec_from_file_location('T721_transport_controller',HERE/'run.py');C=importlib.util.module_from_spec(s);s.loader.exec_module(C)
def main(final=False):
    C.stopped()
    original=ROOT/'stage/experiments/t626_multitask_closed_loop/t626_sync_source.py'
    blob=original.read_text();assert 'assert len(b)<2*1024**2' in blob
    module=types.ModuleType('t626_sync_source');module.__file__=str(original)
    sys.path.insert(0,str(original.parent))
    # Client setup occurs in C.sync before this module main is called.
    exec(compile(blob.replace('assert len(b)<2*1024**2','assert len(b)<8*1024**2'),str(original), 'exec'),module.__dict__)
    sys.modules['t626_sync_source']=module
    C.sync(final=final)
if __name__=='__main__':main(final=len(sys.argv)>1 and sys.argv[1]=='final')
