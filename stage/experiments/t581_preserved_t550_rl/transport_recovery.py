"""Transport-only repair of the ended T581 archive root. No science restart."""
from pathlib import Path
import hashlib,importlib.util,json,sys
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t581_preserved_t550_rl'
p=json.loads((OUT/'protocol.json').read_text())
original=HERE/'transport.py'
assert hashlib.sha256(original.read_bytes()).hexdigest()==p['inputs'][str(original)]
assert json.loads((OUT/'audit.json').read_text())['state']=='PASS'
assert not any(Path('/proc',str(i)).exists() for i in json.loads((OUT/'SEALED.json').read_text())['scientific_pids'])
spec=importlib.util.spec_from_file_location('_ended_t581_transport_original',original)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.OUT=OUT
assert sys.argv[1]=='T581'
module.main()
