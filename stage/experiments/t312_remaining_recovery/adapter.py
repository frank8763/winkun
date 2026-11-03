from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t312_event_contract'))
from event_contract import event_row

def append_event(path,calls,arm,replica,kind,payload):
    row=event_row(arm,replica,kind,payload)
    calls.append(row)
    with Path(path).open('a') as f:f.write(json.dumps(row,separators=(',',':'))+'\n')
