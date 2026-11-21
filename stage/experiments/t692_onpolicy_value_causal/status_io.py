"""Atomic state publication and bounded reads at the resource helper seam."""
from pathlib import Path
import json
import os
import time


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.' + str(os.getpid()) + '.partial')
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    partial.replace(path)


def read_status(path, attempts=10, sleep=time.sleep):
    path = Path(path)
    for attempt in range(attempts):
        try:
            return json.loads(path.read_text()) if path.exists() else {}
        except json.JSONDecodeError:
            if attempt + 1 == attempts:
                raise
            sleep(.01)
