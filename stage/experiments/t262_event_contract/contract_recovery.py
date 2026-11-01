"""Correct native flag ordering before the unused prototype; no simulation."""
import json
from pathlib import Path
import contract as original


raw_targets = original.targets


def native_targets(counts, flags, cap, complete, horizons):
    # Native storage is [contact, goal, timeout]; original prototype reversed 0/1.
    flags = original.np.asarray(flags)
    return raw_targets(counts, flags[:, [1, 0, 2]], cap, complete, horizons)


if __name__ == '__main__':
    plan = json.loads(Path(__file__).with_name('recovery_protocol.json').read_text())
    assert original.sha(Path(__file__).read_bytes()) == plan['adapter_sha256']
    assert original.sha((original.HERE/'contract.py').read_bytes()) == plan['original_sha256']
    # Check against independent metadata for all actual native records.
    base = json.loads((original.HERE/'protocol.json').read_text())
    lines = Path(base['source_rows']).read_bytes().splitlines(keepends=True)[:48]
    for line in lines:
        row = json.loads(line)
        with original.np.load(row['path']) as z:
            flags = list(map(bool, z['end_flags'][-1]))
        assert flags == [row['collision'], row['success'], row['timeout']]
    original.targets = native_targets
    original.OUT = Path(plan['output'])
    original.OUT.mkdir()
    original.main()
