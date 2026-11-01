"""JSON scalar adapter only; original fixed48 audit source stays unchanged."""
import json
import types
from pathlib import Path
import audit


def scalar(value):
    if isinstance(value, audit.np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


if __name__ == '__main__':
    plan = json.loads((Path(__file__).with_name('serialization_recovery_protocol.json')).read_text())
    assert audit.sha(Path(__file__).read_bytes()) == plan['adapter_sha256']
    assert audit.sha((audit.HERE/'audit.py').read_bytes()) == plan['original_audit_sha256']
    json.dumps({'numpy_bool': audit.np.bool_(True), 'numpy_integer': audit.np.int64(1)}, default=scalar)
    audit.json = types.SimpleNamespace(loads=json.loads,
        dumps=lambda *args, **kwargs: json.dumps(*args, default=scalar, **kwargs))
    audit.OUT = Path(plan['recovery_output'])
    audit.main()
