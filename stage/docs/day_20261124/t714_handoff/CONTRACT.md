# T714 handoff relay — 2026-11-24

User is leaving until this evening and requests a relay script. Continue the already registered T714 finite panel; never repeat learning/physics or silently create another scientific job. The locked T714 code, inputs and protocol remain unchanged.

Existing stage-t714.service runs the fixed sequential fit/native/audit panel. Existing stage-t714-finalizer.service waits for real science/driver/helper/unit exit, analyses every arm, attempts the bounded named-model primary backup, verifies final source tree and stops. It remains the sole analysis/archive/source writer; this relay never invokes it again.

The independent stage-t714-handoff-relay.service reads only unit/process metadata and existing JSON, writes untracked supervision snapshots under this directory, and exits when both registered units exit. It performs no model import/forward, optimizer, simulation, archive, source transport, new experiment or restart. Every30s it refreshes STATUS.json and HANDOFF.md. Meaningful failures are recorded distinctly; no healthy progress is presented as a capability improvement. At today21:00 Beijing it may stop only the two named registered T714 units, then exits. Existing earlier learning17:00/native20:15/action deadlines remain enforced by the original driver and helper; no extension.

This script intentionally does not modify locked science, retry a failed consumed block, promote a model, or invent an unreviewed next candidate. Full result interpretation is in the original T714 finalizer's FINDINGS.md and PAIRED_RESULTS.json. Evening user can inspect this directory's HANDOFF.md for completion, audits, receipts and failures. Missing audit is provisional/missing, never zero. T698 remains retained. All original raw/models/failures preserved.

Snapshot files stay untracked while the original finalizer may synchronize its exact committed source tree, avoiding competing tracked metadata writes. The relay code and this contract are committed before that transport; final receipt files need no recursive synchronization. No recurring app automation is created.
