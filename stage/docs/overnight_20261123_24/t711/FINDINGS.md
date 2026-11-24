# T711 frozen prospective initial-state comparison

State: COMPLETE. New learning0. Retained T698 unchanged.

Planned96 initial poses/model; valid96, invalid0, duplicate0; no replacements.

| Model | Audit | Safe | Collision | Timeout | Boundary(overlap) |
|---|---|---:|---:|---:|---:|
| parent | PASS | 32/96 | 40 | 24 | 9 |
| compact | PASS | 32/96 | 40 | 24 | 9 |
| B0 | PASS | 39/96 | 40 | 17 | 12 |
| constrained | PASS | 37/96 | 40 | 19 | 12 |
| coverage | PASS | 38/96 | 40 | 18 | 12 |
| whole_nominal | MISSING | None/None | 0 | 0 | 0 |
| whole_induced | MISSING | None/None | 0 | 0 | 0 |

Historical old403 losses / seven-source success (complete prior audits only):
- compact: oldlost0,source77,knownauditPASS.
- B0: oldlost11,source77,knownauditPASS.
- constrained: oldlost12,source76,knownauditPASS.
- coverage: oldlost17,source76,knownauditPASS.
- whole_nominal: oldlostNone,source7None,knownauditMISSING.
- whole_induced: oldlostNone,source7None,knownauditMISSING.
- parent: oldlost0,source70,knownaudithistorical composite.

Predeclared paired gains/losses:
- vscoverage_vs_constrained: gains2, losses1, common safe36.
- vscompact_vs_parent: gains0, losses0, common safe32.

Complete per-parent/ring/sector/shared-source-snapshot tables and actual IDs are in PAIRED_RESULTS.json. Related starts are not independent environments; no binomial or pseudo-independent bootstrap significance claim.
Primary comparison fixed before outcomes: whole_induced versus whole_nominal, and both versus originalB0. T705constrained/T709/compact/parent remain declared comparators even if weak. Fresh96 no parameters/models selected from outcomes; no sampled-population or new-map/algorithm novelty claim. Data become exposed after this test; no automatic parameter rescue or nextjob.
Report historical known431/old403/source7 outcomes alongside fresh96; all prior audits reused by exactmodelSHA, not retested here. Addedbytes parent0,compact157684, allMLP161584 plus2621976 common parent. No independent maps or trainingseed replication.
Each valid fresh trajectory saved full; chosen exact repeats compact. Independent saved-payload/owncarry/integral/initial/exact audit, not reintegration of every trajectory. Missing/failed checks remain missing/failed.
All local raw retained. Named primary evidence remote coverage only if PRIMARY_BACKUP passes; remaining full raw localpending.
