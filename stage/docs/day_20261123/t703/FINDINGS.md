# T703 fixed projected/per-center-radius comparison

State: COMPLETE. T698 remains retained. T702 remains completed; no original task restarted.

| Block | Audit | Success | Core407 | Lost403 | Sources |
|---|---|---:|---:|---:|---:|
| known_proj8 | PASS | 410/431 | 396 | 0 | 7 |
| new_proj8 | PASS | 45/78 | 0 | None | 0 |

Historical T696:396core/0oldloss/47local; T701quant32:396/0/45 at1320120B added; refined108:387/11/53 at161584B added.
Historical T702:396core/0oldloss/42local at150744B added. The new candidate freezes that exact8D projection/centers/coefficients and replaces one radius by1737 local radii storedfloat32/evaluatedfloat64. Added157684B, unchanged2621976B common parent. Actual tensor/latency/RSS and gates are in fit metadata.
This tests the combined per-center-radius rule and its storage precision. No seed/dimension/threshold search. Projected supports extend along the nullspace; finite-history command preservation gives no neighborhood or unknown-map guarantee.
All missing/failed audits remain explicit. Known500 saves full69+compact431; local85 saves full78+compact7. Independent audit checks actual saved payloads; it does not reintegrate every trajectory. No blind-test, capacity lower-bound, BC impossibility, RL or novelty claim.
All original models, failures and local raw retained; remote coverage is exactly the verified PRIMARY_BACKUP receipt.
