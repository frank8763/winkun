# T712 finite development retention/transfer comparison

State COMPLETE; complete panel True. T698 retained.

| Model | Known audit | Old403 lost | Source7 safe | Development audit | Safe/78 | Collision/78 |
|---|---|---:|---:|---|---:|---:|
| plain | PASS | 5 | 7 | PASS | 45 | 9 |
| replay50 | PASS | 23 | 6 | PASS | 54 | 3 |
| replay80 | PASS | 14 | 6 | PASS | 53 | 1 |
| replay95 | PASS | 9 | 5 | PASS | 48 | 6 |
| B0 | PASS | 11 | 7 | PASS | 53 | 1 |
| compact | PASS | 0 | 7 | PASS | 45 | 13 |

Fixed selection under old403 loss limits4/12/20: {"4": {"selected": "compact", "eligible_in_order": ["compact"], "complete_panel": true, "development_selection_not_unbiased_test": true}, "12": {"selected": "B0", "eligible_in_order": ["B0", "replay95", "compact", "plain"], "complete_panel": true, "development_selection_not_unbiased_test": true}, "20": {"selected": "B0", "eligible_in_order": ["B0", "replay80", "replay95", "compact", "plain"], "complete_panel": true, "development_selection_not_unbiased_test": true}}

All points and gained/lost IDs, source failures, collision/timeout/boundary overlap and actual costs are retained in PAIRED_RESULTS.json. Missing/failed are not zero success. Development78 is exposed same-map selection data; no fresh-test, independent-map, seed-replication, optimal-BC or novelty claim. Historical B0 is unrefitted; replay50 is the equal-extra-compute control. No automatic next experiment.
