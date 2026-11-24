# T714 fixed-capacity behavior/occupancy augmentation

State COMPLETE; complete scheduled panel True. T698 retained.

| Model | Old403 lost | Original7 safe | Exposed safe/78 | Acquired safe/18 | Fresh safe/72 | Fresh collisions/72 |
|---|---:|---:|---:|---:|---:|---:|
| self | 14 | 7 | 48 | 5 | 26 | 29 |
| success | 12 | 6 | 44 | 3 | 26 | 30 |
| B0 | 11 | 7 | 53 | 0 | 28 | 29 |
| nominal | 23 | 6 | 54 | NOT SCHEDULED | 26 | 28 |

Predeclared finite practical gate: {"old_loss_within20": true, "safe_above_B0_and_self": false, "collisions_no_more_than_B0_and_self": false, "acquisition_above_B0": true, "all": false}

All arms, audits, paired gained/lost IDs, per-layout outcomes, collision/timeout/boundary overlap and costs are in PAIRED_RESULTS.json. The fixed head has20198 trainable parameters and161584 added tensor bytes; parent2621976 bytes is shared. Nominal is historical T712 replay50 with a different data/objective mix, not a new identical-objective fit. Source acquisition is training evidence, exposed78 is development; three related perturbed layouts are a limited prospective test with unknown path feasibility. Success-versus-self changes occupancy and behavior together; command canonicalization can make self raw MSE nonzero. No new RL, algorithmic novelty, statistical population claim, automatic promotion or automatic next experiment. Missing is not zero.
