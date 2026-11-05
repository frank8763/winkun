# T356: full-vehicle local narrow-passage evidence

2026-11-05. Stage only. The original benchmark is preserved.

The active collision footprint is 0.6100008816 m wide, including the tyres; the chassis alone is 0.54 m wide. The original wall–circle bottleneck is 0.5699999779 m. This corrects the earlier chassis-only feasibility argument. It does not prove the curved opening globally impossible for every configuration or route.

A separately identified 0.70 m scene moves only circle9 by +0.1300000221 m. Tyres, vehicle dimensions, collision masks, wheel limit 5, force limit 120, timestep 0.002 s, and goal remain unchanged. In stationary local resets, a constant forward command crossed the complete opening without collision at the centre and at lateral offsets −1 cm and +1 cm (14.618, 14.624, 14.626 s). All four probes, including the original-gap control which collided at 1.238 s, were regenerated with exactly identical arrays and native integration state.

The 322 static configuration samples and three successful local dynamic probes are finite evidence. They are not a global reachability or safety certificate. They do not start at the original six-metre origin, reach its goal, train an Actor, or supply an expert demonstration.

Accounting: 12 commands including four zero-physics geometry environments, 45,106 native steps, 88.681485 s main wall time, zero optimizer calls. Independent audit PASS. All 16 raw files were uploaded in two RAM-only segments, fully downloaded, and verified member by member. Archive SHA256: `ef676ca879834f12c36235eac4c731a7bde2bb4bee70bb03aa32b53b4ee79896`.

Next: test the frozen T353 iid recurrent policy from the actual six-metre origin in the new 0.70 m scene. Report this course separately from the original 0.57 m / 56 s benchmark. A local passage witness cannot establish that the policy has learned the approach or the full route.
