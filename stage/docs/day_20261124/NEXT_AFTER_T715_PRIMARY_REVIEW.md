# Next after T715: reuse learned policies, qualify recovery, then integrate

2026-11-24. Research and finalized-metadata review only; **not a registered experiment or launch**. T715 is active during this review. No active T715 result, checkpoint or trajectory was inspected; no model was loaded, no forward pass, native step, optimizer, source edit or experimental output was produced. The only write is this note. Stage only; no Phase research, external planner or waypoint method. Apply the research workflow; this delegated review does not spawn another agent.

## Answer to “have we already learned these tasks?”

**Yes, for many exact fixed starts; no, for their complete union in one fixed-size policy or for general student-state recovery.** The useful correction to a “learn first, integrate later” plan is to reuse existing learned policies rather than train the same skills again:

- T698 is one retained deployed policy with **398/407 core** safe starts and **14/24 supplementary** safe starts: **412 distinct successful starts**, not 412/407. It preserves all prior 410 successful starts. Its explicit compatibility centers/coefficients grew; it is not evidence that the same fixed 161584-byte head can encode that coverage. Its source18 coverage was not measured in the fixed evidence inspected here. [T698 findings](../weekend_20261120_23/t698/FINDINGS.md)
- The **single T703 compact policy already safely solves 14/18 selected source tasks** across the matched historical audits below. The **single T712 replay50 policy safely solves all four remaining tasks** on its audited 13-task block. Thus every one of the 18 exact starts already has an autonomous learned neural policy that solves it. This is a training-teacher inventory, **not an 18/18 deployed policy**. Compact plus replay50 is not a router proposal or a claim that their successes can be concatenated into a single-model score.
- T714 attempted integration using complete successful histories but its final success student solved only **3/18**, with 12 old403 losses; its self-history control solved **5/18**, with 14 losses. On its three new layouts, both scored26/72 versus B028/72. This was not the desired integration breakthrough. T715 changes the supervision objective and remains pending here. [T714 findings](t714/FINDINGS.md), [T715 plan](t715/PLAN.md)

## Audited single-policy coverage on the same source18 IDs

The fixed source selection contains 13 IDs from T712's exposed development78 and five IDs from T711's exposed96. The selected full histories are eight from T712 plain, five from T712 replay50 and five from T703 compact. Selection was one lexicographic full-history producer per eligible ID; it did not establish any producer's full-union performance. [Selection and inventory](t714/SOURCE_SELECTION.json), [source qualification](t714/SOURCE_QUALIFICATION.json)

I read JSON/JSONL-gzip **case metadata only**, fresh rows only, from the fixed T711/T712/T714 roots and the **two exact reused-control paths declared by T712**: T703 `new_proj8` and T701R `new_replay_refined108`. No other source search or model/trajectory-array read. Every included block has audit statePASS. For every counted actual row, I compared the `initial_native_integration`, `scene_sha256` and `task_goal_xy` hashes against that ID's selected B0 baseline: all matched. These checks support the paired initial scene/native/goal identity; the historical independent audits supply observation/history/owncarry/horizon contracts. This review does not independently regenerate physics or re-audit initial120 from raw arrays. Policy hashes must agree across blocks; no model name alone is sufficient.

A=IDs1–13; B=IDs14–18 in the fixed list below. C=collision, T=timeout, Bnd=boundary flag; boundary overlaps timeout and must not be added as another failure. MISSING means no audited row for that policy/block in the fixed evidence, not failure.

| Single frozen policy (policy hash prefix) | A safe / evaluated | B safe / evaluated | Safe / all actually evaluated | C / T / Bnd | Missing of18 |
|---|---:|---:|---:|---:|---:|
| T703 compact `317abfbf8cb2` |9/13|5/5|**14/18**|0 /4 /4|0|
| T712 plain `9e757097518d` |8/13|MISSING|8/13|0 /5 /4|5|
| T712 replay50 `d95f1506eae7` |6/13|MISSING|6/13|2 /5 /2|5|
| T712 replay80 `096575b064b3` |6/13|MISSING|6/13|1 /6 /5|5|
| T712 replay95 `5b10a0a2e8cd` |8/13|MISSING|8/13|0 /5 /5|5|
| T694 parent `11eca188a1c3` |MISSING|5/5|5/5|0 /0 /0|13|
| T705 constrained `07df91dfea42` |MISSING|1/5|1/5|0 /4 /3|13|
| T709 coverage `f4f97cdad86c` |MISSING|1/5|1/5|0 /4 /3|13|
| B0 `7ede24dcda32` |0/13|0/5|0/18|1 /17 /11|0|
| T714 self `45aba2b79527` |4/13|1/5|5/18|0 /13 /9|0|
| T714 success `73914586d0fc` |3/13|0/5|3/18|0 /15 /10|0|

Sources: [T711 new_compact metadata](../../runs/t711_fresh_starts/new_compact/cases.jsonl.gz), [its audit](../../runs/t711_fresh_starts/new_compact/audit.json), [T711 complete comparisons](../overnight_20261123_24/t711/PAIRED_RESULTS.json); [T712 complete comparisons](t712/PAIRED_RESULTS.json), [plain case metadata](../../runs/t712_retention_tradeoff/new_plain/cases.jsonl.gz), [replay50 metadata](../../runs/t712_retention_tradeoff/new_replay50/cases.jsonl.gz); [declared reused T703 compact metadata](../../runs/t703_projected_local/new_proj8/cases.jsonl.gz), [its audit](../../runs/t703_projected_local/new_proj8/audit.json), [declared reused B0 metadata](../../runs/t701r_native_entry/new_replay_refined108/cases.jsonl.gz); [T714 acquisition self](../../runs/t714_success_data/acquisition_self/summary.json), [success](../../runs/t714_success_data/acquisition_success/summary.json), [complete comparison](t714/PAIRED_RESULTS.json). The other T711/T712 rows were read from their corresponding `new_<arm>/cases.jsonl.gz` and audit files, not inferred from the selected successful subset. T714 nominal acquisition was not scheduled; its evaluated A block comes from T712, and its B block remains missing.

Fixed IDs, preserving the selection's order:

|#|Exact job ID|Compact safe?|Replay50 safe on evaluated block?|
|---:|---|---|---|
|1|`T699_new_dev_task6_15_xminus`|yes|no|
|2|`T699_new_dev_task6_15_yawminus`|yes|no|
|3|`T699_new_dev_task6_15_yminus`|yes|no|
|4|`T699_new_hold_task6_03_yplus`|yes|no|
|5|`T699_new_t502_field_sector0_offset1_yawplus`|yes|no|
|6|`T699_new_t502_field_sector0_offset2_xminus`|no|yes|
|7|`T699_new_t502_field_sector0_offset2_xplus`|yes|no|
|8|`T699_new_t502_field_sector0_offset2_yawminus`|no|yes|
|9|`T699_new_t502_field_sector0_offset2_yminus`|no|yes|
|10|`T699_new_t502_field_sector0_offset2_yplus`|no|yes|
|11|`T699_new_t502_field_sector1_offset1_xminus`|yes|no|
|12|`T699_new_t502_field_sector2_offset1_yplus`|yes|yes|
|13|`T699_new_t502_field_sector5_offset1_yawminus`|yes|yes|
|14|`T711_t502_field_sector0_offset1_r04_a22.5`|yes|MISSING|
|15|`T711_t502_field_sector0_offset2_r04_a112.5`|yes|MISSING|
|16|`T711_t502_field_sector0_offset2_r04_a202.5`|yes|MISSING|
|17|`T711_t502_field_sector0_offset2_r04_a292.5`|yes|MISSING|
|18|`T711_t502_field_sector5_offset1_r04_a22.5`|yes|MISSING|

Reuse identities from finalized fit metadata, **not freshly rehashed model bytes**:

|Model|Checkpoint SHA256|Full policy hash|
|---|---|---|
|T703 compact|`19d2b4535d5f7e814e93ce3dfdab788048cf2fc10466c50c1703d29863ffe2b3`|`317abfbf8cb2aa9ac1eca376405c3adbba5d6f5795968285c38161ff2fcb90e3`|
|T712 replay50|`046e65adaa84557d21f2c77b266d8eb17235d9a0792b21a5f264a751f45c774b`|`d95f1506eae7c6548f923ad6bf58e5ca7a83ff73d1ec24e79b6b2d0150006750`|
|B0|`e8bed4b7547a987539563d64e6916b77e8f3ee81688e6447d0d09ee1d1df364d`|`7ede24dcda3207d912ca93c2471d067a562606574512dd93ee1f7c14d805f4a6`|

[T703 fit summary](../../runs/t703_projected_local/fit/summary.json), [T712 fit summary](../../runs/t712_retention_tradeoff/fit/summary.json), [B0 frozen references](t714/REFERENCES.json). Execution must verify full intended checkpoint/source identity before reuse. Do not replace the selected source histories silently; compact's alternative successful histories are a new, explicitly declared training-data choice.

## What the primary methods support, and what they do not

**Progress & Compress (P&C), ICML2018.** Its progress phase learns an active column while the knowledge base is frozen, with lateral knowledge transfer. Compression freezes the learned active policy, trains the knowledge base by policy KL and online EWC, then repeats. The Fisher approximation/previous parameter mean are part of preservation; a small-head BC fit alone is not P&C. It keeps bounded network size across tasks but still uses two components during training; it does not promise exact retention at exhausted capacity. Existing frozen specialists can supply the already-learned side of our engineering analogue; rebuilding an active column merely to recreate their four known successes is unnecessary. Our continuous executed-command targets, replay retention and fixed head would depart from its categorical-policy/Fisher objective. [Original paper, §§2,4, Eq.2](https://proceedings.mlr.press/v80/schwarz18a/schwarz18a.pdf)

**Policy Distillation, ICLR2016.** The original collects teacher-generated observation/Q-value targets, then transfers them to a student. Its multi-game setup uses game-specific output heads selected by game labels. That part does not meet our single-head deployment contract. Continuous clipped-command regression is an adaptation, not its original DQN KL method, and successful teacher paths alone do not establish recovery at student states. [Original paper, §§3.2–3.3,4.1](https://arxiv.org/pdf/1511.06295)

**DAgger, AISTATS2011.** It aggregates expert actions at states visited under successive learner/mixed policies and trains a stationary policy. The distinction is learner occupancy plus a competent expert, not simply “more replay.” Its bounds require the stated no-regret/loss/expert assumptions; no navigation-safety guarantee follows for our neural recurrent actors. A one-batch frozen learner-state collection followed by one fit would be a bounded adaptation, not a full iterative DAgger reproduction. Offline assignment of known neural teachers to fixed training groups is permitted only in training diagnostics; final deployment remains one actor with no ID input or teacher selection. [Original paper, Algorithm3.1 and Theorem3.1](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf)

**Distilling Policy Distillation, AISTATS2019.** It separates the sampling policy from the alignment update: student-generated experience can expose relevant states missed by teacher rollouts. Its analysis also shows common on-policy one-step updates need not be gradients of a single objective, and adding reward can yield non-convergence. Its empirical domains are tabular/random MDPs, not our continuous recurrent vehicle. Therefore collect one bounded dataset, freeze it, use an explicit supervised objective, and avoid claiming a combined RL/distillation theorem. Its Fig.5 also warns that copying an imperfect teacher's wrong decisions more faithfully can lower reward. [Original paper, §§3.1–3.3, Fig.5](https://proceedings.mlr.press/v89/czarnecki19a/czarnecki19a.pdf)

Search/read scope2026-11-24: relevant method/equation/algorithm/limitation sections of these four primary PDFs and the [P&C author's publication page](https://jonathan-schwarz.github.io/). That page links the paper/talk but no P&C implementation. Targeted author/code searches did not verify an original-author implementation of P&C or Distilling Policy Distillation; no claim that none exists, and no third-party notebook was treated as author code. No external code, license-dependent implementation or repository was imported. This is a bounded primary-source review, not an exhaustive novelty search or full-paper/implementation reproduction.

## Why T709 is not the missing recovery experiment

T709 used all102 histories from two pre-existing learners over51 old IDs; the labels were frozen T694 actions evaluated on those histories. Its induced result was382/407 core,17 old403 losses and95/224 joint safe, versus its reconstructed control384/407,15 losses and97/224, and original B0387/407,11 losses and97/224. It added learner-state coverage but did **not** establish that T694 could complete those learner-state continuations. Matching that function can preserve a bad action. [T709 findings](../day_20261123/t709/FINDINGS.md)

T668 provides direct counterevidence to universal expert recovery: T501 took over at the fixed first1cm divergence in four learner-induced states and safely recovered only2/4; the3–4 recovery admission gate failed and the BC experiment correctly did not run. Those conditional successes also did not create an original-start joint policy. This is evidence about that teacher/state set, not proof that compact/replay50 cannot recover. [T668 findings](../weekend_20261120_23/t668/FINDINGS.md)

Thus the new question is not another T709 nominal-function-preservation fit or another T714 success-only replay fit. It is **whether the already successful compact/replay50 policies are qualified correction sources on the frozen student's actual histories**. This must be measured before using those actions as expert correction labels.

## Proposed finite sequence, subject to later registration and admission

**Stage1: no retraining; qualify the existing teachers.** Freeze B0 as the student-state generator, compact as the teacher for its14 verified IDs and replay50 for IDs6/8/9/10. Lock this assignment before any recovery outcome. Reuse audited original-start success evidence and B0 historical full histories after identity verification; do not rerun them to manufacture independent samples. Preserve the exact old403/original7 retention evidence: compact lost0 old403 and kept7/7, whereas replay50 lost23 and kept6/7. Replay50 is a local training specialist, not a wholesale retained replacement. [T712 findings](t712/FINDINGS.md)

One proposed recovery state per ID is the first native decision at which B0's XY differs by at least1cm from that assigned teacher's original-start trajectory at the same native time, strictly before a failure/terminal event. If no admissible point exists, report UNAVAILABLE and fail admission rather than picking another threshold/time. Recompute each teacher's own recurrent carry from the **learner's complete causal observation/history prefix**, then restore the exact native state/goal/time/history and let that teacher autonomously complete within the remaining original horizon. Never transplant foreign hidden states. The collection controller is diagnostic/training only; no hybrid teacher success counts as final navigation capability.

Proposed gate before any fit: all18 fixed original-start teacher successes verified by exact identity; all18 specified recovery continuations safe, complete and independently audited; all18 exact repeats agree; complete provenance and raw histories retained. **Any missing point, invalid reset, failed/censored recovery or audit failure stops this correction-distillation proposal.** Do not turn failed teacher actions into successful supervision, sweep thresholds/teachers/seeds, retry another source or fit the other successful subset. Passing qualifies only these18 recovery conditions, not arbitrary off-trajectory states or final student performance. Learning genuinely absent skills, if later needed, is a separate bounded neural-policy RL study; no traditional navigation teacher is introduced.

A possible standalone gate cap is18 continuations+18 exact repeats=36 scientific native commands. The fixed source caps sum814414 native steps; replaying prefixes and teacher suffixes within each original remaining horizon, twice, is at most**1628828 physical steps**. Count every prefix and repeat, all loader/control commands separately, and lock actual command limits before registration. Candidate wall allocation:900s science+300s independent audit+60s close, with storage, preparation, immutable-source transport and finalizer reserves separately budgeted under the current CPU/memory/storage contracts. These are proposed ceilings, not consumed work or authority to launch.

**Stage2: only after Stage1 passes, fit one final fixed-capacity student versus exact strong B0.** One B0 initialization, one frozen training dataset, one objective/weight specification and one optimization budget. Reuse successful original histories plus the qualified learner-state correction labels and old403/original7 retention data. Candidate architecture remains the same frozen T694 causal120/GRU64 parent plus20198-parameter184→108 tanh→2 residual head,161584 added deployment tensor bytes; freeze all parent tensors. One fixed fit and independent reconstruction, no intermediate-best selection or seed/loss/weight sweep. A proposed80 gradient-closure/64-attempt cap per live/shadow/independent fit matches the existing finite head-study scale, but exact objective, normalization and cap accounting require a fresh locked protocol. Do not add online reward/Q optimization to this test.

Keep B0 unrefitted and reuse its complete audited baseline panels. The final artifact must be one owncarry actor from permitted observations to native clipped-float32 commands, with **no runtime teachers, routing, task ID, model switching, planner or waypoint inputs**. Report teacher-training cost separately from student fit, raw/deployment capacity, fixed buffers, serialized bytes, batch1 latency and RSS. Temporary training teachers do not disappear from compute accounting.

Audit the **same saved final student jointly on the full known431 plus all18 source starts** with the existing exact-repeat contracts, irrespective of forgetting. Also retain the complete exposed78 development regression; compare paired old403 losses, original7, source18 safe/collision/timeout/boundary and grouped costs against B0 and compact's historical matched evidence. A predeclared retention tolerance such as≤20 old403 losses must be fixed before fit, not retrofitted to outcomes. Claim “all covered” only if that one final checkpoint passes **all claimed IDs**; a useful partial gain is not449/449. Source18 and existing layout panels are now exposed training/development evidence. Additional unknown-map claims require a separately fixed prospective world suite; they are not established by this joint known-task test. No hidden old-task screen may suppress later measurements. T698 stays retained unless a separately declared retention decision is actually satisfied.

Changing the trainable GRU at the same time would change recurrence, representation, number of trainable parameters and optimization geometry even if total deployment bytes stayed similar. It would confound the correction-data intervention. Defer it to a separately matched study if this fixed-feature student fails; such failure would not prove the tasks cannot coexist in a sufficiently trained recurrent policy. Conversely T698's larger compatibility basis cannot prove fixed-capacity compression solved. Do not call either engineering adaptation a new algorithm or journal-level breakthrough.

## Timing and admission

Today's unchanged Beijing stops are **learning17:00, native20:15, all work21:00**, with every earlier child/action deadline stricter. “T715 science may finish about16:20” does not release its audit, helper, driver, finalizer, relay, immutable-source or final-source obligations. New work waits until **all of them actually exit**, with exact final-source evidence complete. This review makes no T715 completion claim.

A21-minute recovery gate plus a proposed1200s fit+600s independent fit audit already exceeds the40-minute16:20→17:00 interval before preparation, transport and closing overhead; the final evaluation reserve must fit as well. **Do not admit the two-stage fit today on that schedule.** At most a separately registered standalone zero-learning gate could be considered after every T715 exit if its full worst case fits all existing stops; it creates no automatic delayed fit. Otherwise preserve this reviewed design for a fresh explicit bounded authorization. No reanchoring, overlapping source edits/audits or automatic continuation merely to use remaining evening time.

Reading limits: only required Stage instructions/context/migration/workflow and bounded current research-head entries, T714/T715 plans/reviews, finalized T711/T712/T714 metadata, the two declared reused controls, T698/T709/T668 findings, and the cited primary papers were used. No claim of full Stage/ChatGPT history, Phase evidence, model-byte verification or new experiment result.
