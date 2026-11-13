# T530 — correct at the first persistent trajectory discrepancy

User explicitly requested the next experiment in the authorized 2026-11-13 window ending17:30 BJT. Stage only, Phase fully isolated. No new window or extension. T529 ended with exact native replay and independent audit PASS, zero actual learning: parent neural teacher at30.2s collided38.22s. All raw data and hashes were fully read back from GitHub. Its source control reconstructed the T527 newgoal success exactly, and that full trajectory is reused without simulating the historical control again.

## Hypotheses and deterministic read-only loop

1. **Intervention is late:** if early distribution drift matters, the same frozen T507 teacher applied at the first persistent discrepancy should recover where30.2s failed. Test one fixed earlier time; this does not prove every earlier state recoverable.
2. **Teacher has insufficient recovery coverage:** if it fails even at the preregistered early state, its own-start success is insufficient as a recovery expert; stop this fixed qualification, retain raw failure, zero training.
3. **Learning interference persists:** if teacher recovery succeeds but updated student loses old or new protected IDs, successful labels/replay are insufficient in this fixed optimization; reject the whole Actor/Adam. Do not infer that all architectures are incapable.

Read-only command already run: `CUDA_VISIBLE_DEVICES= taskset -c 14 ./run_stage.sh stage/experiments/t530_early_bidirectional_recovery/divergence.py`. It used actual audited T529 student trajectory and actual T507 old-left successful full history. At equal original physical clocks, first five consecutive decision poses with XY difference>=0.05m OR wrapped heading difference>=2deg begin at index25/5.0s. Freeze takeover one decision earlier: index24/4.8s,2400 native steps. The data-selected threshold is an engineering heuristic frozen before new physics, not a stability/reachability result. At4.8s XY discrepancy0.047273m/yaw0.533348deg; at30.2s XY discrepancy1.828418m. Source arrays/SHA and exact native clock alignment are checked; no native steps or Adam in this computation. The known red signal is T529 full recovery CONTACT, not a software crash. No throwaway unit test can establish navigation recovery; use native GOAL/contact evidence and exact repetition.

## Fixed pilot

Initial full research T527 Actor and both Adam groups unchanged; original accepted T507 old8/8+field11/24 is independent and immutable. Same frozen T507 teacher, teacher's own independently carried history on actual student observations, same actual original goal/clock/actuators, only takeover earlier than T529. One full recovery and exact replay. Actual safe GOAL qualifies labels; a failed short return to an untrained student is not a prerequisite. No teleport, reset, planner or waypoints.

If qualified,16 wholeActor supervised Adam+16 exact shadows, same objective/code/lr asT529: teacher-owned successful old recovery actions plus successful sequence location targets in equal-weight old8/field11/newgoal families. Value/Adam frozen, zero new policy-gradient updates. All20 rehearsal histories are actual successful sources and are verified before preflight; the short recovery-action preflight fixture is synthetic and is not navigation evidence. This is teacher-assisted supervised consolidation of an RL Actor; not a claim of pure new RL, original algorithm, exact DAgger/CLEAR reproduction or safety certificate.

Then autonomous same candidate old8+field24 plus32 exact native repeats. Accept research candidate only if old8all8, original11field IDs and newgoal all retained, field>=12,contact<=9 and no invalid/censor. Else whole Actor+Adam rollback to research initial; original accepted best never overwritten. old275/six54/eight54/fresh16/reload78 remain null. No new7/8 protocols or full-scene claims.

## Bounds and evidence

80 all commands budget includes conservative6 zero-native preflight (actual5),3,247,872 native steps,2400s main/1200 learning/300 independent audit/300 backup. own5MiB/global1GiB/free40GiB; complete worst reserve admission before launch. CPU14/15 aggregate100%=onecore, idle priority, GPUdisabled, actual software renderer asserted. Absolute learning17:00/native17:15/all17:30 or stricter relative deadline. No grid/seeds/extra teacher attempts/restarting old folders. Sole local driver completes audit/manifest/RAM remote all-member SHA readback/source sync after all native workers exit. Original bytes/models/failures preserved; binaries outside ordinaryGit.

## Research checked this round

[DAgger primary paper page](https://proceedings.mlr.press/v15/ross11a.html) and [CLEAR author paper](https://arxiv.org/abs/1811.11682) revisited2026-11-13. DAgger motivates attention to student-induced states; CLEAR motivates retaining old behavior while learning new. Neither establishes that this teacher can recover, that equal-family weights prevent interference, or that allStage tasks can jointly pass. Prior full-source analysis remains in EXISTING_SOLUTION_METHODS.md. Current native evidence must decide the pilot.
