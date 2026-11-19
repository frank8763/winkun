# T652 primary review: discriminate before adding mechanisms

2026-11-19. Research only: zero native steps, zero optimizer steps; this note does not register or launch an experiment. Recommendation: retain T501, first recover the six actual regression trajectories, then consider one matched **adapters-only episodic PG** comparison with valid T651. Earlier comparisons on the bypassed wide/radial factory cannot exclude that option.

## Corrected local evidence

[T650 findings](../t650/FINDINGS.md) establish the initialization confound and its repair; T637–T649 remain real records of their actual scenes, but their cross-scene retention claims are withdrawn. T650's six canonical cases and six exact regenerations matched historical arrays/integration. It did not establish a fresh full407 result.

Counting success IDs in [T651 PG summary](../../../runs/t651_canonical_outcome_pg/pg/summary.json) gives:

| Evaluation | Retained T501 | T651 candidate |
|---|---:|---:|
| old275 | 271 | **266** |
| canonical wide6 | 54 | 54 |
| canonical wide8 | 54 | 54 |
| legacy field24 | 5 | 6 |
| total407 | 384 | 380 |

The six lost IDs are `dev_task3_00`, `dev_task6_00/02/03/14/26`; gains are `dev_task6_15` and `t502_field_sector5_offset1`. Thus 384−6+2=380, and 271−6+1=266 old successes. The [case log](../../../runs/t651_canonical_outcome_pg/pg/cases.jsonl.gz) identifies `dev_task3_00` as **3 m RIGHT**, timing out at 14,000 native steps with 0.207383465 m remaining; the other five are **5 m LEFT collisions**. These are not wide6/wide8 losses.

[Learning records](../../../runs/t651_canonical_outcome_pg/pg/LEARNING.json) show eight accepted full-Actor updates with all384 histories on every update. Final maximum per-task latent Gaussian KL is 0.0004560094; the maximum over **all** steps is 0.001416710, at step 0, below the registered 0.01 cap. Saying every step was below 0.001 would be incorrect. This candidate shows that its replay penalty and sampled KL check did not ensure all-ID deterministic retention. It does **not** identify encoder/GRU damage as the cause. T501 remains retained; the complete candidate is rejected.

## What the primary literature supports

1. [DAgger, Ross et al., AISTATS 2011, §2–3/Algorithm 3.1](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf): actions change subsequent observations, so error on parent trajectories does not directly control error on learner-visited histories. DAgger queries expert actions there. T501 has not been established as a recovery oracle on those histories; collecting its labels would not automatically satisfy the paper's assumptions.
2. [TaSIL, Pfrommer et al., NeurIPS 2022, §3–4](https://arxiv.org/html/2205.14812v2): trajectory closeness follows under explicit incremental input-to-state stability, regularity and sufficiently strong policy matching conditions. Its Proposition 3.1 concerns a neighborhood around the expert trajectory. This explains why nominal replay alone can be insufficient; Stage has not established those assumptions. Derivative matching is a possible later variable, not a justified immediate addition. The [author implementation](https://github.com/unstable-zeros/TaSIL) was inspected at README level only; no external code was run.
3. [R2D2, Kapturowski et al., ICLR 2019, author abstract](https://willdabney.com/publication/r2d2/): parameter lag, representational drift and recurrent-state staleness are distinct recurrent-replay issues worth measuring. Its distributed Q-learning setting differs from this episodic PG pilot. T651 already recomputed full carry under current weights; hidden-state staleness should not be asserted without a failing audit. The OpenReview PDF was blocked by browser verification; no claim of reading its full text.
4. [Kumar et al., ICLR 2022, §2–3 and §4.3](https://arxiv.org/html/2202.10054v1): supervised transfer can favor frozen features over full fine-tuning under some shifts, while full fine-tuning can win with weaker features or smaller shift. This motivates a freezing discriminator, not a theorem for recurrent navigation. Its linear-model assumptions, new-head initialization and image benchmarks differ materially from a trained T501 Actor.

Representation change and fragile closed-loop feedback remain compatible explanations, and can coexist. Neither a large hidden-state difference nor a frozen-backbone win alone proves representation interference caused the six losses.

## First discriminator: actual trajectories

The parent's concurrent [zero-optimizer scope analysis](PARAMETER_SCOPE_ANALYSIS.json) now verifies all384 static scene hashes against the originals and evaluates full causal parent histories. On the six lost histories, median command RMS is 0.0010800 for T651, 0.0001284 for an analytical network carrying only its adapter changes, and 0.0010184 for one carrying only its encoder/GRU/original-head changes. These unexported, untested hybrids motivate freezing; their nonlinear errors are not additive causal shares. Static scene agreement alone does not verify every dynamical parameter. This analysis localizes functional changes on original histories; it does not recover the candidate's visited states.

Each lost candidate NPZ currently contains only `actions`, `native_counts`, and `end_flags`; it has no states, poses or hidden history. A later separately bounded frozen reconstruction of **exactly these six** candidates is therefore needed for the actual-path comparison. Use canonical reset contracts and demand agreement with the original compact arrays, full-array hashes and final integration hashes. Reuse verified parent full records; exact regeneration is not an independent success sample. No such reconstruction is performed by this note.

On both actual histories, compare both policies with independently recomputed carry. Align at shared physical times, report first divergence, path/command differences and terminal events; do not infer obstacle identity from unnamed contact entries. Decomposing total action change into same-history parameter change and changed-history feedback in both orders is descriptive algebra, not a causal stability proof. Raw hidden norms are also coordinate-dependent.

## Next learning discriminator, only under a new finite protocol

Use T651's **same 76 completed trajectories** (60 explicitly reused correct records plus 16 corrected new records), all384 parent histories, initial T501 bytes, terminal safe-GOAL labels, task LOO, clipped-action density, fixed512 scaling, keep10, Adam 3e−5, eight proposal attempts, full causal BPTT, clipping and trust/rollback rules. Only change the trainable subset:

- Reference: valid T651 full Actor, 64,772 parameters. Reuse its audited candidate/results when exact inputs and implementation match; do not duplicate eight all-Actor updates merely to rename the comparison.
- Comparator: freeze normalization, encoder, GRU and original head; train the existing drive/steer adapters, **11,906 parameters**. [T501 source](../../../experiments/t501_joint_command_repair/t501_learning.py) defines that count for the two adapters alone; [T646's `heads_command`](../../../experiments/t646_offline_night/t646_learning.py) likewise freezes the original head. Do not call 11,906 “head plus adapters,” add parameters, or reuse T501's historical mature Adam moments: match T651's fresh Adam initialization.

The fixed batch is on-policy for the initial collection Actor; repeated clipped updates are surrogate optimization, not eight fresh on-policy rounds. Fixed-backbone carry is identical across policies only on an identical recorded observation/action history; actual rollouts can still diverge. Report realized command displacement/KL because the smaller trainable subset can change update magnitude and global gradient clipping. Identical learning rate and step count do not isolate representation change from every optimization effect.

Predeclare final-candidate evaluation: the six lost cases and their exact evidence checks, field24, and canonical sentinels; only a qualifying candidate advances to full407 within the new budget. Keep **all384 original success IDs plus a new field success** as replacement criteria. Recovering six cases or matching the total alone is insufficient. If adapters preserve old tasks and gain, they provide a useful local update subspace; if they preserve old tasks without gaining, plasticity remains unresolved; if they also lose old tasks, body updates are not necessary for this local regression. Failure of eight fixed adapter steps does not prove inadequate capacity or rule out PG.

No learning-rate/keep/rank/seed grid, extra round, critic, EWC penalty or model stack follows automatically. This is a finite discriminator of the present candidates, with training-task reuse disclosed; it supplies no generalization, novelty or safety claim.
