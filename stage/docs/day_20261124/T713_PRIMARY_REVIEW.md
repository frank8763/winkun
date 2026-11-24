# T713 frozen geometry validation: narrow primary review

2026-11-24. This note reviews the proposed comparison, not its implementation or execution. It adds only this Markdown file: no science code, model forward, geometry generation, native step, optimizer, registration or archive operation. Current T712 findings and RESEARCH_STATUS were read; T698 remains retained. The proposed T713 compares three already frozen models—common T694 parent, compact T703 and B0—on three predetermined obstacle layouts with the same24 field start/goal pairs per layout, with zero learning or model selection.

## Decision

Proceed with one bounded **prospective derived-layout comparison**, subject to a verified reset contract and an explicit feasibility limitation. Per-obstacle translations change environmental geometry and can answer a question that another centimetre-scale initial-pose panel cannot. They do not create three independently sampled benchmark worlds. The inference is performance on these particular previously unused geometric perturbations of a familiar base scene.

T712 already selected compact and B0 under declared retention limits; their selection history must accompany the new frozen comparison. Do not change model membership, weights, layout seeds, displacement magnitudes, task set or repeat allocation after looking at T713 actions/results. No tuning or automatic next experiment follows. [T712 findings](t712/FINDINGS.md)

## Three primary-source checks

| Source and actual reading | What it contributes | Limit for T713 |
|---|---|---|
| [Habitat, Savva et al., ICCV2019, §§4–5](https://arxiv.org/html/1904.01201v2) | Uses disjoint train/validation/test scenes, defines episodes by scene/start/orientation/goal, selects checkpoints on validation before test, and samples goals connected by a navigable path. | Changing starts alone is not a scene split. Three derived layouts are a narrower environmental change than its held-out scenes or cross-dataset tests. Its route and difficulty filters are not authorized for T713. |
| [BARN author project page, About and Usage](https://www.cs.utexas.edu/~xiao/Research/BARN/BARN.html) | Offers300 varied cluttered navigation environments, evaluates the full navigation pipeline, and makes robot footprint and environment difficulty explicit. | This is motivation for geometry-sensitive evaluation, not a claim that Stage implements BARN or inherits its benchmark validity. Its planner baselines are not introduced into Stage. |
| [Agarwal et al., NeurIPS2021, abstract](https://proceedings.neurips.cc/paper_files/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html) | Explains why finite-run aggregate point estimates can support unreliable comparative conclusions. | Three related fixed layouts and one trained instance/model do not justify population or training-seed significance. Exact repeats are audit checks, not extra independent samples. |

The Habitat publisher PDF returned403; its primary arXiv v2 HTML was successfully read instead. No secondary benchmark leaderboard or author implementation was used, and this is not an exhaustive literature review.

## Lock the geometry and the split before model actions

Record the source scene hash, ordered obstacle IDs, deterministic PRNG implementation/seeds, displacement distribution and magnitude, and every per-obstacle translation. Preserve dimensions, types, rotations, collision/sensor flags, materials and dynamics unless explicitly part of the intended intervention. Keep vehicle, sensor definition, action mapping, clock, termination and task horizons fixed. Do not make a failed geometry draw easier by retrying it.

Hash the actual runtime geometry, not merely a seed or JSON description. Compare hashes and coordinates against the base and other layouts. Distinct hashes establish different serialized geometry; they do not establish meaningful difficulty or statistical independence. Record actual displacement summaries so the change cannot be a no-op or an unintended rigid translation. No source-scene file may be overwritten; each reset must restore an isolated canonical model before applying its one layout.

All24 task IDs, exact poses, headings, goal coordinates and caps must remain identical across layouts and models. If the existing canonical task factory resamples a goal or retries a seed in response to geometry, bypass that adaptation using the locked actual task manifest; otherwise this becomes a joint geometry-and-goal intervention. Any accidental duplicate, unsupported geometry or reset failure must remain an explicit protocol/engineering outcome, never a replacement draw.

No source file or library of previous results can prove a layout was never exposed. The practical contract is generated after model freeze, with frozen generator and manifest, no training/model choice using its observations or outcomes, and immutable receipts. After evaluation these layouts are exposed and cannot be described as untouched in later tuning.

## Endpoint validity is not path feasibility

Apply the fixed initial/goal collision predicate before any model action and once per scene/task, using the same resulting valid mask for all three models. Preserve every proposed case, exclusion reason and collision geometry; report proposed72, invalid counts by layout/task, valid denominators and no replacements. An invalid initial state is excluded by protocol, not a policy failure. A valid case that later fails remains a policy execution outcome on that case.

Initial validity must use the vehicle collision geometry at its actual heading, not just a point or lidar ray. The goal predicate must be stated precisely: a clear goal point, a collision-free footprint at one specified orientation, and a nonempty reachable success region are different properties. Do not silently switch predicates to keep more cases. Any allowed boundary handling or obstacle-overlap treatment must be explicit and deterministic before the manifest is frozen.

**Neither two collision-free endpoints nor a free geometric curve proves a dynamically feasible trajectory within the unchanged horizon.** Translated obstacles may disconnect free space or require a longer maneuver. Habitat's explicit path-connected episode generation is a useful counterexample to conflating these contracts. Since T713 permits only endpoint exclusions and prohibits route-based filtering, retain every endpoint-valid case and mark path feasibility and time-feasible reachability **unknown**. Do not add a planner, shortest-path test, route-based rejection or successful-parent screen to select the panel. Do not claim that a failure proves inability on a solvable task, or normalize by an unverified shortest path. This limitation does not prevent an honest paired comparison on the fixed proposed distribution. [Habitat episode specification](https://arxiv.org/html/1904.01201v2)

## Reset and observation consistency are the key engineering risk

The geometry must be installed before the first observation and before history initialization. Clear contact/derived state and rebuild the simulator's required geometry transforms/caches through its supported zero-step initialization path. Rebuild all observation histories, timestamps, previous-action fields and recurrent carry from that new reset; no old-scene sensor samples or hidden state may survive. This is an implementation contract to verify, not a claim that any specific simulator API already provides it.

Confirm that collision detection, ray sensors, any geometry-derived reward/termination calculation and visualization consult the same transformed scene. Updating only the visible geometry or a separate obstacle list can create a physically inconsistent task. Verify selected actual initial rays/contacts with an independent geometric check, including a deliberately changed obstacle, without integrating physics. Same initial payload/scene hash must reach all models; distinct trajectories after the first model action are expected.

Keep the default original-scene reset untouched and demonstrate round-trip isolation after a translated-layout reset. Lock the actual worker/loader/audit adapter source and require corruption rejection for a scene-hash mismatch and a stale-history payload. Hashing nominal manifests alone is insufficient. These checks fit the existing zero-step engineering preflight; they must not become a second scientific scene-selection exercise.

## Reporting and bounded evaluation

Report per-layout and pooled safe success, native collision, timeout and boundary-overlap counts with explicit denominators. Compare compact versus parent, B0 versus parent and compact versus B0 using gained/lost/common-success/common-failure task IDs. Also show parent-relative changes separately for each layout so one layout cannot hide another's regression. The pooled count weights layouts by their valid task counts; if additionally showing an equal-layout mean, label that different weighting and treat an empty layout as undefined rather than0%.

Reuse historical old403 retention audits by exact model hash and label them historical original-scene results. Do not require each new-scene model to reproduce old-scene outcomes, pool old403 into the new denominator or screen new-scene evaluation by success. Full rollout payloads and an independent scene-aware audit remain necessary under the final storage protocol. Any incomplete or failed block remains missing/partial, not zero success.

The requested repeat count is still ambiguous (24 per layout across models versus a smaller fixed subset). Resolve it in the registered manifest and resource bound before launch; select repeats by fixed IDs before outcomes. Exact repeats consume physical budget and support reproducibility but add no new unique cases. Budget endpoint initialization, all proposed valid continuations at maximum caps, repeats, management commands, audit, source transport and cleanup. No implicit retry or deadline extension.

This can provide useful prospective evidence that a frozen finite-retention construction preserves—or loses—its relative behavior after a specific geometry intervention. Positive counts do not establish cross-map population generalization; negative counts do not prove all valid tasks were reachable. A descriptive three-layout result with those limits is a substantive next result, without another start-noise panel or an open-ended qualification queue.
