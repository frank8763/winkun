# T375: current-observation and recurrent-feature steering residual

2026-11-05 user authorized the next step after T374. Stage only; Phase isolated. This is one new bounded function-class pilot, not a continued T374fit or automatic expansion to7/8m. Zero-native preflight PASS, clean local commit and exact remote root tree are required before unique launch.

## Change under test

Parent is immutable T369future research model. Freeze all base, encoder, trueGRU16, original drive/steering and future-head parameters. The deployed Actor still receives the same120causal observations and carries the same recurrent hidden state. Its new steering-logit residual receives the concatenation of16frozen recurrent features and120current normalized/clipped observations,136→4ReLU→1,553trainable parameters. No second controller, current-state table, task/side routing, map/planner/waypoint, camera, runtime teacher or explicit time/distance turn gate. Observation time/history fields already exist; learned timing shortcuts remain possible and later unseen-start tests remain essential.

T374only used16features and73new parameters. Input-path and parameter-count change together, so this is not a matched-capacity causal test. The frozen representation being inadequate is a hypothesis, not an established failure cause. Previous negative models are historical comparisons, not fresh independent controls. No width/seed grid.

Use the same fixed seed20261105372 and deterministic initialization strategy: initialize a136→4first layer, centre its bias at minus its projection of the mean five training repair features, zero the final4→1weights/bias. All initial parent outputs, hidden history and tape fields must match exactly on the full fixed data; saved current step/sequence numerics must agree. This changes the dimension of random weights, not the initial parent function. No fit is used to choose initialization.

Only current steering's pre-tanh logit receives the residual; original1.25tanh and float32action clipping remain. Drive computation is unchanged at identical histories. The step implementation explicitly consumes this current observation rather than a mutable cached input; deterministic execution is enforced. It preserves actual prior-action/history/GRU handling. Future predictions remain frozen and unexecuted.

## Data, optimizer and gate fixed from T374

Reference in place the same14verified factual-parent histories/2,639decisions, old211balanced schedule, neutral branch and five real intervention labels248..252. Complete current histories are unrolled. Labels are successful nonhuman forced behavior, not human/expert actions at arbitrary states or on-policy samples. Other targets are immutable initial-parent outputs. No source NPZ copies, new labels, reward/speed/goal/scene/actuator changes or critic.

Exactly300live+300independent exact shadow Adam calls,lr.001,normcap1,all four weights1. Losses unchanged: patch mean squared error, nonpatch-neutral-route **maximum** squared deviation, sampled old211mean squared deviation, one actual-parent factual route maximum squared deviation cycled%14. Same fit seed. No appended steps or changed weight/optimizer/noise grid. On interruption restore the complete Actor+Adam and preserve error.

Reuse the **identical numerical T374gate** locked from rejected T372forward outputs on the same data:

| Diagnostic metric | Upper limit |
|---|---:|
| Five correction steering-location RMSE |0.06256573740427002|
| Centre248prefix maximum executed-steering deviation |0.08370226621627808|
| First20centre maximum executed-steering deviation |0.01432806346565485|
| Fourteen factual-parent routes maximum executed-steering deviation |0.061291828751564026|

Drive on identical histories must remain exact. T374limits/reference are reused unchanged, not recalculated with the new candidate or relaxed after fitting. This is a comparative screening rule, not a safe tolerance, generalization gate or navigation success certificate. Gate failure completes a scientific negative result and skips native simulation. Zero initial parent fails correction, and rejected T372fails locality; preflight checks these controls.

If every check passes, only the same original-start new0.70m/80scentre is executed once with no intervention, followed by one saved-model independent native regeneration. Compare complete arrays/hidden/history/rewards/native integration. Preserve contact/timeout/partial/invalid separately. No32case or14actual-parent native retention sweep follows under this budget. Even a centreGOAL is an unpromoted research result; fullold275/originalsix54/fresh16/release78 remain unexecuted/null and7/8mremain unentered. Original narrow/56s and new0.70m/80s never share a success denominator.

## Preflight, audit and storage

Preflight0geometry/0native and8discarded clone Adam calls. Check all553parameters, initial zero-output equivalence over full histories and factual executed actions, frozen parent bytes, exact current step/sequence/hidden/tape, live/shadow and mature rollback. After a cloned update, verify nonzero gradients on direct-observation weights, current-observation dependence with frozen history features, and zero dependence on other/future observation rows in that direct branch. Do not claim the initial zero-output branch has nonzero input gradients. These gradient checks are not additional optimizer calls or native measurements.

Independent audit reconstructs all300discarded historical optimizer calls, final Actor/Adam, metrics/gate/source SHA and any conditional native full-array/forward/history/reward evidence, with0new native. Driver waits for all main/native processes to exit before the sole audit and RAM archive/full remote every-member SHA readback. No active-run source sync or external run writes. Original safeT255/T353/T365/human/failure files remain intact.

Bounds: conditional at most2all native commands/80,000all physics, zero otherwise. Main1200s final300reserved; audit300/backup600/all2400from unique launch, earlier external timeout stricter. Own2.5MiB: metadata768KiB, two models128KiBeach, one full centre384KiB, one partial384KiB, trace128KiB,logs128KiB plus128KiB failure reserve=2,228,224B<2,621,440B. References are not copied; models must actually fit their128KiBcap or stop with evidence. At T374closure global1,071,047,950B, full2.5MiB reservation1,073,669,390B<1GiB; actual admission is remeasured and free>=40GiB. No quota expansion, unbacked deletion or hidden storage. Stop this mechanism after registered completion/failure, without new-number continuation or threshold relaxation.
