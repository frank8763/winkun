# T707 numerical representation review

Date: 2026-11-23. Stage only. This note reviews the failed T706 fixed-feature convex fitting representation; it does not claim a solved numerical problem or a navigation improvement. No Phase models, results, planning research, external planner or waypoints are used. The researcher inspected T706 source/protocol/evidence and primary solver sources without loading actual models or running a solver. The bounded numerical diagnosis results below were supplied by the main agent; they are separate, consumed work, not a new budget in this note.

## Evidence and limits

T706 stopped in the first, mean-taskwise arm with Clarabel 0.11.1 `NumericalError` after 17 iterations. There was no candidate Actor assignment or native physics, and the paired scientific comparison was not executed. The original failure, raw evidence, deadlines and retained T698 remain unchanged. Local anchors: `stage/experiments/t706_taskwise_replay/fit.py`, `stage/docs/day_20261123/t706/PLAN.md`, `stage/docs/day_20261123/t706/FINDINGS.md`, and the original T706 fit ledger.

The main agent's newly authorized bounded diagnosis reports: exact reproduction of the original error at 17 iterations; removing the unused 219th mean-arm variable still errors at 17; centering at W0 and whitening the mean Hessian still errors at 15. These consumed 49 iterations altogether, with no candidate assignment or native physics. Thus neither removing the redundant variable nor this whitening representation fixed the failure. They are not grounds for further unbounded retries. Reported source augmented Gram eigenvalues were positive, factor reconstruction error about 1e-15, and B0 normalized source norms 1 ± about 1e-14; one source's minimum eigenvalue was about 1e-9. Those observations support validity of the original quadratic representation, not identification of its failure mechanism.

Clarabel's documented problem accepts a positive semidefinite P, so a singular P alone is not an API violation. Its status enum separates numerical error from primal/dual infeasibility. The unused variable has zero P row/column, q entry and A column in the mean arm, so eliminating it is an exact algebraic simplification, but the diagnosis rejects it as a sufficient explanation here. [Official problem definition](https://clarabel.org/stable/), [version-pinned status source](https://raw.githubusercontent.com/oxfordcontrol/Clarabel.rs/v0.11.1/src/solver/core/solver.rs).

## Fixed scientific problem

For each trajectory s, let G_s=Z_s^T Z_s/(2n_s), B_s=Z_s^T Y_s/(2n_s), and c_s=||Y_s||_F²/(2n_s). For W of shape 109 by 2,

    L_s(W) = tr(W^T G_s W) - 2 tr(W^T B_s) + c_s.

The seven source constraints remain L_s(W) <= beta_s, where beta_s is the originally locked B0 loss, not an enlarged or recomputed acceptance target. Mean minimizes mean_old L_i(W)+rho||W-W0||_F²; DRO minimizes max_old L_i(W)+the identical ridge. The original rho, normalization, data, seven bounds, objectives, capacity, solver version/settings and acceptance tolerances stay fixed. A numerical representation change is not a new learning mechanism or an improvement claim.

Writing Delta=W-W0 gives the equivalent source inequality

    tr(Delta^T G_s Delta) + 2 tr(Delta^T (G_s W0-B_s)) <= 0

when beta_s=L_s(W0) in exact arithmetic. This avoids subtracting two full losses when evaluating the difference, but does not supply strict feasibility or guarantee a stable conic solve. In finite arithmetic the frozen beta_s must be preserved explicitly; silently replacing it by a separately rounded L_s(W0) would change the tuple.

## Recommended fixed next representation: completed-square source SOC

For each source, use the same original G_s/B_s/c_s and locked beta_s. Require an unmodified Cholesky factorization G_s=L_s L_s^T; reject if positive definiteness fails. Solve linear systems (do not explicitly form an inverse) for C_s=G_s^{-1}B_s. Define

    kappa_s = c_s - tr(B_s^T C_s)
    r_s²    = beta_s - kappa_s
    F_s     = blockdiag(L_s^T, L_s^T)
    center_s= vec_F(C_s).

Then, in exact arithmetic,

    L_s(W) = ||F_s (vec_F(W)-center_s)||_2² + kappa_s,
    L_s(W) <= beta_s  iff  ||F_s (vec_F(W)-center_s)||_2 <= sqrt(r_s²).

Require finite r_s²>0; reject rather than clip it, add ridge, relax a bound or use a pseudoinverse fallback. With Clarabel convention A x+s=b, the source SOC block has first row A=0, b=sqrt(r_s²), and lower rows A=-F_s, b=-F_s center_s. Therefore its slack is `(sqrt(r_s²), F_s(x-center_s))`, which has the correct sign and cone membership. If the block is divided by its positive radius, both A and b must receive the same scaling. Clarabel's second-order cone definition is exactly `||lower||_2 <= first`. [Official cone definitions](https://clarabel.org/stable/api_cone_types/).

This replaces the augmented `[W;1]` loss factor by a centered 218-coordinate norm and subtracts the irreducible least-squares constant from both sides algebraically. It does **not** remove c_s from the scientific loss, alter beta_s, or assume irreducible loss is zero. It is the selected single next representation, without whitening. The mean arm can eliminate the unused epigraph variable; DRO retains its necessary epigraph variable. Original old-task epigraph SOCs and quadratic objective must remain mathematically unchanged, including all linear terms and fixed ridge.

This representation may change scaling and cancellation patterns. Computing C_s can itself amplify an ill-conditioned G_s, and computing beta_s-kappa_s can lose precision. Hence it is a hypothesis worth one fixed bounded attempt, not a demonstrated repair. An augmented residual factor built directly from `[Z_s,-Y_s]` by QR could avoid explicitly forming the augmented normal equations, but it also changes floating-point operations; it is not an authorized fallback in the selected attempt.

## Alternatives considered, not selected as rescue loops

A fixed common invertible coordinate transform centered at W0 is an exact reparameterization only if every objective/constraint and the original Euclidean ridge transform together. Replacing the ridge by a Euclidean norm in whitened coordinates changes the objective. The performed mean-Hessian whitening still failed, so there is no empirical basis to promise common old-Gram whitening will fix this instance.

The mean problem also has a seven-scalar nonnegative Lagrange dual: with H_s=blockdiag(G_s,G_s), h_s=vec_F(B_s), and x0=vec_F(W0), define K(lambda)=H_old+rho I+sum_s lambda_s H_s and v(lambda)=rho x0+sum_s lambda_s h_s (old labels are zero here). Minimizing the Lagrangian over x gives x=K(lambda)^{-1}v(lambda); its dual value is the corresponding constant minus v^T K^{-1}v. This is a useful independent formulation concept, but any implemented dual optimizer would need its own bounded accounting and original primal/KKT certificate. B0 feasibility alone does not establish Slater strict feasibility or dual attainment. DRO additionally introduces multipliers for the 403 old-ID epigraph inequalities with sum one, so it is not the same seven-variable dual. Do not silently substitute dual search for the selected conic fit. These identities are direct algebraic derivations from the displayed problem, not measured findings.

## Solver sources and what they do not establish

The version-pinned settings source already enables equilibration, KKT static/dynamic regularization and iterative refinement. Such internal KKT regularization is different from increasing the scientific rho. Leave the fixed settings unchanged; no tolerance loosening, alternative solver, version upgrade or regularization sweep is justified by this review. [Clarabel 0.11.1 settings source](https://raw.githubusercontent.com/oxfordcontrol/Clarabel.rs/v0.11.1/src/solver/implementations/default/settings.rs).

The solver paper describes a primal-dual conic method with quadratic objectives and discusses its linear solves; it supplies no instance-specific guarantee for these seven nearly active loss constraints. An upstream numerical-error issue concerns a small unconstrained f32 example and a maintainer's request to try f64, unlike this already-float64 case. The 0.11.1 changelog lists linking changes, not a documented fix for this Stage failure. None proves the local error's cause. [Solver paper](https://arxiv.org/html/2405.12762v1), [upstream issue 195](https://github.com/oxfordcontrol/Clarabel.rs/issues/195), [0.11.1 changelog](https://raw.githubusercontent.com/oxfordcontrol/Clarabel.rs/v0.11.1/CHANGELOG.md).

## Required evidence for the one new bounded attempt

Before actual source-data optimization, lock the representation and verify synthetic known-answer and infeasible rejection cases. Then preserve every actual solver iteration and every failure. Accept only the originally required `Solved` status and unchanged original-space primal feasibility, cone dual feasibility, stationarity and dual-gap certificate. Check completed-square versus original Gram loss and full raw trajectory loss for every ID, and enforce every original source bound with its original tolerance. Report all floating discrepancy magnitudes. Exact algebraic equivalence and improved-looking conditioning do not guarantee identical floating-point tuples, identical optimizer output or a numerical fix.

No scientific fit was run by this research task. This note does not authorize restarting T706, extending any old deadline or spending a second diagnosis allowance. Main-agent supplied T707 anchor is 18:02:20, with learning 19:02, native 22:47 and all-work 23:02; use the exact registered timestamps and earlier applicable limit. The completed-square attempt requires its own locked T707 registration and must stop on failure. Until it passes, the numerical issue remains unresolved and there is no mean-versus-DRO native comparison.
