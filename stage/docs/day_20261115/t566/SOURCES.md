# T566 primary sources and numerical interpretation

2026-11-15. Stage only; documentation research, no simulator, training, optimizer, or experiment launch. Read repository AGENTS.md, current RESEARCH_STATUS.md and NEXT_EXPERIMENT_PLAN_AFTER_T565.md. This note records the proposed numerical method; actual registration, counters and results belong to the locked protocol and run evidence.

## Primary sources actually inspected

1. Clifford Hildreth (1957), *A quadratic programming procedure*, Naval Research Logistics Quarterly 4, 79–85, [publisher record](https://onlinelibrary.wiley.com/doi/10.1002/nav.3800040113). The publisher page confirms authorship, title, date and DOI. Full original article text was unavailable in this retrieval; do not claim the original paper was read in full.
2. Noreen Jamil, Xuemei Chen and Alex Cloninger, *The Hildreth’s Algorithm with Applications to Soft Constraints for User Interface Layout*, [author preprint PDF](https://arxiv.org/pdf/1409.2902). Inspected §2.4 and §4: the stated task is nearest-point projection onto linear inequalities; the cyclic method has an asymptotic convergence analysis. This supplies an accessible primary treatment of the known Hildreth family. Its application-specific soft-constraint prioritization is not part of T566. Neither asymptotic convergence nor a reported rate certifies convergence within 256 sweeps for these Stage gradients.
3. Stephen Boyd and Lieven Vandenberghe, *Convex Optimization*, [author-hosted text](https://www.seas.ucla.edu/~vandenbe/cvxbook/bv_cvxbook.pdf), §5.5, printed pp. 243–244; also [author-hosted slides](https://web.stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf), duality/KKT section. Inspected the KKT conditions and convex sufficiency statement. Primal feasibility, nonnegative dual multipliers, stationarity and complementary slackness together certify the convex subproblem in exact arithmetic. Finite residual tolerance is a numerical acceptance convention, not an exact certificate or a nonlinear-policy theorem.

## Derivation for the proposed Stage subproblem

The following algebra is our specialization, not a quotation or a claimed new algorithm. For fixed proposed parameter displacement d0 and fixed empirical Jacobian rows A, solve

    minimize_delta  0.5 ||delta - d0||_2^2
    subject to      A delta <= b.

The Lagrangian is 0.5||delta-d0||² + lambdaᵀ(A delta-b). Minimizing over delta gives delta=d0-Aᵀlambda. With G=AAᵀ and c=A d0-b, the dual minimization is 0.5 lambdaᵀG lambda-cᵀlambda over lambda>=0 (irrelevant constants omitted). Exact minimization in one coordinate, using the latest other coordinates, gives

    lambda_i <- max(0, lambda_i + (c_i-(G lambda)_i)/G_ii).

Cycle rows in a fixed order from lambda=0, for the registered 256 sweeps. Divide each nonzero row AND its b entry by the same positive row norm; changing A alone changes the feasible set. A zero row with b>=0 is redundant; a zero row with b<0 is infeasible. Dependent rows can make G singular and multipliers nonunique; the strictly convex primal objective still has a unique minimizer whenever the intersection is nonempty. No Gram inverse or added Gram damping is required by this coordinate update.

For the unscaled reconstructed delta, report max positive(A delta-b), max positive(-lambda), ||delta-d0+Aᵀlambda||_infinity, and max abs(lambda_i*(A delta-b)_i). Check all finite and each against the registered 1e-8 tolerance, reporting row-normalized units. Stationarity alone is almost automatic from reconstruction; it cannot replace primal/complementarity checks. A 256-sweep residual failure means numerical contract failure, not proven infeasibility.

## Mapping, guards and limitations

T566 proposes the same T562 93,186 trainable variables and full 27 histories, with damped joint residual d0 using all violating coordinates plus top32. Rows include gradients of all 27 per-history mean squared errors (RMS squared), plus registered near-maximum signed coordinate errors on the protected 24 histories. Right-hand sides are half the current nonnegative margins. For squared RMS the threshold is .02²=.0004; use squared units consistently. For signed residual e_j, a local upper-bound row is sign(e_j)*grad(e_j), with margin .1-abs(e_j). Selection thresholds and counts must be fixed in the protocol. A nonnegative b makes delta=0 feasible; therefore failure of the linear solver in that case cannot demonstrate that its halfspaces are infeasible. A synthetic inconsistent case must allow b<0 and must be distinguished from ordinary finite-iteration nonconvergence.

After solving, cap ||delta|| at .005 by multiplication with a factor in [0,1], then fixed backtracking. Because b>=0 and zero is feasible, such shrinking preserves linear feasibility algebraically. It generally loses the nearest-point optimality/KKT relation for the original d0, so certify/report the projection before shrinking and recheck feasibility after shrinking; do not present the shrunken vector as that exact projection.

Linear rows only describe derivatives at the current parameters. Full nonlinear forward evaluation must still protect original 24 fit gates, every one of the 27 RMS gates, and actual global peak decrease. Omitted near-boundary coordinates, Jacobian changes and temporal nonlinearities can defeat the local surrogate. A rejected step is not global unrepresentability or scene impossibility. This is empirical supervised parameter correction, not CPO, a safety theorem, a new RL update, or an originality claim.

All 27 histories are consumed training/development evidence. Passing their offline action gates is not 27 successful autonomous navigation tasks. Only the separately registered short interface checks and 32 autonomous closed-loop tasks, with original 23 successful IDs preserved and exact replays separately counted, can answer the next closed-loop question. Unexecuted tests remain null; current-layout development results establish neither held-out-layout nor real-vehicle generalization.
