# Experiment 010a: frozen pre-consolidation revalidation audit

## Purpose

Experiment 010 inferred obsolete policies from delayed stochastic reward, but
its 100-lifetime confirmation failed because two provisional new-identity
candidates became recognisable as established identities after their evidence
was averaged. The candidates were still allocated because identity novelty was
tested only when evidence entered the provisional bank.

Experiment 010a tests one architectural correction:

```text
before permanent new-identity allocation,
rerun identity assent on the aggregated candidate
```

This is a new untouched-seed audit. It does not alter or rename the failed
Experiment 010 confirmation.

## Fixed mechanism

The complete Experiment 010 task remains fixed:

- four address families, eight established and four novel identities;
- four actions;
- 800 decisions with shift at 200 and retention phase at 600;
- delayed reward after three decisions;
- 10% misleading reward signs;
- fixed noisy 12-dimensional identity geometry;
- cosine identity boundary 0.62 and sigmoid slope 40;
- identity residual trigger 0.80 and admitted-mass threshold 0.25;
- one shared capacity-eight typed provisional bank;
- two positive outcomes for new-identity promotion;
- two unexpected incumbent failures for policy veto; and
- two positive alternative outcomes for protected policy revision.

No threshold, capacity, support count, reward process, task mixture, or policy
rule changes.

## Revalidation rule

Let `P_l^id` be the normalised averaged identity evidence in a provisional
new-identity candidate at the moment its delayed support criterion is met.

Compare it with all established slots in its broad address family:

```text
s_lj = cosine(P_l_identity, N_j_identity)
```

Permanent allocation is allowed only if:

```text
max_j(s_lj) < 0.62
```

If the candidate now matches an established identity at or above 0.62, the
candidate is removed from provisional storage and counted as an identity
reconciliation. It does not allocate or rewrite any permanent slot. Later work
may route reconciled action evidence into policy evaluation, but Experiment
010a deliberately adds no such path.

## Conditions

1. **Original protected Chevron**: the frozen Experiment 010 protected
   condition, including audit-only duplicate detection but no allocation veto.
2. **Revalidated protected Chevron**: identical except that a matching
   aggregated candidate is reconciled instead of allocated.

Both conditions receive the same paired lifetime and action-randomness seed.

## Seeds

This correction has no tunable parameter and therefore proceeds directly to a
100-lifetime untouched-seed audit using seeds
`102,000,000-102,000,099`. No result from those seeds may alter the rule.

## Frozen criteria

The correction passes only if revalidated Chevron:

- makes zero duplicate identity allocations;
- makes zero established-memory overwrites and zero under-supported writes;
- retains at least 0.90 phase-three stable accuracy;
- reaches at least 0.75 final reversed and novel probe accuracy;
- promotes at least three of four novel identities and revises at least three
  of four changed policy identities on average;
- is non-inferior to original protected Chevron in realised return, clean
  accuracy, retention, reversed probe, and novel probe, with every paired 95%
  lower bound above `-0.01`; and
- records at least as many identity reconciliations as the original condition
  records duplicate allocations on the paired seeds.

The final condition checks that the correction intercepts the mechanism it was
introduced to prevent. If the untouched seeds contain no attempted duplicate,
both quantities may be zero and the deterministic unit test remains the direct
mechanism evidence.

## Interpretation

Passing would close the specific protection gap exposed by Experiment 010 and
support moving to separately learned identity geometry while retaining the
retrospective policy mechanism.

Failure would show either that promotion-time revalidation damages legitimate
novel acquisition or that duplicate creation has another route. No threshold
sweep should follow until the failed cases are explained.

## Audit outcome

All fourteen frozen criteria passed across seeds
`102,000,000-102,000,099`.

The original path produced one duplicate allocation. Revalidated Chevron
intercepted exactly that candidate, recorded one identity reconciliation, and
made zero duplicate allocations. It also made no established-memory overwrite
or under-supported write.

Return changed by -0.000025 with an approximate paired 95% interval from
-0.000074 to +0.000024. Retention changed by -0.000050, interval -0.000148 to
+0.000048. Reversed and novel probes were identical between conditions.

The correction therefore closed the observed protection gap without a
measurable performance cost. Experiment 010 remains a failed confirmation;
Experiment 010a is the successful new correction audit.
