# Experiment 010 findings: retrospective policy assent

## Outcome

The policy result replicated strongly, but the complete confirmation failed one
protection invariant.

Experiment 010 removed the supplied pre-action policy signature. The agent saw
only broad address and noisy identity evidence, acted from retained values, and
received reward three decisions later. Ten percent of reward signs were
deliberately misleading.

Protected retrospective Chevron passed all fourteen frozen development
criteria across twenty paired lifetimes. That authorised the predeclared
100-lifetime fresh-seed confirmation. Thirteen of fourteen confirmation
criteria passed. The failure was two duplicate identity allocations, violating
the declared zero-duplicate criterion.

Confirmation is therefore **not passed**.

## Fresh-seed results

| Condition | Return | Clean accuracy | Retention | Reversed probe | Novel probe | False stable revisions |
|---|---:|---:|---:|---:|---:|---:|
| Direct value adaptation | 0.580 | 0.865 | 0.946 | 0.985 | 0.860 | n/a |
| Protected retrospective Chevron | 0.538 | 0.838 | 0.948 | 0.980 | 0.915 | 0.06 |
| Fast-veto Chevron | 0.316 | 0.699 | 0.791 | 0.965 | 0.898 | 0.30 |
| Immediate-write Chevron | 0.323 | 0.703 | 0.764 | 0.918 | 0.853 | 16.87 |

Protected Chevron consolidated 3.77 of four novel identities and revised 3.96
of four changed policies per lifetime. Its policy residual calibration was
0.614. It made no under-supported writes and no established-identity
overwrites.

## The cost of protection

Relative to direct value adaptation, protected Chevron lost 0.0422 realised
return, with an approximate paired 95% interval from -0.0498 to -0.0346. Clean
accuracy was 0.0274 lower, interval -0.0318 to -0.0229. Both remained inside
the predeclared non-inferiority margin.

Stable retention was statistically indistinguishable: protected Chevron was
0.0011 higher, interval -0.0034 to +0.0055. Reversed probes were also
indistinguishable. Protected Chevron was significantly better on novel probes:
+0.055, interval +0.0109 to +0.0991.

The tradeoff is therefore clear. Direct value adaptation was faster and
simpler. Protected revision paid a modest performance cost while retaining
stable policy and improving novel acquisition.

## Why vigilance mattered

One misleading failure was too weak a trigger. Fast-veto Chevron entered search
after a single unexpected outcome and lost 0.222 return relative to protected
Chevron. Its stable retention fell to 0.791.

Immediate writing was worse. It performed 31.29 under-supported revisions per
lifetime, including 16.87 false revisions of stable policies. Protected Chevron
reduced that to 0.06 false stable revisions and improved retention by 0.184.

Protected Chevron created 75.65 policy suspicions per lifetime and dismissed
64.02 after contradictory evidence. That is computationally cheap in this
small task but behaviourally important: most surprises were correctly treated
as temporary doubts rather than permanent changes.

## The failed invariant

Protected Chevron made two duplicate identity allocations across the 100
untouched confirmation lifetimes, on seeds `101000074` and `101000084`.

Both had the same structure:

- two individually noisy observations were routed as a provisional new
  identity;
- they accumulated two positive delayed outcomes;
- averaging the two observations made the candidate recognisably similar to an
  existing identity; and
- consolidation still allocated a new slot because novelty was tested only on
  entry to the provisional bank.

At promotion, the candidates had cosine similarities `0.6466` and `0.6863` to
their existing memories, both above the frozen `0.62` identity boundary.

The implementation detected and counted the duplication for audit, but did not
veto the write. This reveals a missing protection step:

```text
provisional novelty at entry
does not guarantee
novelty after evidence has accumulated
```

Before permanent allocation, an averaged candidate must be compared again
with established identity memory. If it now matches, it should be reconciled
with that identity or discarded rather than allocated as new.

## Strongest defensible result

> Without access to a pre-action policy signature, protected retrospective
> Chevron inferred policy mismatch from delayed stochastic reward, learned
> almost all four policy reversals, preserved stable retention, and prevented
> the large false-revision failure of one-outcome veto and writing. It remained
> within the declared performance margin of direct value adaptation and
> improved novel probes, but the full confirmation failed because two noisy
> provisional identity candidates were not revalidated before allocation.

## Next experiment

Experiment 010a should make one change only: re-run identity assent on the
aggregated provisional candidate immediately before permanent allocation. No
gate threshold, reward rule, support requirement, capacity, or policy mechanism
should change.

That is a protection correction, not an opportunity to tune performance. It
should use untouched seeds and test that eliminating duplicates does not reduce
novel acquisition or the confirmed retrospective-policy behaviour.
