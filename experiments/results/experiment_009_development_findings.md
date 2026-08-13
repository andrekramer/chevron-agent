# Experiment 009 development findings

## Outcome

Separating identity assent from policy assent was better than collapsing both
relations into one allocation signal, but it did not beat a simpler identity-
only memory that adapted action values directly from delayed reward. The frozen
confirmation gate failed and confirmation seeds were not used.

## Main result

| Condition | Return | Stable | Reversed | Novel | Identity q calibration | Policy q calibration | New promotions | Revisions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dual relation + buffer | 0.587 | 0.951 | 0.826 | 0.598 | 0.273 | 0.991 | 2.85 | 4.20 |
| Collapsed relation + buffer | 0.540 | 0.963 | 0.615 | 0.583 | 0.150 | — | 4.70 | 0.00 |
| Identity only + buffer | 0.746 | 0.956 | 0.973 | 0.802 | 0.277 | — | 3.75 | 0.00 |
| Dual relation + immediate | 0.573 | 0.872 | 0.957 | 0.311 | — | — | 0.00 | 0.00 |

Dual buffered Chevron exceeded the collapsed condition by 0.047 return, with
an approximate paired 95% interval from +0.006 to +0.088. It improved final
reversed-context accuracy by 0.211, interval +0.160 to +0.261. The dual routing
created no duplicate identities, while collapsed routing allocated 5.85
duplicates per lifetime on average.

This supports the narrow causal claim that a policy mismatch should not be
interpreted automatically as a new memory identity.

## Why the full gate failed

The identity-only condition was substantially stronger. It exceeded dual
buffering by 0.158 return, interval +0.107 to +0.209, and reached 0.973 reversed
and 0.802 novel final accuracy.

In this deterministic four-action task, the simpler agent could revise action
values efficiently after negative delayed reward. It did not need to withhold
the old policy, explore under a separate policy residual, accumulate two
positive outcomes, and then consolidate a revision. The additional caution
made dual policy revision slower rather than safer.

There was also a concrete routing bottleneck. Novel-identity candidates and
policy-revision candidates shared one capacity-four provisional buffer. Dual
Chevron averaged 78.85 evictions, compared with 19.90 for identity-only
buffering. It consolidated all four policy reversals but only 2.85 of four
novel identities on average. Its final novel accuracy was therefore only 0.598,
below the frozen 0.75 criterion.

## Immediate revision

Immediate revision produced strong reversal accuracy but wrote before outcome
on 11.8% of decisions. It reduced stable accuracy to 0.872 and novel accuracy
to 0.311. Buffered dual Chevron improved novel accuracy over immediate revision
by 0.287, interval +0.211 to +0.363, and stable accuracy by 0.079, interval
+0.037 to +0.121. Its overall return advantage was not statistically clear.

This preserves the earlier protection finding: fast revision is easy when the
system is allowed to believe unresolved evidence immediately, but it damages
other retained behaviour.

## Interpretation

Experiment 9 refines the Experiment 8 conclusion:

- identity and policy compatibility are genuinely different relations;
- collapsing policy mismatch into identity allocation causes duplicate memory
  and weaker reversal learning;
- separating the relations is not by itself sufficient to outperform simple
  retrospective value adaptation; and
- once the residuals are separated, their consolidation traffic may also need
  separate routing or priority.

This is not a reason to start representation learning for two channels yet.
The next justified diagnostic is smaller: give identity novelty and policy
revision separate bounded provisional queues, without changing the gates,
support threshold, or task. If that removes the novelty deficit but still
loses to identity-only adaptation, the remaining issue is the behavioural cost
of conservative policy veto rather than buffer interference.
