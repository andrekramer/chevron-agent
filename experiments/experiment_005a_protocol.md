# Experiment 005a: frozen geometric-gate isolation

## Question

Did Experiments 004 and 005 fail because the Chevron gate dynamics are wrong,
or because sparse reward did not learn a useful A/N comparison geometry?

## Fixed evaluation

This is a post-Experiment-005 mechanism isolation, not a new training sweep.
It reuses the exact 20 development lifetimes and task configuration from
Experiment 005. No model is trained, no confirmation seed is consumed, and no
parameter is selected from the resulting performance.

The comparison retains the existing content controller and learned projected
Chevron results, then evaluates three parameter-free geometric Chevron
conditions on those same lifetimes:

1. geometric Chevron with a four-entry provisional buffer and separate write
   assent;
2. geometric Chevron with immediate writes;
3. geometric Chevron with coupled read/write assent.

## Frozen gate

For current evidence `A` and retained slot content `N_j`:

```text
M_j = 0.5 * (1 - cosine(A, N_j))
theta_read = 0.5 * (1 - 0.62) = 0.19
r_read_j = sigmoid(40 * (theta_read - M_j))

theta_write = theta_read - 0.05 = 0.14
r_write_j = sigmoid(40 * (theta_write - M_j))
```

The similarity boundary 0.62 is the already-frozen content threshold from
Experiments 004 and 005. The slope 40 makes this gate algebraically match the
content controller's slope 20 when cosine similarity is converted to the
half-cosine mismatch scale. It is not tuned against Experiment 005a outcomes.

Retrieval remains the broad family address distribution:

```text
w_j = alpha_j * r_read_j
q = 1 - sum_j(w_j)
```

Candidate allocation and permanent-memory mechanics are otherwise unchanged.

## Diagnostic rule

The gate dynamics are considered viable in the known geometry if buffered
geometric Chevron achieves all of:

- final old-context accuracy of at least 0.95;
- final novel-context and clean novel-probe accuracy of at least 0.75;
- residual calibration of at least 0.15;
- at least three of four novel-context promotions on average;
- no premature writes and a positive read/write margin; and
- return no more than 0.05 below the fixed content controller.

Passing would localise the earlier failure to representation learning and
justify a new predictive-embedding hypothesis. Failing would implicate the
per-slot gate, residual trigger, or allocation dynamics even when the content
geometry is already informative.

## Fresh-seed confirmation if the diagnostic passes

Because the gate has no trained parameters, confirmation uses 100 fresh task
lifetimes with seeds 60000000 through 60000099. The gate formula, thresholds,
memory rates, capacities, and decision rules remain unchanged. Confirmation
compares the content controller, buffered geometric Chevron, geometric
immediate writing, and coupled-write geometric Chevron.

Confirmation requires all original geometric performance and protection
criteria plus:

- the lower bound of the approximate paired 95% interval for buffered minus
  immediate return is above zero;
- the corresponding lower bound for final novel accuracy is above zero; and
- the lower bounds versus content attention are above -0.05 for both return
  and final novel accuracy.

The learned projected model is not rerun because its weights were not retained
and the question here is the parameter-free gate's robustness, not another
training comparison.

## Outcome

The 20-lifetime diagnostic passed all eight criteria. Buffered geometric
Chevron reached 0.828 return per decision, 0.976 final old accuracy, 0.912
final novel accuracy, 0.975 clean novel-probe accuracy, 0.201 residual
calibration, and 3.90 promotions.

The predeclared confirmation then passed all eleven criteria across 100 fresh
lifetimes. Buffered geometric Chevron reached 0.823 return, 0.972 final old
accuracy, 0.896 final novel accuracy, 0.978 clean novel-probe accuracy, 0.193
residual calibration, and 3.92 promotions.

Relative to immediate writing, buffering improved return by 0.1755, with an
approximate paired 95% interval from +0.1622 to +0.1888. It improved final
novel accuracy by 0.5439, with an interval from +0.5147 to +0.5731. Immediate
writing made premature writes on 8.56% of decisions; buffered Chevron made
none.

Relative to fixed content attention, geometric Chevron improved return by
0.0298 (interval +0.0154 to +0.0442) and final novel accuracy by 0.0943
(interval +0.0603 to +0.1283), while final old accuracy was 0.0090 lower
(interval -0.0116 to -0.0063). Both systems used the same informative cosine
geometry, so this comparison supports the per-slot gate and buffer on this
task, not a general superiority claim.

Separate write assent also produced a small confirmed advantage over coupled
writing: +0.0073 return (interval +0.0010 to +0.0136), with lower established
memory drift. The central conclusion is that the gate dynamics are viable when
the comparison representation is already meaningful. Learning that geometry,
not replacing the sigmoid gate, is the next research problem.
