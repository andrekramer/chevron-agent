# Experiment 005a findings: the gate works when geometry works

## Result

Experiment 005a changes the interpretation of the preceding negative result.
The Chevron gate, residual allocation trigger, and provisional buffer work well
when A and N are compared in an already meaningful geometry. The failed part
of Experiments 004 and 005 was learning that geometry from sparse delayed
reward.

This conclusion passed a frozen 20-lifetime diagnostic and a predeclared
confirmation on 100 fresh task lifetimes. No model was trained and no parameter
was changed between the two runs.

## Frozen formula

The gate used half-cosine mismatch:

```text
M_j = 0.5 * (1 - cosine(A, N_j))
r_read_j = sigmoid(40 * (0.19 - M_j))
r_write_j = sigmoid(40 * (0.14 - M_j))
w_j = alpha_j * r_read_j
q = 1 - sum_j(w_j)
```

The 0.19 read threshold is exactly the existing cosine-similarity boundary of
0.62 expressed on the half-cosine mismatch scale. The slope conversion and
0.05 write margin were frozen before evaluation.

## Fresh-seed confirmation

| Condition | Return | Final old | Final new | New probe | q calibration | Promotions | Premature writes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Content attention + buffer | 0.794 | 0.981 | 0.802 | 0.887 | 0.457 | 3.53 | 0.000 |
| Geometric Chevron + buffer | 0.823 | 0.972 | 0.896 | 0.978 | 0.193 | 3.92 | 0.000 |
| Geometric Chevron + immediate | 0.648 | 0.970 | 0.352 | 0.295 | 0.186 | 0.00 | 0.0856 |
| Geometric Chevron + coupled write | 0.816 | 0.970 | 0.878 | 0.963 | 0.196 | 3.86 | 0.000 |

Buffered geometric Chevron passed every absolute performance, calibration,
promotion, and protection threshold.

## What the buffer contributes

Against geometric immediate writing, the buffered system improved return by
0.1755. The approximate paired 95% interval was +0.1622 to +0.1888, and it won
on 99 of 100 lifetimes.

Final novel accuracy improved by 0.5439, with an interval from +0.5147 to
+0.5731 and wins on all 100 lifetimes. The two conditions retained established
contexts equally well. The difference is that the immediate system committed
speculative actions directly to N, while the buffer waited for two coherent
positive outcomes before promotion.

This is the cleanest evidence so far for the temporary learning buffer. It is
not merely storage capacity: it separates unresolved use and exploration from
permission to alter retained memory.

## Comparison with content attention

Geometric Chevron improved return over the fixed content controller by 0.0298,
with an approximate paired interval from +0.0154 to +0.0442. Final novel
accuracy improved by 0.0943, with an interval from +0.0603 to +0.1283.

The content controller retained old contexts 0.0090 better, with a narrow
interval from +0.0063 to +0.0116 in its favour. Thus the result is a genuine
stability-plasticity trade: geometric Chevron acquired new contexts more
reliably at a small cost in old-context online accuracy. Both old probe scores
were effectively perfect.

Both systems were handed the same useful content geometry. This experiment
therefore does not establish general Chevron superiority over learned standard
attention. It shows that per-slot admission and residual-driven buffering add
value once comparison is meaningful.

## Separate write permission

Buffered Chevron with a stricter write gate modestly exceeded its coupled-write
ablation by 0.0073 return, with an interval from +0.0010 to +0.0136. It also had
slightly higher old and novel accuracy and lower retained-memory drift (0.0442
versus 0.0502).

The effect is small but consistent with the design claim: evidence sufficient
for current use need not automatically receive equal permission to revise N.

## Strongest defensible claim

> On this delayed-context task, a parameter-free Chevron gate operating in a
> meaningful cosine geometry combined high retention with reliable novel
> acquisition. Its provisional buffer decisively outperformed immediate
> writing across 100 fresh lifetimes, while separate write permission supplied
> a smaller additional protection benefit.

The claim is limited to compact synthetic evidence with known address families,
fixed representations, deterministic delayed rewards, and hand-established
comparison scale. It is not yet a result about visual representation learning,
spatial agency, or a persistent core self.

## Next research problem

The next component to change is not the sigmoid assent gate. It is the source
of the geometry supplied to mismatch.

A suitable next hypothesis is a predictive encoder trained with dense temporal
signals: observations should be close when they predict compatible future
outcomes and separate when their predicted transitions or consequences differ.
Chevron assent can then continue to use the confirmed cosine mismatch and
protected buffer unchanged.

That encoder should first be tested on this task by hiding or scrambling the
given content geometry. Only after it reconstructs a comparison space that
recovers the geometric-gate result should the project proceed to a spatial
trap/shortcut environment.
