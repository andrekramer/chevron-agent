# Experiment 009b: frozen shared-capacity confirmation protocol

## Purpose

Experiment 009a's predeclared secondary diagnostic found that increasing the
typed shared provisional bank from four to eight entries improved return,
reversal learning, novel acquisition, and eviction pressure. Its primary split-
queue hypothesis failed, so that result was not eligible for the routing
confirmation.

Experiment 009b is a new, narrowly scoped fresh-seed confirmation of the
capacity result. It does not reopen or rename the failed routing hypothesis.

## Fixed conditions

1. Dual relation with one shared capacity-four provisional bank.
2. Dual relation with one shared capacity-eight provisional bank.
3. Identity-only adaptation with one shared capacity-four bank.

The Experiment 009 task, dual equations, representations, delayed reward,
permanent capacity, two-positive-outcome promotion rule, action sampling,
candidate typing, consolidation destinations, and update rates remain fixed.
No split queue is included because Experiment 009a found no routing advantage.

## Seeds

Confirmation uses 100 untouched lifetimes with seeds
92,000,000–92,000,099. Development seeds 90,000,000–90,000,019 are not reused.
No result from the confirmation seeds may alter a threshold or rule.

## Primary capacity criteria

Shared 8 confirms the capacity result only if all of the following hold:

- final stable accuracy is at least 0.95;
- final reversed and novel accuracy are at least 0.80;
- identity and policy residual calibration are at least 0.15;
- at least three of four novel identities and three of four policy revisions
  consolidate on average;
- no premature writes, established overwrites, or duplicate allocations occur;
- the paired Shared-8-minus-Shared-4 return interval has lower bound above zero;
- the paired reversed-accuracy and novel-accuracy intervals versus Shared 4
  both have lower bounds above zero; and
- mean evictions fall by at least 50% versus Shared 4.

## Comparison with the simple control

Shared 8 is considered competitive with identity-only adaptation if:

- its paired return lower bound is above -0.05;
- its stable-accuracy lower bound is above -0.05;
- its reversed-accuracy lower bound is above -0.05; and
- its novel-accuracy lower bound is above -0.05.

These non-inferiority checks do not claim superiority. They test whether enough
shared capacity removes the large dual-mechanism deficit observed in
Experiment 009.

## Interpretation

Passing the capacity criteria would confirm that one flexible typed candidate
bank can support simultaneous identity novelty and policy revision when sized
for their combined unresolved traffic. Passing non-inferiority would further
show that the protected dual mechanism is competitive with rapid direct value
adaptation on this compact deterministic task.

Failure would leave the capacity effect as a development-only observation and
argue against further architectural expansion on this task.

## Confirmation outcome

All fourteen primary capacity criteria and all four identity-only
non-inferiority criteria passed across seeds 92,000,000–92,000,099.

Shared 8 reached 0.750 return, 0.965 stable accuracy, 0.952 reversed accuracy,
and 0.906 novel accuracy. Relative to Shared 4, it improved return by +0.1557
(approximate paired 95% interval +0.1332 to +0.1782), reversed accuracy by
+0.1644 (+0.1268 to +0.2019), and novel accuracy by +0.2783 (+0.2339 to
+0.3227). Evictions fell from 75.07 to 17.49, a 76.7% reduction.

Shared 8 was non-inferior in return to identity-only adaptation: difference
-0.0100, interval -0.0230 to +0.0030. It was slightly slower on deterministic
policy reversals but better on novel acquisition. No premature writes,
established overwrites, or duplicate allocations occurred.
