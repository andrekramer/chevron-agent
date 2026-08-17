# Experiment 012a development findings

## Result

Experience-derived slot maturity fixed the targeted core-loss failure without a
measurable performance cost, but the broad frozen development gate failed four
criteria. Fresh-seed confirmation was not run.

The key comparison was clean:

| Condition | Post-shift return | Retention | Novel probe | Core slots lost | Mature evictions |
|---|---:|---:|---:|---:|---:|
| Cold baseline | 0.360 | 0.889 | 0.775 | 0.300 | n/a |
| Four-use maturity | 0.364 | 0.889 | 0.775 | 0.000 | 0.000 |
| Immediate protection | 0.360 | 0.884 | 0.750 | 0.000 | 0.000 |
| Four-use maturity, direct policy | 0.472 | 0.919 | 0.663 | 0.000 | 0.000 |

Four-use maturity eliminated every observed core-slot loss. Relative to the
unprotected cold baseline, its post-shift return changed by +0.004 with an
approximate paired 95% interval from -0.002 to +0.009. Retention changed by
+0.001, interval -0.008 to +0.009, and novel probes were identical. The
protection therefore did not merely exchange forgetting for lower acquisition
or lower reward on these development seeds.

## Mechanism behaviour

The learned-cold mature condition:

- promoted 7.55 of eight initial identities;
- matured 6.25 core slots by the exact shift boundary;
- scored 0.906 on the shift-time core probe;
- retained 0.889 stable accuracy;
- reached 0.925 reversed and 0.775 novel final probes;
- promoted 3.0 of four post-shift novel identities;
- recovered 3.55 of four reversed policy identities;
- evicted zero mature slots and 0.15 immature slots per lifetime;
- deferred no allocations on these seeds; and
- made no duplicate allocation, established overwrite, or under-supported
  write.

The causal unit tests separately verify that a mature slot survives capacity
pressure, an immature slot remains eligible, and a fully mature bank returns a
supported candidate to provisional storage rather than deleting permanent
memory.

## Blanket protection was not better

Protecting every slot immediately also eliminated core loss, but it produced
2.95 allocation deferrals per lifetime on average, with high variance, and
slightly lower retention and novel probes. Four-use maturity produced no
deferrals because recently promoted, weakly supported slots remained eligible
when capacity pressure occurred.

The comparison supports earned protection rather than declaring every first
consolidation permanent forever. The performance differences are small and not
confirmation evidence, but maturity supplies the intended flexibility without
reintroducing core loss.

## Why the frozen gate failed

Four criteria failed:

1. Only 6.25 rather than seven initial slots were mature at the exact step-200
   boundary. The agent nevertheless retained every audited core slot through
   the rest of the lifetime.
2. Retention versus the preloaded control had a paired lower bound of -0.059,
   narrowly outside the frozen -0.05 margin.
3. Retention versus mature direct adaptation had a lower bound of -0.053,
   narrowly outside -0.05.
4. The protected novel-probe advantage over direct adaptation was +0.113, but
   its interval, -0.005 to +0.230, narrowly included zero.

Total return remained within the deliberately broad -0.15 direct-adaptation
margin. Protected policy revision still paid a speed cost: total return was
0.302 versus 0.387 and post-shift return 0.364 versus 0.472. The expected novel
memory advantage was present in the mean but too variable to confirm with
twenty development lifetimes.

## Architectural implication

The storage rule is causally useful:

```text
successful recurrent use -> slot maturity
slot maturity -> no eviction permission
no eviction permission -> candidate remains unresolved
```

This is stronger than merely marking all permanent slots untouchable. It lets
new, weakly supported slots absorb capacity pressure while preserving memories
that have repeatedly supported successful behaviour.

The result is development evidence only. The maturity count and comparative
confidence criteria prevent a confirmation claim. It would be inappropriate to
lower the four-use threshold or relax those margins after seeing this result.

## Next decision

There is no evidence for another Chevron-gate or representation change. The
remaining uncertainty is statistical and task-level: whether the maturity rule
continues to protect self-created memory across more seeds and whether the
protected policy path's novel-memory advantage survives in an environment where
memory has visible behavioural consequences.

A conservative path is a new, explicitly scoped storage confirmation whose
primary question is mature-core preservation versus the unprotected allocator,
while treating direct-policy return as a reported trade-off rather than
changing the old result. The more ambitious path is to carry maturity into the
small rooms-and-routes environment and let trap avoidance, changed routes, and
novel-room learning supply the stronger test.

## Strongest defensible claim

On twenty development lifetimes, use-derived maturity eliminated all observed
loss of self-created core slots while matching the unprotected cold-start
agent's return, retention, and novel acquisition. The broad experiment did not
meet every predeclared criterion, so the rule is promising but not yet
confirmed.
