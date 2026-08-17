# Experiment 012 development findings

## Result

Empty-memory Chevron learned a useful core, but the frozen development gate
failed. Fresh-seed confirmation was not run.

Learned-cold protected Chevron passed twenty-one of twenty-four predeclared
criteria. It failed only the three return non-inferiority criteria: against
oracle-cold identity, learned preloaded memory, and learned-cold direct
adaptation.

The first generated report exposed an audit-only implementation error: the
shift-time core probe was scored against the policies that would become correct
after the shift. That made even a perfectly preloaded agent score 0.5. The probe
was corrected to use the initial policies and the same frozen seeds were rerun.
No agent computation, threshold, training rule, comparison, or criterion was
changed.

## What worked

Starting with zero permanent slots, learned-cold protected Chevron:

- promoted 7.5 of eight initial identities before the shift;
- reached 0.815 action accuracy over the final 50 bootstrap decisions;
- scored 0.900 on the eight-context policy probe at the shift;
- retained 0.921 stable accuracy after novelty and reversals;
- reached 0.825 reversed and 0.863 novel final probes;
- promoted 3.5 of four post-shift novel identities;
- recovered 3.25 of four reversed policy identities;
- produced identity residual calibration of 0.262 and policy residual
  calibration of 0.470; and
- made no geometrically detected duplicate allocation, preloaded-established
  overwrite, or under-supported permanent write.

The learned representation remained decisive under cold start. Against the raw
sensor, it improved return by +0.249, with an approximate paired 95% interval
from +0.179 to +0.318. It improved the shift-time core probe by +0.188,
interval +0.122 to +0.253, retention by +0.119, and the final novel probe by
+0.688.

The learned and oracle cold-start agents constructed almost equally accurate
initial cores: shift probes were 0.900 and 0.906, with paired difference
-0.006 and interval -0.054 to +0.041. The remaining representation cost appeared
mainly in online action and policy learning rather than the final core identity
map.

## Why the gate failed

Learned-cold return was 0.316. It trailed oracle-cold Chevron by 0.083, with a
paired interval from -0.129 to -0.037, outside the frozen -0.08 margin.

It trailed the learned preloaded control by 0.189 return, interval -0.224 to
-0.153, outside the -0.15 bootstrap margin. Much of this is the real cost of
discovering eight policies from random initial action rather than receiving
them at decision zero. By the retention phase the gap was only -0.011 and
remained comfortably inside the frozen retention margin.

Protected cold start also trailed direct adaptation by 0.095 return, interval
-0.126 to -0.063, missing the -0.08 margin. Protection again bought stronger
consolidated novelty: its final novel probe exceeded direct adaptation by
+0.163, interval +0.041 to +0.284.

## The more important diagnostic

Although not a predeclared gate criterion, the core-slot audit exposed a real
architectural gap. Five self-created core memory ids were later lost across
three of twenty learned-cold lifetimes. These cases occurred under heavy
unresolved traffic: learned-cold Chevron averaged 71.35 candidate-bank
evictions.

This can coexist with zero `duplicate_allocations` because that counter uses the
learned cosine boundary. A noisy familiar identity can remain below the 0.62
revalidation boundary, be treated as genuinely new, consume a permanent slot,
and eventually force replacement of an older self-created slot. Hidden audit
identity then reveals a semantic false split that the learned geometry could
not recognise.

The existing `established_overwrites` invariant protects memories installed as
established at construction time. A core created by the agent begins as a
normal promoted slot, so the invariant does not yet express when a learned
memory has earned the right not to be casually evicted.

## Architectural implication

Cold-start acquisition itself is viable. The agent can construct a high-quality
core through the ordinary provisional path and later learn novelty and policy
reversals. What is missing is not a new identity threshold. It is an explicit
transition from newly promoted memory to mature retained memory, together with
separate permission to forget.

The next narrow experiment should add slot maturity based on repeated
successful, assented use. Allocation may replace an immature slot, but it
should not replace a mature slot merely because permanent capacity is full. If
every slot is mature, unresolved evidence should remain provisional until an
explicit forgetting or expansion decision is available.

This is the storage analogue of the read/write separation already established:

```text
permission to create a new memory
is not automatically
permission to destroy an old memory
```

That correction should be tested before spatial RL. It directly concerns the
project's aim of constructing and maintaining a core self, rather than adding
another representational objective or sweeping the Chevron gate.

## Strongest defensible claim

On development seeds, learned Chevron constructed most of an eight-memory core
from empty permanent storage, reached a 0.900 core-policy probe, retained 0.921
stable accuracy, and later acquired novelty and policy reversals. It was much
better than the raw-sensor cold-start control but incurred more return cost than
predeclared and occasionally lost self-created core slots under capacity
pressure. The cold-start architecture is therefore promising but not ready for
confirmation or spatial RL.
