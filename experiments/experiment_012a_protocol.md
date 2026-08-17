# Experiment 012a: frozen slot-maturity and eviction-permission protocol

## Purpose

Experiment 012 showed that learned Chevron can construct a high-quality core
from empty permanent memory, but five self-created core memory ids were later
lost across three of twenty development lifetimes. The existing overwrite
invariant protects memories installed as established at construction time; it
does not define when an agent-created memory has earned similar protection.

Experiment 012a tests one architectural separation:

```text
permission to allocate a new memory
is not permission to evict a mature memory
```

It does not rename or rescue the failed Experiment 012 development gate.

## Fixed components

Everything unrelated to eviction eligibility remains fixed:

- the Experiment 011a pairwise temporal identity encoder;
- deterministic nonlinear sensor seed 606;
- 750 encoder-training steps and all representation hyperparameters;
- empty permanent memory at lifetime start;
- four address families, eight initial and four later novel identities;
- four actions and 800 decisions;
- shift at 200 and retention phase at 600;
- reward delayed by three decisions and reversed on 10% of events;
- cosine identity threshold 0.62 and sigmoid slope 40;
- identity residual trigger 0.80 and admitted-mass threshold 0.25;
- one shared capacity-eight typed candidate bank;
- two positive outcomes for identity promotion;
- two unexpected incumbent failures before policy search;
- two positive alternative outcomes before policy revision;
- permanent capacity twelve; and
- promotion-time identity revalidation.

The encoder is pretrained and frozen. Experiment 012a does not test online
representation drift.

## Slot maturity

Every newly promoted permanent slot starts immature with successful-use support
`h_j = 0`.

After delayed reward arrives, support increases by one only when:

- the observation was admitted to that existing slot;
- maximum admitted identity mass was at least 0.25; and
- the observed delayed outcome was positive.

The slot becomes mature after four such post-promotion successful uses:

```text
m_j = 1 if h_j >= 4 else 0
```

Maturity concerns identity retention, not policy correctness. A mature identity
remains mature when its policy later reverses. Negative outcomes may trigger the
existing policy-revision path but do not make the identity disposable.

## Eviction permission

When permanent capacity is full, a new identity may replace only an immature
slot:

```text
eligible_for_eviction_j = 1 - m_j
```

Among eligible slots, the least recently used slot is selected, matching the
existing allocator. A mature slot has no eviction permission.

If every permanent slot is mature, allocation is deferred. The supported
candidate is returned to the provisional bank and remains unresolved. It may be
reconsidered after later evidence, explicit forgetting, or future capacity
expansion. Experiment 012a adds no forgetting rule and no capacity growth.

## Conditions

1. **Learned preloaded protected**: the confirmed preloaded reference.
2. **Learned cold baseline**: the unmodified Experiment 012 cold-start
   allocator; any non-preloaded permanent slot may be replaced under capacity
   pressure.
3. **Learned cold mature**: four successful admitted uses create mature-slot
   eviction protection.
4. **Learned cold immediately protected**: every promoted slot is mature
   immediately. This tests whether blanket protection blocks later legitimate
   acquisition.
5. **Learned cold mature direct**: the same maturity mechanism with direct
   action-value adaptation rather than protected retrospective policy revision.

All conditions use paired encoder, lifetime, reward-noise, and action-randomness
seeds.

## Metrics

Experiment 012 metrics remain, including core acquisition, shift-time core
probe, retention, novelty, policy revision, residual calibration, and memory
invariants. Experiment 012a adds:

- mature initial slots at the shift;
- evictions of immature and mature slots;
- allocation deferrals because no slot has eviction permission;
- post-shift return, excluding the unavoidable first-phase discovery cost; and
- self-created core memory ids lost after the shift.

## Frozen development run

Development trains encoders with seeds `1320-1321`. Each is evaluated on ten
paired lifetimes using seeds:

```text
115,000,000-115,000,009
115,001,000-115,001,009
```

No result may change the maturity support count, mechanism, comparisons,
thresholds, capacities, or criteria.

## Frozen development gate

Fresh-seed confirmation is triggered only if learned-cold mature Chevron:

- promotes at least 7.5 of eight initial identities before the shift;
- has at least seven mature initial slots at the shift;
- reaches at least 0.75 on the shift-time core probe;
- retains at least 0.85 stable accuracy;
- reaches at least 0.70 reversed and novel final probes;
- promotes at least three post-shift novel identities and recovers at least
  three reversed policy identities;
- has identity and policy residual calibration of at least 0.10;
- makes zero mature-slot evictions, duplicate allocations,
  preloaded-established overwrites, under-supported writes, and self-created
  core-slot losses;
- is non-inferior to the learned cold baseline in post-shift return, retention,
  and novel probe, with paired 95% lower bounds above -0.05;
- is non-inferior to immediate protection in post-shift return and novel probe,
  with paired 95% lower bounds above -0.05;
- is non-inferior to learned preloaded protected Chevron in post-shift return
  with a paired 95% lower bound above -0.08 and retention above -0.05;
- is non-inferior to mature direct adaptation in retention with a lower bound
  above -0.05; and
- exceeds mature direct adaptation in novel probe with a paired 95% lower bound
  above zero, while keeping its total-return lower bound above -0.15.

If every criterion passes, confirmation will train ten untouched encoders with
seeds `1330-1339`. Each will be evaluated on twenty paired lifetimes using seed
blocks `116,000,000-116,000,019` through
`116,009,000-116,009,019`. The full protocol will remain frozen.

## Interpretation

Passing would show that a self-created core can acquire protection through use,
without a phase boundary or oracle declaration of which memories constitute
the core. It would justify moving to the rooms-and-routes RL environment with
the encoder frozen during the first visual policy experiment.

Failure would distinguish several possibilities. Too few mature core slots
would indicate an unsuitable maturity signal. Poor novelty under immediate and
four-use protection would indicate that fixed permanent capacity is itself the
bottleneck. Continued core loss despite zero mature eviction would indicate
that the hidden-category audit and learned identity geometry disagree in a way
that eviction protection alone cannot solve.

## Development outcome

The frozen development gate failed four criteria, so confirmation was not run.

The targeted mechanism worked. The unprotected cold baseline lost 0.30
self-created core slots per lifetime on these seeds; four-use maturity lost
zero and evicted zero mature slots. Post-shift return changed by +0.004,
approximate paired 95% interval -0.002 to +0.009. Retention changed by +0.001,
interval -0.008 to +0.009, and novel probes were identical at 0.775.

Mature Chevron promoted 7.55 initial identities, matured 6.25 by the exact
shift boundary, scored 0.906 on the core probe, retained 0.889 accuracy, and
finished with 0.925 reversed and 0.775 novel probes. It promoted 3.0 novel
identities and recovered 3.55 reversed policies.

The failed criteria were mature core slots at the shift, retention
non-inferiority to preloaded memory, retention non-inferiority to direct
adaptation, and a strictly positive novel-probe advantage over direct
adaptation. The last three missed their confidence boundaries by 0.009, 0.003,
and 0.005 respectively.

Immediate protection also prevented core loss but averaged 2.95 allocation
deferrals and slightly lower novelty. Four-use maturity averaged zero deferrals
and only 0.15 immature evictions. The result supports earned protection over
blanket permanence, but only as development evidence.
