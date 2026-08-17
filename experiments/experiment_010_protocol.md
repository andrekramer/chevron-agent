# Experiment 010: frozen retrospective-policy RL protocol

## Question

Can Chevron protect an established policy when policy mismatch must be inferred
from delayed, occasionally misleading reward rather than supplied as a
pre-action policy signature?

Experiment 009b confirmed the dual routing and shared typed bank under supplied
identity and policy geometries. Experiment 010 removes the supplied policy
geometry. It keeps fixed identity geometry so that policy discovery is the only
new problem.

This is a contextual-bandit RL mechanism test. Category ids, correct actions,
reward-noise flags, and context roles exist only in the audit runner. They are
not agent inputs.

## Task

There are four broad address families. Each contains:

- one established context whose action remains stable;
- one established context whose action reverses after the task shift; and
- one novel context introduced after the shift.

The agent receives only:

- a broad address family; and
- a noisy 12-dimensional identity observation.

It chooses one of four actions. Reward arrives three decisions later. The
underlying reward is `+1` for the correct action and `-1` otherwise, but its
sign is independently reversed on 10% of events. The agent therefore cannot
treat one negative result as proof that an established policy is invalid or
one positive result as proof that a new action is correct.

The 800-decision lifetime has three phases:

1. steps 0-199: the eight established contexts use their initial policies;
2. steps 200-599: four novel contexts appear and one established context per
   family reverses its policy; and
3. steps 600-799: stable original contexts return as a retention probe after
   novelty and policy interference.

Phase two samples novel, reversed, and stable contexts with probabilities 0.30,
0.35, and 0.35. Phases one and three sample their eligible established stable
contexts uniformly.

## Fixed identity mechanism

Identity retrieval and assent are inherited from Experiment 009b:

```text
alpha_j = uniform retrieval over slots in the observed address family
r_id_j = sigmoid(40 * (similarity_j - 0.62))
w_id_j = alpha_j * r_id_j
q_id = 1 - sum_j w_id_j
```

A new-identity candidate is routed when:

```text
q_id > 0.80 and max_j(w_id_j) < 0.25
```

New identities use the confirmed shared provisional bank and require two
positive delayed outcomes for the same action before promotion.

## Retrospective policy evidence

No policy signature is present in the observation. For a recognised identity,
the agent acts from retained action values and stores the selected memory id,
action, and predicted success in an eligibility record.

When delayed reward arrives, define observed success:

```text
y_t = 1 if reward_t > 0 else 0
```

and outcome surprise:

```text
surprise_t = abs(y_t - predicted_success_t)
```

An unexpected failure of an action whose predicted success was at least 0.60
creates or reinforces a typed `policy_revision` candidate for the recognised
memory. The confirmed identity is preserved.

For the protected condition, policy vigilance has two stages:

```text
q_policy = min(1, supported_failures / 2)
```

- one supported failure means suspicion but the retained action is retried;
- two supported failures veto the retained policy and initiate search;
- exploratory outcomes remain provisional; and
- two positive outcomes for the same alternative action are required before
  the retained policy is revised.

A later positive outcome for the incumbent action dismisses a suspicion. Once
search is active, alternatives are sampled in a balanced cycle. Policy
candidates and new-identity candidates share one capacity-eight bank while
retaining their type and consolidation destination.

## Conditions

1. **Direct value adaptation**: the strong simple control from Experiment 009b.
   It uses the same identity gate and novelty buffer but updates the selected
   established action value directly from every delayed reward.
2. **Protected retrospective Chevron**: two supported incumbent failures
   activate search and two positive alternative outcomes permit revision.
3. **Fast-veto Chevron**: one unexpected failure activates search, while two
   positive alternative outcomes are still required for revision.
4. **Immediate-write Chevron**: one unexpected failure activates search and
   one positive alternative outcome revises permanent policy. This retains the
   delay but removes repeated-evidence write protection.

All conditions receive identical context streams, reward-noise flags, initial
memories, identity geometry, permanent capacity, provisional capacity, and
action-value initialisation. Random action/search samplers use condition-local
generators derived from the paired lifetime seed.

## Metrics

Primary performance metrics:

- realised return under noisy reward;
- clean action accuracy using the hidden correct action for audit only;
- phase-three stable-context retention;
- late phase-two reversed and novel accuracy; and
- final stable, reversed, and novel policy probes.

Mechanism metrics:

- new-identity and policy-revision promotions;
- false policy revisions on stable contexts;
- occurrences from shift to correct policy revision;
- stable and reversed policy-alarm rates;
- policy residual calibration;
- buffer evictions;
- premature or under-supported permanent writes;
- established identity overwrites; and
- duplicate identity allocations.

Policy residual calibration is:

```text
mean(q_policy | unresolved reversed context)
- mean(q_policy | stable context)
```

## Frozen development gate

Development uses twenty paired lifetimes with seeds
`100,000,000-100,000,019`. No threshold or rule may be changed after inspecting
those results.

Fresh-seed confirmation is triggered only if protected retrospective Chevron:

- reaches at least 0.90 phase-three stable accuracy;
- reaches at least 0.75 final reversed and novel probe accuracy;
- promotes at least three of four novel identities and three of four policy
  revisions on average;
- has policy residual calibration of at least 0.10;
- averages no more than 0.25 false stable-context policy revisions;
- makes no under-supported writes, established-identity overwrites, or
  duplicate identity allocations;
- is non-inferior to direct value adaptation in realised return and clean
  accuracy with paired 95% lower bounds above `-0.08`;
- is non-inferior to direct value adaptation in stable retention with paired
  95% lower bound above `-0.03`; and
- has fewer false policy revisions than immediate-write Chevron, with the
  paired 95% upper bound below zero.

If every criterion passes, confirmation will use 100 untouched lifetimes with
seeds `101,000,000-101,000,099` and the complete protocol will remain frozen.

## Interpretation

Passing would show that the protected policy path remains useful when mismatch
must be discovered from delayed stochastic outcomes. It would justify learning
identity geometry next and then moving into a small visual RL environment.

Failure would not invalidate the fixed-geometry Chevron mechanism. It would
show that the present retrospective evidence accumulator is too slow, too
fragile, or unnecessary relative to direct value adaptation. The failed
criterion should determine whether the next step is simplification, a better
outcome model, or stopping the dual-policy branch.

## Development and confirmation outcome

All fourteen frozen development criteria passed across seeds
`100,000,000-100,000,019`, so the predeclared confirmation was run without
changing the mechanism.

On 100 untouched lifetimes, protected retrospective Chevron passed thirteen of
fourteen confirmation criteria. It reached 0.948 stable retention, 0.980
reversed probe accuracy, 0.915 novel probe accuracy, and 0.614 policy residual
calibration. It made no under-supported writes or established-memory
overwrites. Its return was 0.538 versus 0.580 for direct value adaptation,
within the frozen non-inferiority margin.

Confirmation nevertheless failed because two noisy provisional candidates
were allocated as duplicate identities. Both candidates matched an established
identity above the 0.62 boundary after their evidence had been averaged, but
identity novelty was not rechecked at promotion. The result requires a narrow
new experiment adding pre-consolidation identity revalidation; this
confirmation must not be renamed or counted as passing.
