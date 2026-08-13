# Experiment 009: frozen dual-relation assent protocol

## Question

Does Chevron work better when memory identity and policy compatibility remain
separate relations rather than being collapsed into one cosine gate?

Experiment 008 showed that consequence similarity alone is too coarse for
memory identity. Experiment 009 tests the resulting architectural hypothesis
directly, using fixed meaningful geometries before attempting to learn them.

## Task

The task retains four broad address families, each with two established
contexts and one novel context. Outcomes remain delayed by three decisions.

After step 200, two kinds of change occur together:

1. the four novel contexts begin to appear; and
2. one established context in each family changes its correct action while
   retaining the same identity.

The other four established contexts remain stable. The agent must therefore
distinguish a new situation from a familiar situation whose policy has become
invalid.

Permanent memory has twelve slots: exactly enough for eight established
identities and four novel identities. Creating duplicate identities for the
four policy reversals exceeds capacity and exposes the cost of collapsing the
two relations.

## Fixed representations

Identity uses the same 12-dimensional prototype geometry and observation-noise
distribution as Experiment 005a. Policy consequence is a normalised centred
four-action preference vector. Matching policies have cosine one; different
one-hot action preferences have cosine minus one third.

These supplied geometries make Experiment 009 a causal mechanism test, not a
representation-learning result.

## Dual computation

For established slot `j`:

```text
alpha_j = broad address retrieval
r_id_j = sigmoid(40 * (0.19 - M_id(A_id, N_id_j)))
w_id_j = alpha_j * r_id_j
q_id = 1 - sum_j w_id_j

r_policy_j = sigmoid(40 * (0.19 - M_policy(A_policy, N_policy_j)))
w_policy_j = w_id_j * r_policy_j
q_policy = sum_j w_id_j * (1 - r_policy_j) / (sum_j w_id_j + epsilon)
```

`q_id` controls new-identity allocation. Conditional `q_policy` controls policy
veto and revision for an already recognised identity. Current action reads use
`w_policy`; an incompatible retained policy cannot guide action merely because
its identity matches.

Both novelty and policy revision enter the same capacity-four provisional
buffer, but entries retain their type. Two coherent positive delayed outcomes
are required before consolidation:

- a new-identity candidate creates a new permanent slot;
- a policy-revision candidate updates the policy content and action values of
  the recognised existing slot without creating a new identity.

## Conditions

1. Dual relation + buffer: separate identity allocation and policy revision.
2. Collapsed relation + buffer: multiplies identity and policy assent, but
   interprets all rejected mass as a new identity and therefore allocates a
   duplicate for a policy reversal.
3. Identity-only + buffer: recognises identity but has no pre-action policy
   veto; it may adapt action values retrospectively from reward.
4. Dual relation + immediate revision: writes a sampled policy before delayed
   outcome, removing provisional protection.

All conditions receive the same identities, policy signal, delayed rewards,
memory capacity, action sampler, and update rates. Identity-only deliberately
ignores the policy signal as its ablation.

## Development and confirmation

Development uses twenty paired lifetimes with seeds 90,000,000–90,000,019.
No parameter is trained or selected.

Fresh-seed confirmation is triggered only if dual buffered Chevron:

- reaches at least 0.95 final stable-context accuracy;
- reaches at least 0.75 final reversed-context and novel-context accuracy;
- reaches at least 0.15 calibration for both identity and policy residuals;
- consolidates at least three of four novel identities and three of four policy
  revisions on average;
- makes no premature permanent writes or established-identity overwrites;
- has paired return intervals with lower bounds above zero versus collapsed,
  identity-only, and immediate revision; and
- has a paired novel-acquisition interval with lower bound above zero versus
  collapsed allocation, while creating no duplicate permanent identities for
  policy reversals.

If triggered, confirmation uses 100 untouched lifetimes with seeds
91,000,000–91,000,099. The equations, thresholds, capacities, task mixture, and
decision rules remain unchanged.

## Interpretation

Passing would support the dual-relation mechanism under supplied geometry and
justify learning identity from persistence while learning policy compatibility
from sampled consequences. It would not yet establish a visual or spatial
agent.

Failure would mean that merely separating the two residuals does not solve the
combined novelty/reversal problem under the current buffer mechanics. The next
step should inspect consolidation routing rather than representation learning.

## Development outcome

The frozen gate failed. Dual buffered Chevron reached 0.587 return, 0.951
stable accuracy, 0.826 reversed accuracy, and 0.598 novel accuracy. It beat the
collapsed condition by +0.047 return, with an approximate paired 95% interval
from +0.006 to +0.088, and improved reversal accuracy by +0.211. It also avoided
the collapsed condition's 5.85 duplicate allocations per lifetime.

However, identity-only buffering reached 0.746 return, 0.973 reversed accuracy,
and 0.802 novel accuracy. Direct retrospective value adaptation was more
efficient than conservative policy veto and two-positive-outcome revision in
this deterministic task. Dual novelty and revision candidates also competed in
one four-entry buffer, producing 78.85 evictions and only 2.85 of four novel
promotions on average.

Confirmation was not run. The next diagnostic should separate identity and
policy provisional queues before changing either representation or gate.
