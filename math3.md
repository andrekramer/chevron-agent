# Chevron Agent Mathematics and Architecture, Version 3

Status: architecture checkpoint updated through Experiment 010a.

This document supersedes `maths2.md` as the current working description of
Chevron Agent. It preserves the original retrieval, assent, residual, and
protected-write factorisation, then incorporates the main result of the later
experiments: recognising a situation and accepting its retained policy are two
different relations.

The compact form is:

```text
retrieve a possible identity
-> test identity assent
-> test policy assent, conditional on identity
-> act from admitted retained content or remain unresolved
-> hold unresolved evidence in a typed provisional bank
-> consolidate only after delayed outcome support
```

The fixed geometries used in the experiments are an existence proof for the
mechanism, not yet a learned perceptual solution.

## 1. Architectural roles

At time `t`, the agent has three memory timescales.

### Fast adaptive state A

`A_t` is the current, continuously changing state derived from observation,
action, and recent experience. For the present architecture it exposes
different views:

- `A_t^addr`: broad address cues used for retrieval;
- `A_t^id`: evidence about situation identity;
- `A_t^pi`: evidence about the policy or consequence currently appropriate.

These views may later share a backbone, but the computations that consume them
must remain causally distinguishable.

### Retained memory N

Each established slot `j` contains:

```text
N_j = {
  address,
  identity,
  policy,
  action values or value content,
  provenance and protection metadata
}
```

Identity answers, "what situation is this?" Policy content answers, "what
should be done in this situation?" A familiar identity can therefore survive
while its policy is revised.

### Provisional bank P

`P_t` holds unresolved A-derived evidence while delayed outcomes are pending.
It is neither ordinary established N nor merely another permanent A address.
It is a short-lived bridge between fast experience and retained memory.

Its content may influence behaviour through an explicit, limited provisional
route. In Experiments 009-009b it did not blend directly into the established
read: a routed candidate instead caused cautious exploratory action until
outcomes supported consolidation.

## 2. Retrieval proposes; it does not assent

Retrieval uses broad A-side address cues:

$$
\alpha_{tj}
=
\operatorname{softmax}_j\left(
\frac{Q_A(A_t^{addr})K_A(A^{mem}_j)^\top}{\sqrt{d_k}}
\right).
$$

Therefore:

$$
\alpha_{tj}\geq 0,
\qquad
\sum_j \alpha_{tj}=1.
$$

`alpha` says which memories are relevant enough to inspect. It does not say
that their identity or retained policy should be accepted.

The fixed Experiment 009 implementation used a deliberately broad address:
uniform mass over established slots in the observed address family. This
isolated the assent and consolidation mechanisms from representation learning.

## 3. Identity assent

Identity mismatch is computed from current identity evidence and the retained
identity of each proposed slot:

$$
M^{id}_{tj}
=
M_{id}\left(E^{id}_A(A_t^{id}),E^{id}_N(N^{id}_j)\right).
$$

The fixed experiments used half-cosine distance:

$$
M^{id}_{tj}
=
\frac{1-\cos(A_t^{id},N_j^{id})}{2}.
$$

Identity assent is:

$$
r^{id}_{tj}
=
\sigma\left(k_{id}(\theta_{id}-M^{id}_{tj})\right).
$$

The admitted identity mass and rejected identity mass are:

$$
w^{id}_{tj}=\alpha_{tj}r^{id}_{tj},
$$

$$
u^{id}_{tj}=\alpha_{tj}(1-r^{id}_{tj}).
$$

Total unresolved identity mass is:

$$
q^{id}_t
=
\sum_j u^{id}_{tj}
=
1-\sum_j w^{id}_{tj}.
$$

This retains both the scalar unresolved quantity `q_id` and the per-slot
rejection vector `u_id`. A high `q_id` means the retrieved established
identities did not explain the current evidence. It is evidence for possible
new identity, not proof of novelty.

Identity conservation is exact:

$$
\sum_j w^{id}_{tj}+q^{id}_t=1.
$$

## 4. Policy assent is conditional on identity

Policy assent asks a second question only after identity mass has been
admitted:

$$
M^{\pi}_{tj}
=
M_{\pi}\left(E^{\pi}_A(A_t^{\pi}),E^{\pi}_N(N^{\pi}_j)\right),
$$

$$
r^{\pi}_{tj}
=
\sigma\left(k_{\pi}(\theta_{\pi}-M^{\pi}_{tj})\right).
$$

The final admitted policy mass is:

$$
w^{\pi}_{tj}
=
w^{id}_{tj}r^{\pi}_{tj}.
$$

Policy-rejected mass remains attached to its recognised identity:

$$
u^{\pi}_{tj}
=
w^{id}_{tj}(1-r^{\pi}_{tj}).
$$

Define the raw policy-rejected mass:

$$
Q^{\pi}_t=\sum_j u^{\pi}_{tj}.
$$

For routing, it is useful to normalise this by admitted identity mass:

$$
q^{\pi}_t
=
\begin{cases}
\dfrac{Q^{\pi}_t}{\sum_jw^{id}_{tj}},
& \sum_jw^{id}_{tj}>\epsilon,\\
0, & \text{otherwise.}
\end{cases}
$$

`Q_policy` is actual residual mass. `q_policy` is the conditional fraction of
recognised-identity mass whose policy was rejected. They should not be
conflated.

The full two-stage conservation law is:

$$
\sum_j w^{\pi}_{tj}+Q^{\pi}_t+q^{id}_t=1.
$$

This decomposition distinguishes three states:

- admitted identity and policy: `w_policy`;
- admitted identity but rejected policy: `Q_policy`;
- rejected identity: `q_id`.

Collapsing the last two into a single null mass loses the difference between a
new situation and a familiar situation requiring policy revision.

## 5. Read and action output

The established-memory read is:

$$
z^N_t
=
\sum_j w^{\pi}_{tj}V_N(N_j)
+
(q^{id}_t+Q^{\pi}_t)V_{null}.
$$

For diagnostic experiments, `V_null = 0` and the residuals are passed
separately to the decision and routing logic. This prevents a learned null
vector from becoming an unrestricted bypass around the memory mechanism.

Experiment 009 computed action scores as:

$$
s_t(a)=\sum_jw^{\pi}_{tj}N_j^{value}(a),
$$

and selected the maximum-scoring action when no provisional candidate was
triggered. When identity or policy evidence was unresolved, it sampled an
exploratory action and awaited the delayed outcome. A later agent can learn a
cautious policy from `(A_t, z_t^N, q_id, q_policy)` without changing the memory
factorisation.

## 6. Typed routing into one shared provisional bank

The two residuals have different meanings and different consolidation
destinations.

One useful hard routing rule, matching the fixed experiments, is:

$$
new\_identity_t
=
\mathbb{I}\left[
q^{id}_t>\tau_{id}
\land
\max_jw^{id}_{tj}<\tau_{admit}
\right],
$$

$$
policy\_revision_t
=
\mathbb{I}\left[
\max_jw^{id}_{tj}\geq\tau_{admit}
\land
q^{\pi}_t>\tau_{\pi}
\right].
$$

The second branch is evaluated only when an established identity has a strong
claim. A policy mismatch must not allocate a duplicate identity.

Both candidate types enter one flexible shared bank, but every entry retains
its type and destination:

```text
P_l = {
  candidate_id,
  kind: new_identity | policy_revision,
  broad address family,
  target_memory_id if revising,
  averaged identity evidence,
  averaged policy evidence,
  pending event ids,
  positive outcome support by action,
  observation count,
  last_seen
}
```

Compatible observations merge only when candidate type, broad address, and
revision target agree, and their identity similarity clears the match
threshold. At capacity, the tested bank evicts the least recently seen
provisional entry. It never evicts an established slot merely to make room for
unresolved evidence.

Experiments 009a and 009b support one shared typed bank rather than rigidly
partitioned identity and policy queues. Types remain separate; capacity is
pooled.

## 7. Delayed outcome and consolidation

An unresolved observation creates a pending record:

$$
pending_t=(P_l,a_t,event_t).
$$

When its delayed reward arrives, a positive outcome increments support for the
action that was taken:

$$
h_l(a_t)\leftarrow h_l(a_t)+\mathbb{I}[R_{t+d}>0].
$$

The tested promotion condition was:

$$
\max_a h_l(a)\geq L,
\qquad L=2.
$$

Thus two coherent positive delayed outcomes for one action were required. A
single uncertain event could influence exploration but could not write
permanent memory.

Consolidation is typed.

For a new identity candidate:

```text
re-run identity assent on the aggregated candidate
```

Provisional evidence can change as observations accumulate. A candidate that
looked novel when it entered the bank may match an established identity after
averaging. Permanent allocation is permitted only if the aggregated candidate
still fails to match established identity memory. Otherwise it is reconciled
with the matched identity or discarded; it must not create a duplicate slot.

Only after this pre-consolidation revalidation passes:

$$
N_p^{id}\leftarrow T_{id}(P_l),
\qquad
N_p^{\pi}\leftarrow T_{\pi}(P_l),
\qquad
N_p^{value}\leftarrow onehot(\arg\max_a h_l(a)).
$$

A free permanent slot is preferred. Replacement must be restricted to
explicitly evictable, unprotected slots; the experiments supplied exactly
enough permanent capacity for the true identities.

For a policy revision candidate targeting established memory `j`:

$$
N_j^{id}\leftarrow N_j^{id},
$$

$$
N_j^{\pi}\leftarrow T_{\pi}(P_l),
\qquad
N_j^{value}\leftarrow onehot(\arg\max_a h_l(a)).
$$

The memory keeps its identity and memory id. Only policy content and action
values change.

## 8. Read admission and write permission remain distinct

The provisional promotion rule above was the operative write-protection
mechanism in Experiments 009-009b. In a differentiable implementation,
compatible incremental writes may also use a stricter smooth gate:

$$
r^{write}_{tj}
=
\sigma\left(k_w(\theta_w-M_{tj})\right),
$$

$$
g^{write}_{tj}
=
\alpha_{tj}r^{write}_{tj}e_{tj},
\qquad e_{tj}\in[0,1].
$$

With a shared mismatch scale and slope, require:

$$
\theta_w<\theta_r
$$

so write permission is stricter than read admission. More generally enforce
`r_write <= r_read` pointwise.

A safe convex update is:

$$
N_j\leftarrow
(1-\eta_Ng^{write}_{tj})N_j
+\eta_Ng^{write}_{tj}T_N(A_t),
$$

with:

$$
0\leq\eta_Ng^{write}_{tj}\leq1.
$$

Any learned address update must be gated too. Otherwise an observation can be
rejected as content while still silently rewriting where that content will be
retrieved.

The key rule is:

```text
permission to retrieve != permission to read != permission to write
```

## 9. Fixed experimental instantiation

The following values describe the confirmed mechanism test, not universal
Chevron hyperparameters.

| Quantity | Experiment 009b value |
|---|---:|
| Broad address families | 4 |
| Actions | 4 |
| Established identities before shift | 8 |
| Novel identities after shift | 4 |
| Permanent slots | 12 |
| Stream length | 600 |
| Task shift | step 200 |
| Outcome delay | 3 decisions |
| Identity dimension | 12 |
| Cosine similarity boundary | 0.62 |
| Equivalent half-cosine mismatch threshold | 0.19 |
| Sigmoid slope | 40 |
| Identity residual trigger | 0.80 |
| Conditional policy residual trigger | 0.80 |
| Minimum admitted identity mass | 0.25 |
| Promotion support | 2 positive outcomes for one action |
| Confirmed shared provisional capacity | 8 |

Identity used fixed noisy 12-dimensional prototypes. Policy used a normalised,
centred four-action preference vector: matching policies had cosine `1`, while
different one-hot preferences had cosine `-1/3`. These supplied
representations made the experiment a mechanism test.

Capacity eight is not proposed as a general constant. It was sufficient for
the combined novelty-and-reversal traffic in this task. Capacity four was not.
Future capacity should be related to unresolved arrival rate, outcome delay,
candidate persistence, and consolidation throughput.

## 10. Empty memory, top-k, and numerical rules

An implementation should make the following cases explicit.

### Empty established memory

If no N slot exists:

```text
alpha = empty
q_id = 1
Q_policy = 0
q_policy = 0
route = new_identity
```

The first evidence enters provisional storage rather than forcing an invalid
softmax.

### Top-k retrieval

If top-k retrieval discards address mass, that mass must enter a named residual
or alpha must be renormalised with the semantic consequence documented. It
must not disappear. The conservation test applies after the chosen policy.

### Bounds

Clamp and test:

$$
\alpha,r,w,u,q,g,e\in[0,1].
$$

Normalise identity and policy embeddings before cosine comparison. Define the
zero-norm fallback. Enforce a positive read/write threshold margin and verify
it over the full bounded mismatch interval.

## 11. Protection and conservation invariants

The minimum invariant suite is:

1. `sum(alpha) = 1` for non-empty memory.
2. `sum(w_id) + q_id = 1` within numerical tolerance.
3. `sum(w_policy) + Q_policy + q_id = 1` within tolerance.
4. `q_policy` is conditional and is zero when no identity mass is admitted.
5. A pure policy mismatch cannot create a new permanent identity.
6. A policy revision preserves identity, address, memory id, and protected
   metadata.
7. No permanent write occurs before its declared delayed evidence criterion.
8. Established protected slots are never overwritten by ordinary allocation.
9. Candidate kinds remain typed even though capacity is shared.
10. Every candidate removal is explained by promotion, explicit expiry, or a
    counted provisional eviction.
11. Address updates are gated by write permission.
12. Read and write gates satisfy their declared ordering.
13. A new-identity candidate reruns identity assent immediately before
    permanent allocation; a candidate that now matches N cannot create a
    duplicate slot.

The Experiment 009b confirmation recorded zero premature permanent writes,
zero established overwrites, and zero duplicate identity allocations.

Experiment 010a directly tested invariant 13 on 100 untouched lifetimes. The
original path created one duplicate identity; promotion-time revalidation
intercepted exactly that candidate and produced zero duplicates without a
measurable performance cost.

## 12. Gradient boundaries for a learned agent

Experiments 005a and 009b used fixed geometry, so no gradient crossed memory
mutation or candidate allocation. A learned version should declare boundaries
rather than relying on accidental autodiff behaviour.

A conservative initial policy is:

- gradients train observation, identity, and policy encoders through explicit
  losses;
- the retrieval and assent paths receive different diagnostic signals;
- discrete allocation, eviction, delayed support counts, and permanent memory
  mutation are stop-gradient state transitions;
- retrospective outcome or TD error may supervise policy assent without being
  allowed to redefine identity;
- provisional reads, if added, are separately labelled and bounded;
- policy optimisation cannot reduce loss merely by forcing every gate open or
  every residual to zero.

This can later be relaxed with differentiable memory methods, but the causal
factorisation should first survive the same fixed-address and fixed-content
interventions used in the early experiments.

## 13. What is experimentally supported

### Confirmed

- Retrieval and assent can be made causally different computations.
- With meaningful fixed geometry, the assent gate, residual mass, delayed
  provisional buffer, and separate write protection work together.
- Immediate writing is substantially worse under delayed evidence.
- A consequence representation alone is not a safe identity geometry.
- On the combined novelty-and-reversal task, a shared capacity-eight typed bank
  decisively outperformed the undersized capacity-four bank across 100 fresh
  lifetimes.
- The capacity-eight dual mechanism was non-inferior overall to rapid
  identity-only value adaptation, with better novel acquisition and slightly
  slower deterministic policy reversal.
- Promotion-time identity revalidation eliminated duplicate allocation in the
  Experiment 010a correction audit while preserving return, retention,
  reversal learning, and novel probes within tight paired bounds.

### Supported only as a development finding

- Separate identity and policy relations beat the collapsed relation and
  avoided its duplicate identities. Experiment 009 did not trigger its planned
  confirmation because the original capacity-four bank was overloaded.

### Fresh-seed evidence with a failed overall confirmation

- Experiment 010 removed the supplied policy signature and derived policy
  suspicion from delayed reward with 10% misleading outcomes. Across 100 fresh
  lifetimes, protected retrospective revision retained 0.948 stable accuracy,
  revised 3.96 of four changed policies, reached 0.614 policy residual
  calibration, and reduced false stable-policy revisions from 16.87 under
  immediate writing to 0.06.
- The Experiment 010 confirmation failed one of fourteen criteria because two
  provisional identity candidates became matches to established memory after
  averaging but were not revalidated before allocation. This exposed the new
  pre-consolidation identity invariant above.
- Experiment 010a subsequently tested that single correction on new untouched
  seeds and passed every criterion. It does not retroactively turn Experiment
  010 into a passing confirmation.

### Failed or not yet established

- Sparse reward did not learn the required comparison geometry in the first RL
  attempts.
- Temporal and action-predictive encoders improved geometry but did not meet
  every frozen criterion.
- Consequence geometry did not outperform temporal geometry and could not
  safely serve as identity by itself.
- Fixed split identity/policy queues did not improve a capacity-matched shared
  bank.
- The dual architecture has not yet been demonstrated with learned identity
  and policy representations in a visual, spatial, or embodied agent.

## 14. Current architectural claim

The strongest justified claim is narrow but useful:

> Under supplied meaningful identity and policy geometries, a Chevron agent can
> separate retrieval from assent, distinguish novel identities from invalid
> policies, hold both as typed unresolved evidence, and consolidate them to
> different destinations without premature writes or duplicate memories. A
> shared provisional bank must have enough capacity for the combined unresolved
> traffic.

This does not yet show a general agent or a coherently evolving self. It gives a
testable memory primitive for one: current experience can remain plastic while
retained identity and policy are protected, challenged separately, and revised
only through delayed evidence.

## 15. Next mathematical step

The next representation-learning experiment should keep promotion-time
identity revalidation, the two-residual typed bank, and protected consolidation
fixed while learning the two comparison spaces from different evidence:

```text
identity geometry <- persistence, recurrence, and temporal continuity
policy geometry   <- action-conditioned outcomes and retrospective value
```

Success requires more than return. It should preserve the conservation and
protection invariants, calibrate both residuals, avoid duplicate identities,
acquire novel contexts, revise familiar policies, and pass causal tests showing
that identity assent and policy assent have not collapsed into retrieval twice.
