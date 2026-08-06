# The Mathematics of Chevron Attention, Version 2

Status: working mathematical specification for the first Chevron Agent
experiments.

This version refines the original scheme in four ways:

1. residual mass is interpreted as **unresolved evidence**, not intrinsically
   as novelty;
2. admission to the current read is separated from permission to write;
3. rejected mass is retained per slot before being aggregated; and
4. unresolved evidence enters provisional storage before it can create or
   replace long-term memory.

The central factorisation remains:

```text
retrieval != read admission != write permission
```

## 1. State and memory

At time `t`, the agent has a fast adaptive state `A_t`. Each established memory
slot `j` contains:

- `A_mem[j]`: an adaptive address trace used for retrieval;
- `N_mem[j]`: slower retained content;
- provenance, usage, age, and importance metadata.

The agent also has a small provisional buffer `P_t` for unresolved A-derived
evidence. Provisional entries are not yet established N memories, although they
may influence behaviour through a separately identified provisional read.

## 2. Retrieval

Retrieval asks which established memory is relevant to the current fast state:

$$
\alpha_{tj}
=
\operatorname{softmax}_j\left(
\frac{
Q_A(A_t^{\mathrm{address}})
K_A(A_{\mathrm{mem},j})^\top
}{\sqrt{d_k}}
\right).
$$

Therefore:

$$
\alpha_{tj}\ge 0,
\qquad
\sum_j\alpha_{tj}=1.
$$

Retrieval uses broad address cues. It does not by itself determine whether the
retained content fits the current evidence.

The symbol $\alpha$ is reserved for retrieval. The symbol $\rho$ should remain
available for ART-style vigilance.

## 3. Assent and read admission

Assent uses a computation that is asymmetric with retrieval. Current diagnostic
evidence and retained N content are projected into a comparison space:

$$
M_{tj}
=
M\left(
E_A(A_t^{\mathrm{evidence}}),
E_N(N_{\mathrm{mem},j})
\right).
$$

`M` should be bounded or normalised. The first experiments should compare
cosine distance with normalised L2 distance.

Read assent is:

$$
r^{\mathrm{read}}_{tj}
=
\sigma\left(
k_r(\theta_r-M_{tj})
\right),
\qquad k_r>0.
$$

Larger mismatch lowers assent. On this convention, increasing $\theta_r$ makes
the gate more permissive because it tolerates a larger mismatch.

The admitted read mass is:

$$
w^{\mathrm{read}}_{tj}
=
\alpha_{tj}r^{\mathrm{read}}_{tj}.
$$

This answers:

> This memory is relevant, but may its retained content influence the current
> decision?

## 4. Per-slot and total residual mass

The mass retrieved from slot `j` but not admitted is:

$$
u_{tj}
=
\alpha_{tj}\left(1-r^{\mathrm{read}}_{tj}\right).
$$

The vector $u_t$ retains the source of the unresolved evidence. It distinguishes
one strong local conflict from diffuse rejection across several slots.

Total residual mass is:

$$
q_t
=
\sum_j u_{tj}.
$$

Because retrieval mass sums to one:

$$
q_t
=
1-\sum_jw^{\mathrm{read}}_{tj}.
$$

Consequently:

$$
0\le q_t\le1,
\qquad
\sum_jw^{\mathrm{read}}_{tj}+q_t=1.
$$

The correct interpretation is:

```text
q_t = total unassented retrieval mass
```

A high $q_t$ may result from novelty, contradiction, ambiguity, inadequate
memory, poor representation, or a badly calibrated gate. Its behavioural
meaning must be learned or established experimentally.

## 5. Read output

The established-memory read is:

$$
z^N_t
=
\sum_j
w^{\mathrm{read}}_{tj}
V_N(N_{\mathrm{mem},j})
+q_tV_{\mathrm{null}}.
$$

For the first diagnostic experiments:

$$
V_{\mathrm{null}}=0.
$$

The scalar $q_t$ should be passed separately to the policy. This prevents a
learned null vector from becoming an unrestricted alternative route around the
intended memory mechanism.

If provisional evidence is allowed to guide current behaviour, it is read
through a separately labelled route:

$$
z^P_t
=
\operatorname{ProvisionalRead}(A_t,P_t).
$$

The policy and value functions may then receive:

$$
(\text{policy logits}_t,\,V_t)
=
f_{\mathrm{actor\text{-}critic}}
\left(A_t,z^N_t,z^P_t,q_t\right).
$$

Provisional influence must remain distinguishable from established N assent.

## 6. Write permission

Safe use does not imply permission to learn:

```text
read admission != write permission
```

A conservative smooth write-assent gate is:

$$
r^{\mathrm{write}}_{tj}
=
\sigma\left(
k_w(\theta_w-M_{tj})
\right),
\qquad k_w>0.
$$

If the read and write gates use the same mismatch scale, stricter write
permission means:

$$
\theta_w<\theta_r.
$$

An optional eligibility term $e_{tj}\in[0,1]$ can represent persistence,
provenance, retrospective outcome agreement, or whether the observation adds
useful information. The final write permission is:

$$
g^{\mathrm{write}}_{tj}
=
\alpha_{tj}
r^{\mathrm{write}}_{tj}
e_{tj}.
$$

A hard experimental alternative is:

$$
g^{\mathrm{write}}_{tj}
=
\alpha_{tj}r^{\mathrm{read}}_{tj}
\mathbb{I}
\left[r^{\mathrm{read}}_{tj}>\tau_w\right],
$$

but the smooth form is preferable for end-to-end learning.

## 7. Convex retained-memory update

An established N slot updates as:

$$
N_{\mathrm{mem},j}
\leftarrow
\left(1-\eta_Ng^{\mathrm{write}}_{tj}\right)
N_{\mathrm{mem},j}
+
\eta_Ng^{\mathrm{write}}_{tj}
T_N(A_t).
$$

Require:

$$
0\le\eta_Ng^{\mathrm{write}}_{tj}\le1
$$

so that the update remains a convex interpolation rather than an extrapolation.

An address trace may update more quickly:

$$
A_{\mathrm{mem},j}
\leftarrow
\left(1-\eta_Ag^A_{tj}\right)
A_{\mathrm{mem},j}
+
\eta_Ag^A_{tj}
T_A(A_t),
$$

with $\eta_A>\eta_N$. However, $g^A$ should not freely rewrite an address when
the corresponding N update was rejected. Otherwise retrieval may drift toward
evidence that the retained content does not represent.

## 8. Provisional-allocation trigger

A simple diagnostic allocation trigger is:

$$
a_t
=
\mathbb{I}\left[
q_t>\tau_q
\;\land\;
\max_j w^{\mathrm{read}}_{tj}<\tau_a
\right].
$$

The second condition prevents allocation when one established memory still has
a strong admitted claim. An alternative is to check assent only among slots
with substantial retrieval mass:

$$
\max_{j:\alpha_{tj}>\tau_\alpha}
r^{\mathrm{read}}_{tj}<\tau_r.
$$

An unweighted $\max_j r_{tj}$ should not be used: an irrelevant slot with
$\alpha_{tj}\approx0$ and high assent could incorrectly block allocation.

The initial response to $a_t=1$ is provisional storage, not permanent N
allocation:

$$
P_t
\leftarrow
\operatorname{ProvisionalUpdate}
\left(
P_{t-1},
T_A(A_t),
q_t,
u_t,
\text{provenance}_t
\right).
$$

The provisional entry should track at least:

- content and address;
- support or persistence;
- internal coherence;
- provenance;
- age and recency;
- nearby rejected N slots, derived from $u_t$;
- retrospective outcomes when available.

## 9. Consolidation and permanent allocation

A provisional entry `l` may be proposed for promotion only after declared
criteria are satisfied, for example:

$$
\operatorname{promote}_l
=
\mathbb{I}\left[
s_l\ge L
\;\land\;
c_l\ge\tau_c
\;\land\;
h_l\ge\tau_h
\right],
$$

where $s_l$ is persistent support, $c_l$ is coherence, and $h_l$ is an optional
outcome or provenance quality measure.

Promotion should select storage in this order:

1. a free permanent slot;
2. an explicitly evictable slot with low long-term importance;
3. no promotion if no safe slot exists.

Current irrelevance is not an eviction criterion. An unrelated memory may be
irrelevant now but important later. Protected self, commitment, or safety slots
must not be overwritten by ordinary allocation.

On accepted promotion into slot `p`:

$$
A_{\mathrm{mem},p}
\leftarrow
T_A(P_l),
\qquad
N_{\mathrm{mem},p}
\leftarrow
T_N(P_l).
$$

The exact consolidation schedule remains experimental. It may be based on
fixed persistence, episode boundaries, event-triggered rest, replay, or learned
control.

## 10. Why the provisional buffer is structural

Without provisional storage, rejected evidence has only unattractive outcomes:

- disappear into null;
- be forced into an incompatible N slot;
- cause immediate category proliferation; or
- create optimisation pressure for assent to imitate retrieval.

The buffer lets the system reject an established interpretation without
discarding the experience. It therefore supports, but does not mathematically
guarantee, the separation between retrieval and assent.

The current scheme is:

```text
A address cues -> alpha -> established candidate
A evidence + N content -> r_read -> admit or reject
rejected per-slot mass -> u
total unresolved mass -> q
unresolved A evidence -> P
stricter r_write + eligibility -> established N update
persistent coherent P -> promotion into N
```

## 11. Preventing assent from becoming retrieval twice

The factorisation is meaningful only if retrieval and assent are genuinely
different computations.

Required safeguards:

- retrieval uses A address traces and never reads retained N content;
- assent compares diagnostic A evidence with retained N content;
- retrieval and assent have separate projections and inputs;
- the training task dissociates broad address cues from match-defining cues;
- causal interventions can vary N compatibility while holding retrieval cues
  fixed;
- the candidate buffer preserves rejected evidence rather than rewarding a
  forced match.

Different parameter matrices alone are not sufficient. The two computations
must answer different questions using different information.

## 12. Causal tests of the factorisation

The first mathematical experiments should include:

1. **Fixed address, changed content.** Hold the A address cue fixed and swap N
   content. Expected: $\alpha$ remains stable while $r^{\mathrm{read}}$
   changes.
2. **Changed address, fixed content.** Change the address cue while holding N
   content fixed. Expected: $\alpha$ changes independently of the content
   comparison.
3. **Second-retrieval control.** Replace assent with a second dot product over
   the retrieval features. Expected: it fails tasks requiring the dissociated
   N comparison.
4. **No-buffer control.** Remove provisional storage and measure whether assent
   saturates, shadows retrieval, or routes all unresolved evidence through
   null.
5. **Write-gate ablation.** Replace $g^{\mathrm{write}}$ with $\alpha$ and
   measure contamination of adjacent established memories.
6. **Read/write coupling ablation.** Set
   $g^{\mathrm{write}}=w^{\mathrm{read}}$ and test whether the stricter write
   gate improves retention without preventing adaptation.

Useful diagnostics include conditional correlations between $\alpha$ and $r$,
intervention responses, slot drift, false allocation, false consolidation, and
retained-task performance.

## 13. Initial implementation defaults

For the first controlled experiments:

- use one Chevron Attention head;
- use normalised comparison representations;
- use $V_{\mathrm{null}}=0$ and expose $q_t$ separately;
- use smooth read and write gates;
- enforce $\theta_w<\theta_r$;
- use a small provisional buffer;
- do not allocate permanent N from one observation;
- prefer a free slot and forbid automatic eviction of protected slots;
- log $\alpha$, $r^{\mathrm{read}}$, $w^{\mathrm{read}}$, $u$, $q$,
  $r^{\mathrm{write}}$, $g^{\mathrm{write}}$, and every provisional update;
- verify all conservation and convexity invariants in unit tests.

## 14. Claim boundary

This mechanism does not prove that residual mass is novelty, that every
mismatch deserves search, or that Chevron Agent implements the whole of
Adaptive Resonance Theory.

The testable architectural claim is narrower:

> Chevron Attention factorises candidate retrieval, current read admission,
> and retained-memory write permission. Per-slot residuals preserve the source
> of unassented evidence, while provisional storage allows that evidence to
> remain behaviourally useful without immediately modifying established
> memory.

Whether this factorisation improves persistent RL behaviour is the subject of
the Chevron Agent experiments.
