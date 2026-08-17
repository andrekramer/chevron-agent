# Experiment 012: frozen empty-memory bootstrap protocol

## Question

Can the confirmed learned-identity Chevron mechanism construct the memory core
that it later protects, rather than beginning with established identities and
policies already installed?

Experiment 011a confirmed a pairwise temporal identity encoder with preloaded
established memory. Experiment 012 removes that remaining initial-memory
oracle. It is the final synthetic bridge before a small spatial RL task.

## Fixed representation

The nonlinear sensor and pairwise temporal encoder are exactly those confirmed
in Experiment 011a:

- deterministic sensor seed 606;
- normalised 12-to-48-to-12 residual MLP;
- 750 AdamW steps;
- 128 temporal pairs, or 256 observations, per step;
- independently noisy views with noise 0.15;
- symmetric InfoNCE at temperature 0.10;
- learning rate 0.003 and weight decay 0.00001; and
- gradient clipping at 1.0.

The encoder is pretrained on newly sampled persistence pairs and then frozen
for the complete agent lifetime. It receives no task identities, actions,
rewards, policies, or correctness signals.

This experiment tests memory bootstrap, not simultaneous representation drift.

## Task

The Experiment 010a lifetime remains fixed:

- four address families;
- eight contexts present during decisions 0-199;
- four novel contexts and four familiar policy reversals during decisions
  200-599;
- a stable retention phase during decisions 600-799;
- four actions;
- reward delayed by three decisions and independently reversed on 10% of
  events; and
- noisy 12-dimensional identity observations behind the frozen sensor.

For cold-start conditions, permanent memory `N` begins with zero slots. Every
initial identity must therefore enter the shared provisional bank, accumulate
two positive delayed outcomes for one action, pass promotion-time identity
revalidation, and allocate its own permanent slot. The successful action
initialises that slot's retained policy.

The same rules apply after the shift. There is no special bootstrap write path,
phase-boundary bulk copy, or oracle installation of the initial policies.

## Fixed Chevron mechanism

All accepted rules remain unchanged:

```text
r_id_j = sigmoid(40 * (cosine(z_t, N_j_identity) - 0.62))
w_id_j = alpha_j * r_id_j
q_id = 1 - sum_j(w_id_j)
```

- new-identity routing requires `q_id > 0.80` and maximum admitted mass below
  0.25;
- identity and policy candidates share one typed capacity-eight provisional
  bank;
- new identities require two positive outcomes for one action;
- protected policy search requires two unexpected incumbent failures;
- protected policy revision requires two positive alternative outcomes; and
- identity novelty is revalidated immediately before permanent allocation.

An empty memory is defined explicitly as zero admitted mass and `q_id = 1`.
The observation is routed to a new-identity candidate without attempting to
stack, index, or read a nonexistent slot.

## Conditions

1. **Oracle-preloaded protected**: supplied latent identity and the eight
   established memories installed at lifetime start.
2. **Learned-preloaded protected**: confirmed pairwise identity encoder with
   the eight established memories installed. This is the Experiment 011a
   reference.
3. **Oracle-cold protected**: latent identity geometry but no permanent memory
   at lifetime start. This isolates bootstrap from representation error.
4. **Raw-sensor cold protected**: distorted sensor geometry and empty memory.
5. **Learned-cold protected**: the proposed condition—confirmed pairwise
   identity encoder, empty memory, protected retrospective policy revision.
6. **Learned-cold direct adaptation**: the same learned encoder, empty memory,
   provisional bank, promotion rule, and identity revalidation, but retained
   action values update directly from every delayed outcome.

All conditions receive paired task and action-randomness seeds.

## Bootstrap audit metrics

In addition to return, retention, final probes, residual calibration, and
protection invariants, Experiment 012 records:

- unique initial identities promoted before the shift, out of eight;
- the decision at which all eight initial identities have been promoted;
- clean action accuracy during decisions 0-49 and 150-199;
- a clean eight-context core-policy probe immediately before the shift; and
- unique post-shift novel identities and reversed policies recovered.

Category identity and correct policy are used only by this audit. They are not
inputs to retrieval, assent, action selection, candidate matching, or writing.

## Frozen development run

Development trains encoders with seeds `1300-1301`. Each is evaluated on ten
paired lifetimes:

```text
113,000,000-113,000,009
113,001,000-113,001,009
```

No result may change the architecture, thresholds, support counts, training
schedule, comparisons, or criteria below.

## Frozen development gate

Fresh-seed confirmation is triggered only if learned-cold protected Chevron:

- promotes at least 7.5 of eight initial identities before the shift;
- reaches at least 0.70 action accuracy during decisions 150-199;
- reaches at least 0.75 on the eight-context core probe at the shift;
- retains at least 0.85 stable accuracy in decisions 600-799;
- reaches at least 0.70 final reversed and novel probe accuracy;
- promotes at least three of four post-shift novel identities and recovers at
  least three of four reversed policy identities on average;
- has identity and policy residual calibration of at least 0.10;
- averages no more than 0.50 false stable-context policy revisions;
- makes zero duplicate identity allocations, established-memory overwrites,
  and under-supported permanent writes;
- improves realised return and the shift-time core probe over raw-sensor cold
  Chevron with paired 95% lower bounds above zero;
- is non-inferior to oracle-cold protected Chevron in realised return and clean
  accuracy with paired 95% lower bounds above -0.08, and in the shift-time core
  probe with a lower bound above -0.10;
- is non-inferior to learned-preloaded protected Chevron in realised return
  with a paired 95% lower bound above -0.15 and retention with a lower bound
  above -0.08; and
- is non-inferior to learned-cold direct adaptation in realised return and
  clean accuracy with paired 95% lower bounds above -0.08 and retention with a
  lower bound above -0.05.

If every criterion passes, confirmation will train ten untouched encoders with
seeds `1310-1319`. Each will be evaluated on twenty paired lifetimes using seed
blocks `114,000,000-114,000,019` through
`114,009,000-114,009,019`. The complete protocol will remain frozen.

## Interpretation

Passing would show that the current agent can construct an initial permanent
identity-and-policy core through the same protected path later used for novelty.
Together with Experiment 011a, that would remove both the supplied identity
geometry and the preloaded-memory assumptions on this synthetic task. The next
step would be a small partially observable rooms-and-routes environment.

Failure would identify bootstrap as a separate unsolved problem. Likely causes
would include insufficient initial exploration, candidate fragmentation,
incorrect early policy consolidation, or later inability to revise a
self-created slot. The failed metric should select the next intervention; the
write threshold will not be swept after the result.

## Development outcome

The frozen development gate failed, so confirmation was not run.

Learned-cold protected Chevron passed twenty-one of twenty-four criteria. It
promoted 7.5 of eight initial identities, reached 0.815 late-bootstrap accuracy
and a 0.900 shift-time core probe, retained 0.921 stable accuracy, and finished
with 0.825 reversed and 0.863 novel probes. It promoted 3.5 post-shift novel
identities and recovered 3.25 reversed policy identities.

Against raw-sensor cold start it improved return by +0.249, approximate paired
95% interval +0.179 to +0.318, and the core probe by +0.188, interval +0.122 to
+0.253. Its core probe was statistically indistinguishable from oracle cold
start.

It failed the three return non-inferiority criteria. The paired return gaps were
-0.083 versus oracle cold, -0.189 versus learned preloaded memory, and -0.095
versus learned-cold direct adaptation. Retention non-inferiority passed in every
comparison. Protected Chevron also finished with a +0.163 novel-probe advantage
over direct adaptation, interval +0.041 to +0.284.

An additional audit diagnostic found five self-created core memory ids lost
across three of twenty lifetimes. No geometrically detected duplicate was
allocated, but a familiar identity can be split by the imperfect learned
geometry, fill permanent capacity, and displace an older self-created slot.
This motivates slot maturity and explicit forgetting permission rather than a
gate-threshold sweep.

The first development report was regenerated after correcting an audit-only
bug: the shift probe initially compared against post-shift policies, causing
even the preloaded control to score 0.5. The corrected probe uses initial
policies. Agent behaviour, seeds, criteria, and every learning rule were
unchanged.
