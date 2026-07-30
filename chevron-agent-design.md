# Chevron Agent design

Status: initial design  
Target: part-time development on an Apple Silicon MacBook or Google Colab  
Primary implementation: conventional reinforcement learning with PyTorch  

## 1. Purpose

Chevron Agent is a small, reproducible reinforcement-learning agent whose
decisions and lifetime learning arise from an explicit interaction between:

- **A**: fast, adaptive, particular activity; and
- **N**: slower, retained, generalised structure.

The agent must use either a Chevron Network or Chevron Attention inside its
decision and memory loop. It must not be an ordinary recurrent agent with a
Chevron-labelled external database.

The first implementation will use **Chevron Attention**, because the preceding
Chevron Attention experiments already provide evidence that separating
retrieval, assent, and write permission can protect retained memory when
novelty lies close to existing structure.

The project asks:

> Can a small RL agent become more capable over its lifetime without losing
> the retained structure that makes it the same agent?

## 2. Scope and constraints

### In scope

- A conventional RL agent trained from scratch.
- Small symbolic or compact visual game environments.
- Persistent state across a sequence of episodes forming one lifetime.
- Fast adaptation, retained generalisation, and controlled memory revision.
- An operational, behavioural definition of a minimal core self.
- Seeded, reproducible experiments on modest hardware.

### Out of scope for the first implementation

- Language models or pretrained foundation models.
- Claims about consciousness or personhood.
- Solving Minecraft, NetHack, or another large environment from scratch.
- Large-scale distributed training.
- Assuming that one specific consolidation schedule is correct.
- A general claim that Chevron replaces standard attention or recurrent
  networks.

## 3. Two coupled design tasks

The project has two distinct but coupled components:

1. **Agent design**: an A/N policy and memory architecture that is genuinely
   Chevron.
2. **Environment design**: a lifetime of problems that requires both fast
   adaptation and slow retention.

The environment must expose a real stability-plasticity problem. If it is
stationary, or if every episode resets all relevant structure, the Chevron
mechanism is unnecessary and can be bypassed.

Conversely, the environment must not encode the desired Chevron solution. The
same lifetime transformations should eventually be applied to more than one
standard environment family.

## 4. Agent architecture

### 4.1 State

At time step `t`, the agent receives:

- observation `o_t`;
- previous action `a_(t-1)`;
- previous reward `R_(t-1)`; and
- the previous assented memory read `z_(t-1)`.

An encoder produces:

```text
x_t = encode(o_t, a_(t-1), R_(t-1))
```

The fast A state is a simple leaky recurrent state:

```text
A_t = (1 - lambda_A) A_(t-1)
      + lambda_A f_A(x_t, z_(t-1))
```

This recurrence should remain small and explicit. A large GRU or LSTM must not
sit alongside the Chevron mechanism, because it could solve the memory problem
while ignoring A/N.

### 4.2 Memory slots

Each Chevron memory slot `j` contains:

- `A_mem[j]`: an adaptive address trace used for retrieval;
- `N_mem[j]`: slower retained content or template;
- slot type and provenance metadata;
- confidence, age, and update statistics.

The initial implementation should use roughly 8--16 slots with 32--64
dimensions. Exact sizes will be selected on development seeds and frozen before
confirmation.

### 4.3 Chevron Attention

Retrieval is driven by A:

```text
alpha_t = softmax(Q_A(A_t) @ K_A(A_mem)^T)
```

Each retrieved slot is independently compared with current evidence:

```text
r_tj = sigmoid(k * (theta - M(A_t, N_mem[j])))
```

`M` is a bounded or normalised mismatch. The assent value `r_tj` is high when
the current A state is compatible with the retained N slot and low when it is
not.

The admitted read is:

```text
w_tj = alpha_tj * r_tj
q_t  = 1 - sum_j(w_tj)

z_t = sum_j(w_tj * V_N(N_mem[j])) + q_t * V_null
```

This preserves the central distinction:

```text
retrieval != assent != write permission
```

Rejected mass is not redistributed among the remaining slots. The residual
`q_t` is an explicit novelty or unresolved signal.

### 4.4 Actor and critic

The policy and value heads receive the adaptive state, the assented retained
read, and the unresolved mass:

```text
policy_logits_t, value_t = actor_critic(A_t, z_t, q_t)
```

The heads must be deliberately small. They should not have enough hidden
capacity or independent recurrence to reconstruct an alternative memory
system.

### 4.5 Writes

Established slots use the same local mass that controls their read:

```text
w_tj = alpha_tj * r_tj
```

A generic retained update is:

```text
N_mem[j] <- (1 - eta_N * w_tj) * N_mem[j]
            + eta_N * w_tj * T_N(A_t)
```

The address trace may update more quickly:

```text
A_mem[j] <- (1 - eta_A * w_tj) * A_mem[j]
            + eta_A * w_tj * T_A(A_t)
```

with:

```text
eta_A > eta_N
```

The precise retained update rule is an experimental choice. The architectural
commitment is that an attended slot does not receive a full write unless it
also receives assent.

### 4.6 Residual and provisional state

Residual mass `q_t` may create or update a small A-derived candidate buffer.
Previous experiments support temporal persistence before consolidation, but
they do not establish one uniquely correct staging mechanism.

Chevron Agent therefore exposes a consolidation interface rather than baking
one schedule into the core:

```text
observe_candidate(A_t, q_t, provenance)
propose_updates(A_state, N_state, candidates)
evaluate_updates(proposals, experience)
commit_or_reject(proposals)
```

Candidate implementations include:

- immediate thresholded allocation;
- evidence-persistence allocation;
- episode-boundary consolidation;
- event-triggered rest;
- replay-based consolidation;
- gradually reduced vigilance;
- a learned consolidation controller.

These are hypotheses to compare after the basic agent works.

## 5. Learning and lifetime boundaries

Three kinds of state must remain distinct:

1. **Model parameters** are learned by PPO or another conventional RL
   algorithm across training lifetimes.
2. **A state** changes rapidly during interaction.
3. **N state** develops within a lifetime and persists across its episodes.

Training consists of many independent lifetimes. At the beginning of each
training or evaluation lifetime:

- A is reset;
- N and provisional state are reset to the specified initial core;
- model parameters are retained.

Within a lifetime:

- the world may reset between episodes;
- A may reset partially or decay between episodes;
- N persists;
- the same agent continues to act.

During final evaluation, PPO parameters are frozen. Only the declared A/N
lifetime update rules may change agent state.

For tractable optimisation, gradients may flow through A and Chevron reads
within a rollout window. A and N are detached between rollout chunks while
their numerical state persists. Whether future rewards should differentiate
through explicit N writes is an implementation decision to test carefully.

## 6. Operational core self

The first Chevron Agent will not claim to possess a self in a philosophical
sense. It will implement a minimal behavioural core whose accumulated history
constrains future action.

N may contain typed slots with different update rates:

- **world slots**: environmental regularities and object relations;
- **capability slots**: acquired skills and competence estimates;
- **self slots**: role, commitments, and significant autobiographical state.

A small initial drive or task objective seeds the lifetime. The developed core
is the retained structure produced by the agent's subsequent choices,
experiences, capabilities, and commitments.

The desired pattern is:

```text
initial drive
    -> experience and action
    -> acquired capabilities
    -> commitments and history
    -> retained behavioural identity
```

World knowledge should generally be more plastic than commitments. Protected
experimental constraints may be external to learnable N, rather than being
silently treated as ordinary writable memories.

## 7. Environment design

### 7.1 First platform

The first environment will be built with
[MiniGrid](https://github.com/Farama-Foundation/Minigrid). It is maintained,
fast, configurable, compatible with Gymnasium, and small enough for many seeded
runs on a MacBook.

MiniGrid missions can use symbolic task vectors rather than language. The
agent will receive no pretrained knowledge.

### 7.2 Chevron Gridlife

One **lifetime** is a sequence of related missions. The environment resets
between missions, but Chevron Agent's N state does not.

A candidate campaign is:

1. **Formation**: learn basic mechanics and develop or select a role.
2. **Competence**: acquire several reusable navigation, key, tool, and hazard
   skills.
3. **Adjacent novelty**: encounter a new object or rule that nearly matches an
   established one.
4. **Instability**: experience noise, temporary failures, and misleading
   feedback.
5. **Real change**: a previously reliable environmental rule changes
   persistently.
6. **Identity pressure**: short-term reward or local evidence conflicts with a
   retained commitment.
7. **Return**: revisit early tasks and measure retained competence and
   behavioural continuity.

Each individual mission may already be solvable by ordinary RL. The research
problem is successful learning across the entire non-stationary lifetime.

### 7.3 Reusable lifetime wrapper

Where possible, non-stationarity should be introduced through a reusable
wrapper or protocol:

```text
stable regime
    -> temporary perturbation
    -> adjacent novelty
    -> persistent change
    -> return to prior tasks
```

This protocol can later be applied to:

- [MinAtar](https://github.com/kenjyoung/MinAtar), for a compact second game
  family; and
- [Craftax-Classic](https://github.com/michaeltmatthews/craftax), for a richer
  and more visually compelling demonstration if compute permits.

## 8. Initial implementation budget

The first model should aim for:

- one small symbolic observation encoder;
- `A` dimension of 32--64;
- 8--16 paired A/N memory slots;
- one Chevron Attention head;
- small actor and critic heads;
- fewer than approximately 100,000 trainable parameters;
- PPO as the initial optimiser;
- CPU compatibility and optional Apple MPS acceleration.

The implementation should provide deterministic seeds, serialisable lifetime
state, trajectory logging, and a CPU smoke-test configuration.

## 9. Controls and ablations

The complete agent is the main result. A minimal comparison set is still
required to determine whether the architecture caused its behaviour:

1. ordinary recurrent PPO with matched parameter count;
2. standard-attention PPO without independent assent;
3. full Chevron Attention Agent.

Useful Chevron ablations include:

- `r = 1`, making writes depend on retrieval alone;
- no residual or candidate state;
- N reset between episodes;
- A-only policy;
- N-only or frozen-N policy;
- immediate versus delayed consolidation.

A later architectural experiment can replace Chevron Attention with a Chevron
Network containing explicit `W_AA`, `W_AN`, `W_NA`, and `W_NN` transformations.

## 10. Measurements

Primary measurements:

- lifetime return and mission success;
- retained performance when earlier tasks return;
- adaptation delay after a genuine regime change;
- forward transfer to related new tasks;
- false consolidation after temporary perturbations;
- failure to consolidate persistent novelty;
- interference with established capabilities;
- commitment or protected-constraint violations;
- candidate capacity and purity;
- N drift and slot usage;
- steps, wall-clock time, and peak memory.

The main visual result should be a lifetime trace showing:

- tasks and regime changes;
- performance;
- A/N mismatch and assent;
- residual mass and candidate state;
- N writes and consolidations;
- retained commitments and capabilities.

## 11. Experimental sequence

### Phase 9A: infrastructure and smoke task

- Implement the small PPO training loop.
- Implement the A/N state and Chevron Attention read.
- Verify that the agent can solve a stationary MiniGrid task.
- Verify deterministic seeded evaluation and MPS/CPU parity within tolerance.

### Phase 9B: persistent memory

- Run several related missions in one lifetime.
- Require N to preserve information across episode resets.
- Confirm that the policy uses N rather than bypassing it.

### Phase 9C: stability and plasticity

- Add temporary perturbation, adjacent novelty, and persistent change.
- Add explicit writes, residual mass, and a minimal candidate mechanism.
- Measure false updates, genuine adaptation, and retained competence.

### Phase 9D: core-self campaign

- Add role formation, capabilities, commitments, and delayed consequences.
- Produce complete developmental trajectories.
- Compare the full agent with the minimal controls.

### Phase 9E: confirmation and transfer

- Freeze architecture, hyperparameters, lifetime protocol, and metrics.
- Run fresh confirmation seeds.
- Apply the lifetime protocol to a second environment family.

## 12. Success criterion and claim boundary

The project succeeds if Chevron Agent:

- solves the component missions;
- improves across its lifetime;
- retains earlier capabilities;
- adapts to genuine persistent changes;
- avoids treating temporary contradictions as permanent changes; and
- maintains its declared behavioural core better than matched controls.

A defensible positive claim would be:

> An explicit fast-A/slow-N RL architecture with local retrieval, assent, and
> write control improves the stability-plasticity trade-off of a persistent
> agent in non-stationary game environments.

The experiments would not by themselves establish consciousness, personhood,
or fully general agency.

## 13. Open design decisions

- Exact mismatch `M(A,N)` and its normalisation.
- Whether assent threshold and slope are fixed or learned.
- Whether N is slot-based, factorised by slot type, or initially homogeneous.
- How much A state persists across episode boundaries.
- Which consolidation policy is used first.
- Whether vigilance is fixed, event-triggered, or scheduled during rest.
- Whether write updates are differentiated through time.
- MiniGrid campaign mechanics and reward design.
- PPO implementation: small local implementation or a lightweight maintained
  dependency.

These decisions should be resolved by the cheapest diagnostic experiments
before committing to the full lifetime campaign.
