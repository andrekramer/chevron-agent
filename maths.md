A possibly overly optimistic interpretation of the maths follows.

More conservatively:
Chevron Attention factorises normalised retrieval mass into admitted and residual components. When the admitted mass also gates explicit memory writes, incompatible retained slots are protected from direct updates. Residual mass can signal unresolved evidence and be routed to provisional storage, although it is not intrinsically equivalent to novelty.

# The Mathematics of Chevron Attention

Chevron Attention is a variation of standard neural attention. The core mathematical innovation is that it breaks the attention mechanism into three mostly separated components: **retrieval**, **assent**, and **write permission**.

In standard attention (such as in a Transformer), the softmax function forces all attention weights to sum to 1. If an agent encounters a completely novel situation, standard attention is forced to distribute that "novelty" across existing, unrelated memories, potentially corrupting them. Chevron Attention uses a mathematical gate to prevent this.

Here is the step-by-step mathematical breakdown.

## 1. Retrieval ($\alpha$)

First, the agent uses its fast, adaptive state ($A_t$) to query the memory. It looks at the "address trace" ($A_{mem}$) of each memory slot $j$.

$$ \alpha_t = \text{softmax}(Q_A(A_t) K_A(A_{mem})^T) $$

* **$Q_A$ and $K_A$**: Standard Query and Key projection matrices.
* **$\alpha_t$**: A standard vector of probabilities summing to 1.
* **Conceptually**: *"Based on my current short-term state, which memory slot seems most relevant?"*

## 2. Assent ($r$)

This is the unique "Chevron" step. Just because a memory slot was retrieved does not mean its content actually matches the current situation. The agent independently compares the fast state ($A_t$) against the **slow, retained content** ($N_{mem}$) of the retrieved slot.

$$ r_{tj} = \text{sigmoid}(k \cdot (\theta - M(A_t, N_{mem}[j]))) $$

* **$M(A_t, N_{mem}[j])$**: A mismatch function (such as cosine distance or L2 norm). It measures how *different* the current state is from the retained memory.
* **$\theta$ ($\text{theta}$)**: A threshold of tolerance.
* **$k$**: A scalar that controls the steepness of the sigmoid (how harsh the cut-off is).
* **$r_{tj}$**: The assent value (between 0 and 1).
* **Conceptually**: *"Does the deep structure of this retrieved memory actually agree with my current evidence?"* If the mismatch $M$ is greater than the threshold $\theta$, the result is negative, and the sigmoid squashes $r_{tj}$ down to near 0.

## 3. The Admitted Read ($w$) and the Residual ($q$)

Next, the retrieval is filtered through the assent gate.

$$ w_{tj} = \alpha_{tj} \times r_{tj} $$

Because $r_{tj}$ can be less than 1, the sum of $w_{tj}$ across all slots is no longer guaranteed to sum to 1. The "rejected" mass becomes a highly valuable signal of its own—the residual mass:

$$ q_t = 1 - \sum_j w_{tj} $$

* **$w_{tj}$**: The final, admitted attention weight for slot $j$.
* **$q_t$**: The residual mass.
* **Conceptually**: $q_t$ is an explicit mathematical measurement of **novelty, surprise, or unresolved conflict**. If the agent retrieves a memory but the assent rejects it, $q_t$ spikes.

## 4. The Final Memory Output ($z$)

The actual memory vector passed to the Actor and Critic networks is assembled using only the admitted weights.

$$ z_t = \sum_j (w_{tj} \cdot V_N(N_{mem}[j])) + q_t \cdot V_{null} $$

* **$V_N$**: A Value projection matrix.
* **$V_{null}$**: A fixed or learnable vector representing an "unknown" or "novel" context.
* **Conceptually**: The agent uses the retained memory *only* if it matches. If it doesn't match, it relies on the $V_{null}$ vector, explicitly telling the policy and value heads: *"We are in uncharted territory."*

## 5. Write Permission

The mathematical beauty of Chevron Attention culminates in how memories are updated. The system uses the **exact same local mass ($w_{tj}$)** to gate the updates to the slow memory ($N_{mem}$).

$$ N_{mem}[j] \leftarrow (1 - \eta_N w_{tj}) N_{mem}[j] + \eta_N w_{tj} T_N(A_t) $$

* **$\eta_N$ ($\text{eta}_N$)**: The learning rate / update speed for retained memories.
* **$T_N(A_t)$**: The new target information derived from the current fast state.
* **Conceptually**: This is a gated Exponential Moving Average (EMA). **If a memory does not receive assent ($r_{tj} \approx 0$), then $w_{tj} \approx 0$, and the memory is mathematically protected from being overwritten.**

---

### Summary: Addressing the Stability-Plasticity Dilemma

In a standard RL setup, if the agent enters a new room that looks *slightly* similar to a room it already knows, standard attention ($\alpha$) will retrieve the old memory and force an update to it, destroying the old memory's structure (catastrophic forgetting).

In the Chevron Agent:
1. $\alpha$ retrieves the old memory.
2. $r$ calculates the mismatch, realizes this is a *new* room, and drops to 0.
3. $w$ drops to 0.
4. $q_t$ spikes to 1, telling the policy *"this is new."*
5. Because $w=0$, the write equation bypasses the old memory, leaving it largely untouched and preserved. The novelty ($q_t$) is instead routed to a provisional candidate state for later consolidation.

----

