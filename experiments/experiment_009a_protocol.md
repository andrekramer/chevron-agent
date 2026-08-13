# Experiment 009a: frozen provisional-routing diagnostic

## Question

Did Experiment 009's dual mechanism fail mainly because novel-identity and
policy-revision candidates competed in one provisional queue, or merely because
four entries were insufficient for the combined traffic?

## Fixed task and mechanism

The Experiment 009 task, seeds, identity and policy gates, action sampler,
delayed reward, two-positive-outcome promotion rule, permanent capacity, and
all update rules remain unchanged. No representation is learned and no gate is
recalibrated.

Only provisional storage layout changes.

## Conditions

1. Shared 4: the original capacity-four queue.
2. Split 2+2: two dedicated queues with capacity two each, preserving total
   provisional capacity four.
3. Shared 8: one queue with total capacity eight.
4. Split 4+4: two dedicated queues with capacity four each, preserving total
   provisional capacity eight relative to Shared 8.
5. Identity only, shared 4: the strongest simple control from Experiment 009.

Identity candidates can enter only the identity queue and policy-revision
candidates only the policy queue. Delayed eligibility remains separate from
both queues.

The comparison answers three different questions:

- Shared 8 versus Shared 4 measures capacity alone.
- Split 2+2 versus Shared 4 measures routing at equal total capacity.
- Split 4+4 versus Shared 8 measures routing at equal total capacity and
  sufficient per-route capacity.

## Development and confirmation

Development reuses the twenty frozen Experiment 009 lifetime seeds
90,000,000–90,000,019. This is a paired post-failure mechanism diagnostic; it
does not alter Experiment 009's failed confirmation decision.

The routing hypothesis passes only if Split 4+4:

- reaches at least 0.95 stable, 0.75 reversed, and 0.75 novel final accuracy;
- reaches at least 0.15 identity and policy residual calibration;
- consolidates at least three novel identities and three policy revisions;
- makes no premature writes, established overwrites, or duplicate allocations;
- has paired return and novel-accuracy intervals with lower bounds above zero
  versus Shared 8; and
- reduces mean evictions by at least 25% versus Shared 8.

Capacity is diagnosed separately: Shared 8 improves capacity if its paired
return or novel-accuracy interval has a lower bound above zero versus Shared 4.
Equal-budget routing is useful if Split 2+2 improves return or novel accuracy
over Shared 4 without reducing either metric by more than 0.05.

If the primary routing gate passes, confirmation uses 100 fresh lifetimes with
seeds 92,000,000–92,000,099. Otherwise confirmation is withheld.

## Interpretation

If Split 4+4 beats Shared 8, queue interference was a genuine routing problem.
If Shared 8 and Split 4+4 improve similarly, the failure was mostly capacity.
If neither closes the gap to identity-only adaptation, conservative policy veto
remains behaviourally expensive even after storage interference is removed.

## Development outcome

The primary routing gate failed. Split 4+4 was statistically indistinguishable
from Shared 8 on return and novel accuracy and produced 3.30 more evictions per
lifetime. Split 2+2 was mildly harmful because fixed partitions stranded spare
capacity during uneven traffic.

The predeclared capacity diagnostic passed cleanly. Shared 8 improved return
over Shared 4 by +0.1375, with an approximate paired 95% interval from +0.0928
to +0.1822, and novel accuracy by +0.2663, interval +0.1755 to +0.3570.
Evictions fell from 78.85 to 22.75. Shared 8 reached 0.725 return, 0.944 reversed
accuracy, and 0.864 novel accuracy.

Confirmation was withheld because the protocol reserved it for a routing
benefit. The result supports typed candidates in one adequately sized flexible
queue, not rigidly separate identity and policy queues.
