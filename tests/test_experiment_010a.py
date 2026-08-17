from __future__ import annotations

import unittest

import torch

from experiments.experiment_010_retrospective_policy import (
    Candidate,
    ExperimentConfig,
    RetrospectiveAgent,
    make_lifetime,
)
from experiments.experiment_010a_revalidation import CONDITIONS, run_audit


class PreconsolidationRevalidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ExperimentConfig(
            stream_steps=40,
            shift_step=10,
            retention_step=30,
            reward_flip_probability=0.0,
        )
        self.lifetime = make_lifetime(self.config, 102_000_000)

    def _matching_candidate(self, agent: RetrospectiveAgent) -> Candidate:
        slot = agent.slots[0]
        positives = torch.zeros(self.config.action_dim, dtype=torch.long)
        positives[int(slot.action_values.argmax())] = 2
        return Candidate(
            candidate_id=99,
            kind="new_identity",
            family=slot.family,
            target_memory_id=None,
            identity=slot.identity.clone(),
            observations=2,
            pending_events=set(),
            positive_actions=positives,
            last_seen=20,
        )

    def test_original_path_still_allocates_and_counts_duplicate(self) -> None:
        agent = RetrospectiveAgent(
            "retrospective_protected",
            self.config,
            self.lifetime,
            seed=1,
        )
        before = len(agent.slots)
        allocated = agent._allocate(self._matching_candidate(agent), action=0)
        self.assertTrue(allocated)
        self.assertEqual(len(agent.slots), before + 1)
        self.assertEqual(agent.counters.duplicate_allocations, 1)
        self.assertEqual(agent.counters.identity_reconciliations, 0)

    def test_revalidation_reconciles_without_permanent_write(self) -> None:
        agent = RetrospectiveAgent(
            "retrospective_protected",
            self.config,
            self.lifetime,
            seed=2,
            revalidate_identity_promotion=True,
        )
        before_ids = [slot.memory_id for slot in agent.slots]
        allocated = agent._allocate(self._matching_candidate(agent), action=0)
        self.assertFalse(allocated)
        self.assertEqual([slot.memory_id for slot in agent.slots], before_ids)
        self.assertEqual(agent.counters.duplicate_allocations, 0)
        self.assertEqual(agent.counters.identity_reconciliations, 1)
        self.assertEqual(agent.counters.new_promotions, 0)

    def test_audit_conditions_are_narrow(self) -> None:
        self.assertEqual(
            CONDITIONS,
            ("protected_original", "protected_revalidated"),
        )

    def test_audit_seed_boundary_is_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "102,000,000"):
            run_audit(
                self.config,
                seeds=1,
                seed_offset=101_999_999,
            )


if __name__ == "__main__":
    unittest.main()
