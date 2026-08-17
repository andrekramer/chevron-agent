from __future__ import annotations

import unittest

import torch

from experiments.experiment_010_retrospective_policy import Candidate, make_lifetime
from experiments.experiment_012a_slot_maturity import (
    ExperimentConfig,
    MatureColdStartAgent,
)


class SlotMaturityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ExperimentConfig(
            stream_steps=40,
            shift_step=10,
            retention_step=30,
            reward_flip_probability=0.0,
            permanent_capacity=3,
            maturity_support=4,
        )
        self.lifetime = make_lifetime(self.config, 115_000_000)

    def _candidate(
        self,
        candidate_id: int,
        family: int,
        identity: torch.Tensor,
        event_id: int,
    ) -> Candidate:
        positives = torch.zeros(self.config.action_dim, dtype=torch.long)
        positives[0] = self.config.identity_promotion_support
        return Candidate(
            candidate_id=candidate_id,
            kind="new_identity",
            family=family,
            target_memory_id=None,
            identity=identity,
            observations=2,
            pending_events=set(),
            positive_actions=positives,
            last_seen=event_id,
        )

    def _agent(self, config: ExperimentConfig | None = None) -> MatureColdStartAgent:
        return MatureColdStartAgent(
            "retrospective_protected",
            config or self.config,
            self.lifetime,
            seed=1,
        )

    def test_four_successful_assented_uses_make_slot_mature(self) -> None:
        agent = self._agent()
        identity = self.lifetime.events[0].observation.identity
        self.assertTrue(agent._allocate(self._candidate(0, 0, identity, 0), 0))
        memory_id = agent.slots[0].memory_id
        for _ in range(self.config.maturity_support - 1):
            agent._record_successful_use(memory_id)
            self.assertNotIn(memory_id, agent.mature_memory_ids)
        agent._record_successful_use(memory_id)
        self.assertIn(memory_id, agent.mature_memory_ids)
        self.assertTrue(agent.slots[0].established)

    def test_full_mature_memory_defers_allocation_without_eviction(self) -> None:
        config = ExperimentConfig(
            stream_steps=40,
            shift_step=10,
            retention_step=30,
            permanent_capacity=1,
        )
        agent = self._agent(config)
        first = torch.nn.functional.normalize(torch.randn(12), dim=0)
        second = -first
        self.assertTrue(agent._allocate(self._candidate(0, 0, first, 0), 0))
        original_id = agent.slots[0].memory_id
        agent._make_mature(original_id)
        self.assertFalse(agent._allocate(self._candidate(1, 1, second, 1), 0))
        self.assertEqual([slot.memory_id for slot in agent.slots], [original_id])
        self.assertEqual(agent.mature_slots_evicted, 0)
        self.assertEqual(agent.allocation_deferrals, 1)
        self.assertIsNotNone(agent.bank.get(1))

    def test_capacity_pressure_evicts_immature_not_mature_slot(self) -> None:
        config = ExperimentConfig(
            stream_steps=40,
            shift_step=10,
            retention_step=30,
            permanent_capacity=2,
        )
        agent = self._agent(config)
        basis = torch.eye(12)
        self.assertTrue(agent._allocate(self._candidate(0, 0, basis[0], 0), 0))
        first_id = agent.slots[0].memory_id
        agent._make_mature(first_id)
        self.assertTrue(agent._allocate(self._candidate(1, 1, basis[1], 1), 0))
        second_id = max(slot.memory_id for slot in agent.slots)
        self.assertTrue(agent._allocate(self._candidate(2, 2, basis[2], 2), 0))
        remaining = {slot.memory_id for slot in agent.slots}
        self.assertIn(first_id, remaining)
        self.assertNotIn(second_id, remaining)
        self.assertEqual(agent.mature_slots_evicted, 0)
        self.assertEqual(agent.immature_slots_evicted, 1)

    def test_immediate_protection_marks_new_slot_mature(self) -> None:
        agent = MatureColdStartAgent(
            "retrospective_protected",
            self.config,
            self.lifetime,
            seed=2,
            immediate_protection=True,
        )
        identity = self.lifetime.events[0].observation.identity
        self.assertTrue(agent._allocate(self._candidate(0, 0, identity, 0), 0))
        self.assertIn(agent.slots[0].memory_id, agent.mature_memory_ids)


if __name__ == "__main__":
    unittest.main()
