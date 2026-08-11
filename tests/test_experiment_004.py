from __future__ import annotations

from dataclasses import fields
import unittest

import torch

from chevron_agent import DirectPairMLP, ProjectedCosineAssent
from experiments.experiment_004_reward_memory import (
    AgentObservation,
    ExperimentConfig,
    RewardMemoryAgent,
    make_lifetime,
)


class RewardMemoryExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ExperimentConfig(stream_steps=40, shift_step=20)
        self.lifetime = make_lifetime(self.config, seed=17)

    def _rejecting_gate(self) -> ProjectedCosineAssent:
        gate = ProjectedCosineAssent(12, 12, 13, initial_threshold=0.25, initial_slope=8.0)
        with torch.no_grad():
            gate.evidence_projection.weight.zero_()
            gate.retained_projection.weight.zero_()
        return gate

    def test_agent_observation_excludes_audit_labels(self) -> None:
        names = {field.name for field in fields(AgentObservation)}
        self.assertEqual(names, {"event_id", "family", "evidence"})
        self.assertNotIn("category", names)
        self.assertNotIn("correct_action", names)

    def test_lifetime_has_eight_old_four_novel_and_distinct_family_actions(self) -> None:
        self.assertEqual(len(self.lifetime.initial_categories), 8)
        self.assertEqual(len(self.lifetime.novel_categories), 4)
        for family in range(4):
            actions = self.lifetime.correct_actions[3 * family : 3 * family + 3]
            self.assertEqual(len(set(actions.tolist())), 3)

    def test_same_seed_reproduces_hidden_lifetime(self) -> None:
        other = make_lifetime(self.config, seed=17)
        torch.testing.assert_close(self.lifetime.prototypes, other.prototypes)
        torch.testing.assert_close(self.lifetime.correct_actions, other.correct_actions)
        for first, second in zip(self.lifetime.events, other.events, strict=True):
            self.assertEqual(first.category, second.category)
            torch.testing.assert_close(first.observation.evidence, second.observation.evidence)

    def test_models_have_equal_parameter_budget(self) -> None:
        chevron = ProjectedCosineAssent(12, 12, 13)
        direct = DirectPairMLP(12, 12, 12)
        self.assertEqual(sum(p.numel() for p in chevron.parameters()), 314)
        self.assertEqual(sum(p.numel() for p in direct.parameters()), 314)

    def test_buffer_prevents_permanent_write_before_delayed_reward(self) -> None:
        agent = RewardMemoryAgent(
            "chevron_buffer",
            self.config,
            self.lifetime,
            model=self._rejecting_gate(),
            training=False,
            action_seed=1,
        )
        category = self.lifetime.novel_categories[0]
        observation = AgentObservation(100, category // 3, self.lifetime.prototypes[category])
        initial_slots = len(agent.slots)
        _, trace = agent.act(observation)
        self.assertTrue(trace.should_candidate)
        self.assertEqual(len(agent.slots), initial_slots)
        self.assertEqual(agent.metrics.premature_writes, 0)
        self.assertEqual(len(agent.buffer), 1)

    def test_immediate_ablation_writes_before_reward(self) -> None:
        agent = RewardMemoryAgent(
            "chevron_immediate",
            self.config,
            self.lifetime,
            model=self._rejecting_gate(),
            training=False,
            action_seed=1,
        )
        category = self.lifetime.novel_categories[0]
        observation = AgentObservation(100, category // 3, self.lifetime.prototypes[category])
        initial_slots = len(agent.slots)
        agent.act(observation)
        self.assertEqual(len(agent.slots), initial_slots + 1)
        self.assertEqual(agent.metrics.premature_writes, 1)

    def test_two_coherent_rewards_are_required_for_promotion(self) -> None:
        agent = RewardMemoryAgent(
            "chevron_buffer",
            self.config,
            self.lifetime,
            model=self._rejecting_gate(),
            training=False,
            action_seed=1,
        )
        category = self.lifetime.novel_categories[0]
        observation_a = AgentObservation(100, category // 3, self.lifetime.prototypes[category])
        action_a, _ = agent.act(observation_a)
        reward_a = 1.0
        first = agent.resolve_reward(100, reward_a)
        self.assertIsNone(first.promotion_id)

        observation_b = AgentObservation(101, category // 3, self.lifetime.prototypes[category])
        action_b, _ = agent.act(observation_b)
        # Confirm the same action twice, as two positive delayed outcomes would.
        if action_b != action_a:
            agent.pending[101].action = action_a
        reward_b = 1.0
        second = agent.resolve_reward(101, reward_b)
        self.assertIsNotNone(second.promotion_id)
        promoted = next(slot for slot in agent.slots if slot.promotion_id == second.promotion_id)
        self.assertEqual(int(promoted.action_values.argmax()), action_a)


if __name__ == "__main__":
    unittest.main()
