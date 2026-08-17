from __future__ import annotations

import unittest

import torch

from experiments.experiment_010_retrospective_policy import (
    ExperimentConfig,
    RetrospectiveAgent,
    RetrospectiveObservation,
    make_lifetime,
    run_development,
)


class RetrospectivePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ExperimentConfig(
            stream_steps=40,
            shift_step=10,
            retention_step=30,
            reward_flip_probability=0.0,
        )
        self.lifetime = make_lifetime(self.config, 100_000_000)

    def _observation(self, event_id: int, category: int) -> RetrospectiveObservation:
        return RetrospectiveObservation(
            event_id,
            category // 3,
            self.lifetime.identity_prototypes[category],
        )

    def test_observation_contains_no_policy_or_correct_action(self) -> None:
        fields = tuple(RetrospectiveObservation.__dataclass_fields__)
        self.assertEqual(fields, ("event_id", "family", "identity"))

    def test_identity_mass_is_conserved(self) -> None:
        category = self.lifetime.stable_categories[0]
        agent = RetrospectiveAgent(
            "retrospective_protected",
            self.config,
            self.lifetime,
            seed=1,
        )
        trace = agent.distribution(self._observation(0, category))
        torch.testing.assert_close(
            trace.identity_mass.sum() + trace.q_identity,
            torch.tensor(1.0),
        )

    def test_one_failure_is_suspicion_not_veto(self) -> None:
        category = self.lifetime.reversed_categories[0]
        incumbent = int(self.lifetime.initial_actions[category])
        agent = RetrospectiveAgent(
            "retrospective_protected",
            self.config,
            self.lifetime,
            seed=2,
        )
        action, _ = agent.act(self._observation(0, category))
        self.assertEqual(action, incumbent)
        agent.resolve_reward(0, -1.0)

        trace = agent.distribution(self._observation(1, category))
        self.assertAlmostEqual(float(trace.q_policy), 0.5)
        self.assertFalse(trace.policy_search_active)
        self.assertEqual(agent.counters.revision_promotions, 0)

    def test_second_failure_activates_search_without_writing(self) -> None:
        category = self.lifetime.reversed_categories[0]
        agent = RetrospectiveAgent(
            "retrospective_protected",
            self.config,
            self.lifetime,
            seed=3,
        )
        for event_id in range(2):
            agent.act(self._observation(event_id, category))
            agent.resolve_reward(event_id, -1.0)

        trace = agent.distribution(self._observation(2, category))
        self.assertEqual(float(trace.q_policy), 1.0)
        self.assertTrue(trace.policy_search_active)
        self.assertEqual(agent.counters.revision_promotions, 0)
        self.assertEqual(agent.counters.under_supported_writes, 0)

    def test_incumbent_success_dismisses_single_failure(self) -> None:
        category = self.lifetime.stable_categories[0]
        agent = RetrospectiveAgent(
            "retrospective_protected",
            self.config,
            self.lifetime,
            seed=4,
        )
        agent.act(self._observation(0, category))
        agent.resolve_reward(0, -1.0)
        agent.act(self._observation(1, category))
        agent.resolve_reward(1, 1.0)

        trace = agent.distribution(self._observation(2, category))
        self.assertEqual(float(trace.q_policy), 0.0)
        self.assertEqual(agent.counters.policy_dismissals, 1)
        self.assertEqual(agent.counters.revision_promotions, 0)

    def test_protected_revision_preserves_identity_and_requires_two_positives(self) -> None:
        category = self.lifetime.reversed_categories[0]
        incumbent = int(self.lifetime.initial_actions[category])
        correct = int(self.lifetime.current_actions[category])
        agent = RetrospectiveAgent(
            "retrospective_protected",
            self.config,
            self.lifetime,
            seed=5,
        )
        target = next(slot for slot in agent.slots if slot.origin_category == category)
        memory_id = target.memory_id
        identity = target.identity.clone()

        for event_id in range(2):
            action, _ = agent.act(self._observation(event_id, category))
            self.assertEqual(action, incumbent)
            agent.resolve_reward(event_id, -1.0)

        resolution = None
        positive_correct = 0
        for event_id in range(2, 20):
            action, _ = agent.act(self._observation(event_id, category))
            reward = 1.0 if action == correct else -1.0
            if action == correct:
                positive_correct += 1
            resolution = agent.resolve_reward(event_id, reward)
            if resolution.kind == "policy_revision":
                break
        self.assertEqual(positive_correct, 2)
        self.assertIsNotNone(resolution)
        self.assertEqual(resolution.kind, "policy_revision")
        revised = next(slot for slot in agent.slots if slot.memory_id == memory_id)
        torch.testing.assert_close(revised.identity, identity)
        self.assertEqual(int(revised.action_values.argmax()), correct)
        self.assertEqual(agent.counters.under_supported_writes, 0)

    def test_immediate_write_records_under_supported_revision(self) -> None:
        category = self.lifetime.reversed_categories[0]
        agent = RetrospectiveAgent(
            "retrospective_immediate_write",
            self.config,
            self.lifetime,
            seed=6,
        )
        agent.act(self._observation(0, category))
        agent.resolve_reward(0, -1.0)
        agent.act(self._observation(1, category))
        resolution = agent.resolve_reward(1, 1.0)
        self.assertEqual(resolution.kind, "policy_revision")
        self.assertEqual(agent.counters.under_supported_writes, 1)

    def test_novel_identity_routes_to_shared_bank(self) -> None:
        category = self.lifetime.novel_categories[0]
        agent = RetrospectiveAgent(
            "retrospective_protected",
            self.config,
            self.lifetime,
            seed=7,
        )
        observation = self._observation(0, category)
        trace = agent.distribution(observation)
        self.assertGreater(float(trace.q_identity), 0.8)
        self.assertEqual(trace.candidate_kind, "new_identity")
        agent.act(observation)
        self.assertEqual(agent.bank.entries[0].kind, "new_identity")
        self.assertEqual(agent.counters.new_promotions, 0)

    def test_development_seed_boundary_is_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "100,000,000"):
            run_development(
                self.config,
                seeds=1,
                seed_offset=99_999_999,
            )


if __name__ == "__main__":
    unittest.main()
