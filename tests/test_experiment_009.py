from __future__ import annotations

import unittest

import torch

from experiments.experiment_009_dual_relation import (
    DualObservation,
    DualRelationAgent,
    ExperimentConfig,
    make_lifetime,
    policy_signature,
)


class DualRelationTests(unittest.TestCase):
    def test_policy_signatures_separate_different_actions(self) -> None:
        config = ExperimentConfig()
        signatures = torch.stack(
            [policy_signature(action, config.action_dim) for action in range(config.action_dim)]
        )
        torch.testing.assert_close(torch.diag(signatures @ signatures.T), torch.ones(4))
        off_diagonal = (signatures @ signatures.T)[~torch.eye(4, dtype=torch.bool)]
        torch.testing.assert_close(off_diagonal, torch.full((12,), -1.0 / 3.0))

    def test_lifetime_has_stable_reversed_and_novel_roles(self) -> None:
        config = ExperimentConfig(stream_steps=30)
        lifetime = make_lifetime(config, 90_000_000)
        self.assertEqual(len(lifetime.stable_categories), 4)
        self.assertEqual(len(lifetime.reversed_categories), 4)
        self.assertEqual(len(lifetime.novel_categories), 4)
        for category in lifetime.reversed_categories:
            self.assertNotEqual(
                int(lifetime.initial_actions[category]),
                int(lifetime.current_actions[category]),
            )

    def test_same_identity_changed_policy_routes_to_revision(self) -> None:
        config = ExperimentConfig(stream_steps=10)
        lifetime = make_lifetime(config, 90_000_001)
        category = lifetime.reversed_categories[0]
        agent = DualRelationAgent("dual_buffer", config, lifetime, seed=1)
        observation = DualObservation(
            0,
            category // 3,
            lifetime.identity_prototypes[category],
            policy_signature(int(lifetime.current_actions[category]), config.action_dim),
        )
        trace = agent.distribution(observation)
        # Broad family retrieval gives half its mass to the incompatible sibling.
        self.assertLess(float(trace.q_identity), 0.7)
        self.assertGreater(float(trace.q_policy), 0.8)
        self.assertEqual(trace.candidate_kind, "policy_revision")

    def test_new_identity_routes_to_identity_allocation(self) -> None:
        config = ExperimentConfig(stream_steps=10)
        lifetime = make_lifetime(config, 90_000_002)
        category = lifetime.novel_categories[0]
        agent = DualRelationAgent("dual_buffer", config, lifetime, seed=2)
        observation = DualObservation(
            0,
            category // 3,
            lifetime.identity_prototypes[category],
            policy_signature(int(lifetime.current_actions[category]), config.action_dim),
        )
        trace = agent.distribution(observation)
        self.assertGreater(float(trace.q_identity), 0.8)
        self.assertEqual(trace.candidate_kind, "new_identity")

    def test_collapsed_gate_misroutes_reversal_as_identity(self) -> None:
        config = ExperimentConfig(stream_steps=10)
        lifetime = make_lifetime(config, 90_000_003)
        category = lifetime.reversed_categories[0]
        agent = DualRelationAgent("collapsed_buffer", config, lifetime, seed=3)
        observation = DualObservation(
            0,
            category // 3,
            lifetime.identity_prototypes[category],
            policy_signature(int(lifetime.current_actions[category]), config.action_dim),
        )
        trace = agent.distribution(observation)
        self.assertEqual(trace.candidate_kind, "collapsed_identity")

    def test_immediate_revision_records_premature_write(self) -> None:
        config = ExperimentConfig(stream_steps=10)
        lifetime = make_lifetime(config, 90_000_004)
        category = lifetime.reversed_categories[0]
        agent = DualRelationAgent("dual_immediate", config, lifetime, seed=4)
        observation = DualObservation(
            0,
            category // 3,
            lifetime.identity_prototypes[category],
            policy_signature(int(lifetime.current_actions[category]), config.action_dim),
        )
        agent.act(observation)
        self.assertEqual(agent.counters.premature_writes, 1)


if __name__ == "__main__":
    unittest.main()
