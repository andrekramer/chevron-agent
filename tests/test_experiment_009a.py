from __future__ import annotations

import unittest

from experiments.experiment_009_dual_relation import (
    DualObservation,
    DualRelationAgent,
    ExperimentConfig,
    make_lifetime,
    policy_signature,
)


class SplitQueueTests(unittest.TestCase):
    def test_split_layout_has_independent_capacities(self) -> None:
        config = ExperimentConfig(stream_steps=10)
        lifetime = make_lifetime(config, 90_000_010)
        agent = DualRelationAgent(
            "dual_buffer",
            config,
            lifetime,
            seed=10,
            buffer_layout="split",
            identity_capacity=2,
            policy_capacity=4,
        )
        self.assertEqual(agent.buffers["identity"].capacity, 2)
        self.assertEqual(agent.buffers["policy"].capacity, 4)
        self.assertIsNot(agent.buffers["identity"], agent.buffers["policy"])

    def test_candidate_types_route_to_different_queues(self) -> None:
        config = ExperimentConfig(stream_steps=10)
        lifetime = make_lifetime(config, 90_000_011)
        agent = DualRelationAgent(
            "dual_buffer",
            config,
            lifetime,
            seed=11,
            buffer_layout="split",
            identity_capacity=2,
            policy_capacity=2,
        )
        novel = lifetime.novel_categories[0]
        agent.act(
            DualObservation(
                0,
                novel // 3,
                lifetime.identity_prototypes[novel],
                policy_signature(int(lifetime.current_actions[novel]), config.action_dim),
            )
        )
        reversed_category = lifetime.reversed_categories[1]
        agent.act(
            DualObservation(
                1,
                reversed_category // 3,
                lifetime.identity_prototypes[reversed_category],
                policy_signature(
                    int(lifetime.current_actions[reversed_category]), config.action_dim
                ),
            )
        )
        self.assertEqual(len(agent.buffers["identity"].entries), 1)
        self.assertEqual(len(agent.buffers["policy"].entries), 1)
        self.assertEqual(agent.buffers["identity"].entries[0].kind, "new_identity")
        self.assertEqual(agent.buffers["policy"].entries[0].kind, "policy_revision")

    def test_default_layout_remains_shared_capacity_four(self) -> None:
        config = ExperimentConfig(stream_steps=10)
        lifetime = make_lifetime(config, 90_000_012)
        agent = DualRelationAgent("dual_buffer", config, lifetime, seed=12)
        self.assertEqual(tuple(agent.buffers), ("shared",))
        self.assertEqual(agent.buffers["shared"].capacity, 4)


if __name__ == "__main__":
    unittest.main()
