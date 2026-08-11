from __future__ import annotations

import math
import unittest

import torch

from experiments.experiment_004_reward_memory import (
    AgentObservation,
    ExperimentConfig,
    RewardMemoryAgent,
    geometric_cosine_assent,
    make_lifetime,
)


class GeometricChevronTests(unittest.TestCase):
    def test_similarity_boundary_maps_to_half_assent(self) -> None:
        evidence = torch.tensor([1.0, 0.0])
        boundary = 0.62
        retained = torch.tensor(
            [
                [boundary, math.sqrt(1.0 - boundary**2)],
                [0.90, math.sqrt(1.0 - 0.90**2)],
                [0.20, math.sqrt(1.0 - 0.20**2)],
            ]
        )
        assent = geometric_cosine_assent(
            evidence,
            retained,
            similarity_threshold=boundary,
        )
        torch.testing.assert_close(assent[0], torch.tensor(0.5))
        self.assertGreater(assent[1].item(), 0.5)
        self.assertLess(assent[2].item(), 0.5)

    def test_write_margin_is_stricter(self) -> None:
        evidence = torch.nn.functional.normalize(torch.randn(5), dim=0)
        retained = torch.nn.functional.normalize(torch.randn(3, 5), dim=-1)
        read = geometric_cosine_assent(
            evidence,
            retained,
            similarity_threshold=0.62,
        )
        write = geometric_cosine_assent(
            evidence,
            retained,
            similarity_threshold=0.62,
            threshold_margin=0.05,
        )
        self.assertTrue(torch.all(write < read).item())

    def test_mass_conservation_and_buffer_protection(self) -> None:
        config = ExperimentConfig(buffer_capacity=4)
        lifetime = make_lifetime(config, seed=23)
        agent = RewardMemoryAgent(
            "geometric_chevron_buffer",
            config,
            lifetime,
            model=None,
            training=False,
            action_seed=3,
        )
        category = lifetime.novel_categories[0]
        observation = AgentObservation(
            event_id=1000,
            family=category // 3,
            evidence=lifetime.prototypes[category],
        )
        initial_slots = len(agent.slots)
        _, trace = agent.act(observation)
        torch.testing.assert_close(trace.slot_mass.sum() + trace.q, torch.tensor(1.0))
        self.assertTrue(torch.all((trace.read_assent >= 0) & (trace.read_assent <= 1)))
        self.assertTrue(torch.all(trace.write_assent < trace.read_assent))
        self.assertTrue(trace.should_candidate)
        self.assertEqual(len(agent.slots), initial_slots)
        self.assertEqual(len(agent.buffer), 1)
        self.assertEqual(agent.metrics.premature_writes, 0)

    def test_immediate_ablation_writes_before_reward(self) -> None:
        config = ExperimentConfig(buffer_capacity=4)
        lifetime = make_lifetime(config, seed=23)
        agent = RewardMemoryAgent(
            "geometric_chevron_immediate",
            config,
            lifetime,
            model=None,
            training=False,
            action_seed=3,
        )
        category = lifetime.novel_categories[0]
        observation = AgentObservation(
            event_id=1000,
            family=category // 3,
            evidence=lifetime.prototypes[category],
        )
        initial_slots = len(agent.slots)
        agent.act(observation)
        self.assertEqual(len(agent.slots), initial_slots + 1)
        self.assertEqual(agent.metrics.premature_writes, 1)


if __name__ == "__main__":
    unittest.main()
