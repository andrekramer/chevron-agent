from __future__ import annotations

import inspect
import unittest

import torch
from torch.nn import functional as F

from experiments.experiment_008_consequence_geometry import (
    ExperimentConfig,
    FixedAffordanceWorld,
    consequence_metric_loss,
    make_affordance_lifetime,
    train_consequence_encoder,
)


def _world(config: ExperimentConfig) -> FixedAffordanceWorld:
    return FixedAffordanceWorld(
        config.action_dim,
        config.content_dim,
        affordance_seed=config.affordance_seed,
        dynamics_seed=config.dynamics_seed,
        discount=config.consequence_discount,
        reward_scale=config.reward_scale,
    )


class ConsequenceGeometryTests(unittest.TestCase):
    def test_world_is_deterministic_bounded_and_parameter_free(self) -> None:
        config = ExperimentConfig()
        first = _world(config)
        second = _world(config)
        latent = F.normalize(torch.randn(16, config.content_dim), dim=-1)
        torch.testing.assert_close(
            first.consequence_values(latent), second.consequence_values(latent)
        )
        rewards = first.immediate_rewards(latent)
        self.assertTrue(torch.all(rewards >= -1.0).item())
        self.assertTrue(torch.all(rewards <= 1.0).item())
        self.assertEqual(sum(p.numel() for p in first.parameters()), 0)

    def test_signatures_are_unit_and_action_matches_value_argmax(self) -> None:
        config = ExperimentConfig()
        world = _world(config)
        latent = F.normalize(torch.randn(24, config.content_dim), dim=-1)
        signature = world.consequence_signature(latent)
        torch.testing.assert_close(
            torch.linalg.vector_norm(signature, dim=-1), torch.ones(24)
        )
        torch.testing.assert_close(
            world.optimal_action(latent),
            world.consequence_values(latent).argmax(dim=-1),
        )

    def test_metric_loss_prefers_matching_consequence_geometry(self) -> None:
        torch.manual_seed(8)
        target = F.normalize(torch.randn(12, 4), dim=-1)
        represented = F.pad(target, (0, 8)).requires_grad_()
        aligned = consequence_metric_loss(represented, represented, target)
        shuffled = consequence_metric_loss(
            represented, represented.roll(1, dims=0), target
        )
        self.assertLess(aligned.item(), shuffled.item())
        aligned.backward()
        self.assertTrue(torch.isfinite(represented.grad).all().item())

    def test_each_family_requires_three_distinct_actions(self) -> None:
        config = ExperimentConfig(stream_steps=20)
        lifetime = make_affordance_lifetime(config, _world(config), seed=52_000_008)
        for family in range(config.groups):
            actions = lifetime.correct_actions[3 * family : 3 * family + 3]
            self.assertEqual(len(set(actions.tolist())), 3)
        self.assertEqual(len(lifetime.events), config.stream_steps)

    def test_training_api_has_no_categories_memory_or_compatibility(self) -> None:
        names = tuple(inspect.signature(train_consequence_encoder).parameters)
        self.assertEqual(names, ("config", "sensor", "world", "seed"))
        loss_names = tuple(inspect.signature(consequence_metric_loss).parameters)
        self.assertEqual(
            loss_names,
            ("first_embedding", "second_embedding", "consequence_signature"),
        )

    def test_confirmed_gate_defaults_remain_frozen(self) -> None:
        config = ExperimentConfig()
        self.assertEqual(config.standard_similarity_threshold, 0.62)
        self.assertEqual(config.geometric_slope, 40.0)
        self.assertEqual(config.write_threshold_margin, 0.05)
        self.assertEqual(config.buffer_capacity, 4)
        self.assertEqual(config.promotion_support, 2)


if __name__ == "__main__":
    unittest.main()
