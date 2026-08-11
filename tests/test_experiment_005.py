from __future__ import annotations

import inspect
import unittest

import torch

from chevron_agent import (
    ProjectedBilinearNullAttention,
    ProjectedCosineAssent,
)
from experiments.experiment_004_reward_memory import (
    AgentObservation,
    ExperimentConfig as BaseConfig,
    RewardMemoryAgent,
    make_lifetime,
    retrospective_consistency_loss,
)
from experiments.experiment_005_retrospective_assent import ExperimentConfig


class RetrospectiveAssentExperimentTests(unittest.TestCase):
    def test_task_change_is_limited_to_buffer_and_loss_weight(self) -> None:
        previous = BaseConfig()
        current = ExperimentConfig()
        self.assertEqual(current.buffer_capacity, 4)
        self.assertEqual(current.retrospective_loss_weight, 1.0)
        for field in (
            "groups",
            "action_dim",
            "content_dim",
            "stream_steps",
            "shift_step",
            "outcome_delay",
            "novel_probability",
            "permanent_capacity",
            "promotion_support",
        ):
            self.assertEqual(getattr(current, field), getattr(previous, field))

    def test_retrospective_objective_uses_only_support_and_reward(self) -> None:
        names = tuple(inspect.signature(retrospective_consistency_loss).parameters)
        self.assertEqual(names, ("selected_action_support", "reward"))

    def test_reward_reverses_preferred_support(self) -> None:
        low = torch.tensor(0.2)
        high = torch.tensor(0.8)
        self.assertLess(
            retrospective_consistency_loss(high, 1.0),
            retrospective_consistency_loss(low, 1.0),
        )
        self.assertLess(
            retrospective_consistency_loss(low, -1.0),
            retrospective_consistency_loss(high, -1.0),
        )

    def test_retrospective_objective_has_finite_gate_gradient(self) -> None:
        support = torch.tensor(0.4, requires_grad=True)
        retrospective_consistency_loss(support, 1.0).backward()
        self.assertTrue(torch.isfinite(support.grad).item())
        self.assertLess(support.grad.item(), 0.0)

    def test_retrospective_gradient_reaches_chevron_comparison_paths(self) -> None:
        config = ExperimentConfig(stream_steps=20, shift_step=10)
        lifetime = make_lifetime(config, seed=41)
        model = ProjectedCosineAssent(12, 12, 13)
        agent = RewardMemoryAgent(
            "chevron_buffer",
            config,
            lifetime,
            model=model,
            training=True,
            action_seed=7,
        )
        category = lifetime.initial_categories[0]
        observation = AgentObservation(
            event_id=0,
            family=category // 3,
            evidence=lifetime.prototypes[category],
        )
        trace = agent._distribution(observation)
        values = torch.stack([slot.action_values for slot in agent.slots])
        action = int(torch.einsum("s,sa->a", trace.slot_mass, values).argmax())
        support = torch.sum(
            trace.slot_mass * values[:, action].clamp(min=0.0, max=1.0)
        )
        retrospective_consistency_loss(support, 1.0).backward()
        self.assertIsNotNone(model.evidence_projection.weight.grad)
        self.assertIsNotNone(model.retained_projection.weight.grad)
        self.assertGreater(
            model.evidence_projection.weight.grad.abs().sum().item(), 0.0
        )
        self.assertGreater(
            model.retained_projection.weight.grad.abs().sum().item(), 0.0
        )

    def test_strong_control_matches_chevron_parameter_budget(self) -> None:
        chevron = ProjectedCosineAssent(12, 12, 13)
        control = ProjectedBilinearNullAttention(12, 12, 13)
        self.assertEqual(sum(p.numel() for p in chevron.parameters()), 314)
        self.assertEqual(sum(p.numel() for p in control.parameters()), 314)


if __name__ == "__main__":
    unittest.main()
