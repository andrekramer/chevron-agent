from __future__ import annotations

import inspect
import unittest

import torch

from experiments.experiment_007_action_prediction import (
    ActionConditionedPredictor,
    FixedLatentDynamics,
    transition_prediction_loss,
    train_action_encoder,
)


class ActionPredictionTests(unittest.TestCase):
    def test_dynamics_are_fixed_deterministic_and_norm_preserving(self) -> None:
        first = FixedLatentDynamics(4, 12, seed=607)
        second = FixedLatentDynamics(4, 12, seed=607)
        latent = torch.nn.functional.normalize(torch.randn(8, 12), dim=-1)
        action = torch.arange(8) % 4
        output = first(latent, action)
        torch.testing.assert_close(output, second(latent, action))
        torch.testing.assert_close(
            torch.linalg.vector_norm(output, dim=-1),
            torch.ones(8),
            atol=1e-5,
            rtol=1e-5,
        )
        self.assertEqual(sum(p.numel() for p in first.parameters()), 0)

    def test_actions_produce_distinct_transitions(self) -> None:
        dynamics = FixedLatentDynamics(4, 12, seed=607)
        latent = torch.nn.functional.normalize(torch.randn(1, 12), dim=-1)
        repeated = latent.expand(4, -1)
        outputs = dynamics(repeated, torch.arange(4))
        pairwise = outputs @ outputs.T
        self.assertTrue(torch.all(pairwise[~torch.eye(4, dtype=torch.bool)] < 0.99))

    def test_predictor_outputs_unit_vectors(self) -> None:
        predictor = ActionConditionedPredictor(4, 12)
        embedding = torch.randn(7, 12)
        output = predictor(embedding, torch.arange(7) % 4)
        torch.testing.assert_close(
            torch.linalg.vector_norm(output, dim=-1),
            torch.ones(7),
        )

    def test_transition_loss_prefers_true_pairing_and_has_gradient(self) -> None:
        prediction = torch.eye(6, requires_grad=True)
        observed = torch.eye(6, requires_grad=True)
        aligned = transition_prediction_loss(
            prediction,
            observed,
            temperature=0.1,
        )
        shuffled = transition_prediction_loss(
            prediction,
            observed.roll(1, dims=0),
            temperature=0.1,
        )
        self.assertLess(aligned.item(), shuffled.item())
        aligned.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all().item())
        self.assertTrue(torch.isfinite(observed.grad).all().item())

    def test_training_api_has_no_task_or_reward_labels(self) -> None:
        names = tuple(inspect.signature(train_action_encoder).parameters)
        self.assertEqual(names, ("config", "sensor", "dynamics", "seed"))
        loss_names = tuple(inspect.signature(transition_prediction_loss).parameters)
        self.assertEqual(
            loss_names,
            ("predicted_next", "observed_next", "temperature"),
        )


if __name__ == "__main__":
    unittest.main()
