from __future__ import annotations

import inspect
import unittest

import torch

from experiments.experiment_006_predictive_geometry import FixedNonlinearSensor
from experiments.experiment_010_retrospective_policy import make_lifetime
from experiments.experiment_011_persistent_identity import (
    ExperimentConfig,
    ResidualIdentityEncoder,
    persistence_contrastive_loss,
    sample_persistence_windows,
    train_identity_encoder,
    transform_lifetime,
)


class PersistentIdentityTests(unittest.TestCase):
    def test_residual_encoder_starts_as_normalised_sensor_identity(self) -> None:
        encoder = ResidualIdentityEncoder(12, 24)
        observed = torch.randn(16, 12)
        torch.testing.assert_close(
            encoder(observed),
            torch.nn.functional.normalize(observed, dim=-1),
        )

    def test_persistence_loss_prefers_grouped_views_and_has_gradients(self) -> None:
        base = torch.eye(4)
        grouped = base[:, None, :].repeat(1, 3, 1).requires_grad_()
        mixed = grouped.detach().transpose(0, 1).reshape(4, 3, 4)
        grouped_loss = persistence_contrastive_loss(grouped, temperature=0.1)
        mixed_loss = persistence_contrastive_loss(mixed, temperature=0.1)
        self.assertLess(grouped_loss.item(), mixed_loss.item())
        grouped_loss.backward()
        self.assertTrue(torch.isfinite(grouped.grad).all().item())

    def test_persistence_objective_has_no_task_label_arguments(self) -> None:
        names = tuple(inspect.signature(persistence_contrastive_loss).parameters)
        self.assertEqual(names, ("embeddings", "temperature"))

    def test_hard_persistence_sampler_has_expected_shape(self) -> None:
        config = ExperimentConfig(
            pretraining_observations=32,
            persistence_views=4,
        )
        sensor = FixedNonlinearSensor(12, 32, seed=606)
        windows = sample_persistence_windows(
            config,
            sensor,
            torch.Generator().manual_seed(4),
            hard_negatives=True,
        )
        self.assertEqual(windows.shape, (8, 4, 12))
        torch.testing.assert_close(
            torch.linalg.vector_norm(windows, dim=-1),
            torch.ones(8, 4),
        )

    def test_short_training_is_finite_and_updates_residual(self) -> None:
        config = ExperimentConfig(
            pretraining_steps=2,
            pretraining_observations=32,
            persistence_views=4,
            encoder_hidden_dim=16,
        )
        sensor = FixedNonlinearSensor(12, 32, seed=606)
        encoder, losses = train_identity_encoder(
            config,
            sensor,
            seed=0,
            objective="hard_persistence",
        )
        self.assertEqual(len(losses), 2)
        self.assertTrue(torch.isfinite(torch.tensor(losses)).all().item())
        final = encoder.correction[-1]
        self.assertGreater(float(final.weight.detach().abs().sum()), 0.0)

    def test_lifetime_transform_changes_only_observed_identity_geometry(self) -> None:
        config = ExperimentConfig(
            stream_steps=40,
            shift_step=10,
            retention_step=30,
        )
        lifetime = make_lifetime(config, 110_000_000)
        sensor = FixedNonlinearSensor(12, 32, seed=606)
        transformed = transform_lifetime(lifetime, sensor)
        self.assertEqual(transformed.initial_actions.tolist(), lifetime.initial_actions.tolist())
        self.assertEqual(transformed.current_actions.tolist(), lifetime.current_actions.tolist())
        self.assertEqual(transformed.stable_categories, lifetime.stable_categories)
        self.assertEqual(transformed.reversed_categories, lifetime.reversed_categories)
        self.assertEqual(transformed.novel_categories, lifetime.novel_categories)
        self.assertEqual(len(transformed.events), len(lifetime.events))
        self.assertFalse(
            torch.allclose(
                transformed.events[0].observation.identity,
                lifetime.events[0].observation.identity,
            )
        )


if __name__ == "__main__":
    unittest.main()
