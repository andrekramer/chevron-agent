from __future__ import annotations

import inspect
import unittest

import torch

from experiments.experiment_004_reward_memory import make_lifetime
from experiments.experiment_006_predictive_geometry import (
    ExperimentConfig,
    FixedNonlinearSensor,
    TemporalContrastiveEncoder,
    _transform_lifetime,
    calibrate_gate_from_temporal_pairs,
    temporal_contrastive_loss,
)


class PredictiveGeometryTests(unittest.TestCase):
    def test_sensor_is_fixed_and_deterministic(self) -> None:
        sensor_a = FixedNonlinearSensor(12, 32, seed=606)
        sensor_b = FixedNonlinearSensor(12, 32, seed=606)
        latent = torch.randn(8, 12)
        torch.testing.assert_close(sensor_a(latent), sensor_b(latent))
        self.assertEqual(sum(p.numel() for p in sensor_a.parameters()), 0)

    def test_encoder_outputs_unit_vectors(self) -> None:
        encoder = TemporalContrastiveEncoder(12, 32)
        output = encoder(torch.randn(16, 12))
        torch.testing.assert_close(
            torch.linalg.vector_norm(output, dim=-1),
            torch.ones(16),
        )

    def test_contrastive_loss_prefers_aligned_pairs_and_has_gradients(self) -> None:
        first = torch.eye(6, requires_grad=True)
        aligned = torch.eye(6, requires_grad=True)
        shuffled = aligned.roll(1, dims=0)
        aligned_loss = temporal_contrastive_loss(
            first,
            aligned,
            temperature=0.1,
        )
        shuffled_loss = temporal_contrastive_loss(
            first,
            shuffled,
            temperature=0.1,
        )
        self.assertLess(aligned_loss.item(), shuffled_loss.item())
        aligned_loss.backward()
        self.assertTrue(torch.isfinite(first.grad).all().item())
        self.assertTrue(torch.isfinite(aligned.grad).all().item())

    def test_contrastive_objective_has_no_task_label_arguments(self) -> None:
        names = tuple(inspect.signature(temporal_contrastive_loss).parameters)
        self.assertEqual(names, ("first", "second", "temperature"))

    def test_lifetime_transform_preserves_task_audit_only(self) -> None:
        config = ExperimentConfig(stream_steps=30, shift_step=10)
        lifetime = make_lifetime(config, seed=19)
        sensor = FixedNonlinearSensor(12, 32, seed=606)
        transformed = _transform_lifetime(lifetime, sensor)
        self.assertEqual(transformed.correct_actions.tolist(), lifetime.correct_actions.tolist())
        self.assertEqual(transformed.initial_categories, lifetime.initial_categories)
        self.assertEqual(transformed.novel_categories, lifetime.novel_categories)
        self.assertEqual(len(transformed.events), len(lifetime.events))
        self.assertFalse(
            torch.allclose(transformed.events[0].observation.evidence, lifetime.events[0].observation.evidence)
        )

    def test_gate_calibration_is_label_free_and_bounded(self) -> None:
        names = tuple(inspect.signature(calibrate_gate_from_temporal_pairs).parameters)
        self.assertEqual(names, ("config", "sensor", "encoder", "seed"))
        config = ExperimentConfig(representation_evaluation_size=256)
        sensor = FixedNonlinearSensor(12, 32, seed=606)
        encoder = TemporalContrastiveEncoder(12, 32)
        calibration = calibrate_gate_from_temporal_pairs(
            config,
            sensor,
            encoder,
            seed=4,
        )
        self.assertGreater(calibration.similarity_threshold, -1.0)
        self.assertLess(calibration.similarity_threshold, 1.0)
        self.assertGreaterEqual(calibration.mismatch_slope, 20.0)
        self.assertLessEqual(calibration.mismatch_slope, 120.0)


if __name__ == "__main__":
    unittest.main()
