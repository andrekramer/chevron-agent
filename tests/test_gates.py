from __future__ import annotations

import unittest

import torch

from chevron_agent import ProjectedCosineAssent


class ProjectedCosineAssentTests(unittest.TestCase):
    def test_ranges_and_positive_parameterisation(self) -> None:
        gate = ProjectedCosineAssent(4, 4, 6)
        evidence = torch.randn(3, 4)
        retained = torch.randn(3, 5, 4)
        output = gate(evidence, retained)
        self.assertEqual(tuple(output.assent.shape), (3, 5))
        self.assertTrue(torch.all((output.mismatch >= 0) & (output.mismatch <= 1)).item())
        self.assertTrue(torch.all((output.assent >= 0) & (output.assent <= 1)).item())
        self.assertGreater(output.threshold.item(), 0.0)
        self.assertLess(output.threshold.item(), 1.0)
        self.assertGreater(output.slope.item(), 0.0)

    def test_gradients_reach_both_comparison_paths(self) -> None:
        gate = ProjectedCosineAssent(4, 4, 4)
        evidence = torch.randn(2, 4, requires_grad=True)
        retained = torch.randn(2, 3, 4, requires_grad=True)
        output = gate(evidence, retained)
        output.logits.square().mean().backward()
        self.assertTrue(torch.isfinite(evidence.grad).all().item())
        self.assertTrue(torch.isfinite(retained.grad).all().item())
        self.assertIsNotNone(gate.evidence_projection.weight.grad)
        self.assertIsNotNone(gate.retained_projection.weight.grad)

    def test_write_margin_is_stricter_than_read_gate(self) -> None:
        gate = ProjectedCosineAssent(3, 3, 3, initial_threshold=0.3)
        evidence = torch.randn(4, 3)
        retained = torch.randn(4, 2, 3)
        read = gate(evidence, retained).assent
        write = gate.assent_with_margin(evidence, retained, threshold_margin=0.05)
        self.assertTrue(torch.all(write < read).item())
        with self.assertRaises(ValueError):
            gate.assent_with_margin(evidence, retained, threshold_margin=0.0)


if __name__ == "__main__":
    unittest.main()
