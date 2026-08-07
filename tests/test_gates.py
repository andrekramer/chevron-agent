from __future__ import annotations

import unittest

import torch

from chevron_agent import DirectPairMLP, ProjectedCosineAssent


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


class DirectPairMLPTests(unittest.TestCase):
    def test_parameter_budget_matches_chevron_gate(self) -> None:
        chevron = ProjectedCosineAssent(12, 12, 13)
        direct = DirectPairMLP(12, 12, 12)
        self.assertEqual(
            sum(parameter.numel() for parameter in chevron.parameters()),
            sum(parameter.numel() for parameter in direct.parameters()),
        )
        self.assertEqual(sum(p.numel() for p in direct.parameters()), 314)

    def test_direct_probabilities_include_null_and_conserve_mass(self) -> None:
        model = DirectPairMLP(5, 5, 4)
        evidence = torch.randn(3, 5)
        retained = torch.randn(3, 2, 5)
        retrieval = torch.softmax(torch.randn(3, 2), dim=-1)
        output = model(evidence, retained, retrieval_mass=retrieval)
        self.assertEqual(output.logits.shape, (3, 3))
        self.assertTrue(
            torch.allclose(output.probabilities.sum(dim=-1), torch.ones(3))
        )
        self.assertTrue(torch.all((output.null_mass >= 0) & (output.null_mass <= 1)))


if __name__ == "__main__":
    unittest.main()
