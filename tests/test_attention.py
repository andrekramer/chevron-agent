from __future__ import annotations

import unittest

import torch

from chevron_agent import (
    ChevronAttentionConfig,
    apply_convex_write,
    chevron_attention,
    normalized_cosine_mismatch,
)


class ChevronAttentionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.address_memory = torch.tensor(
            [[4.0, 0.0], [4.0, 0.0], [0.0, 4.0], [0.0, 4.0]]
        )
        self.content_memory = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
        )
        self.config = ChevronAttentionConfig()

    def test_empty_memory_routes_everything_to_residual(self) -> None:
        output = chevron_attention(
            torch.tensor([1.0, 0.0]),
            torch.tensor([1.0, 0.0]),
            torch.empty((0, 2)),
            torch.empty((0, 2)),
            config=self.config,
        )
        self.assertEqual(tuple(output.alpha.shape), (1, 0))
        self.assertTrue(torch.equal(output.total_residual, torch.ones(1)))
        self.assertTrue(torch.equal(output.read_output, torch.zeros((1, 2))))
        self.assertTrue(output.allocate_provisional.item())

    def test_conservation_and_gate_ranges(self) -> None:
        output = chevron_attention(
            torch.tensor([4.0, 0.0]),
            torch.tensor([1.0, 0.0]),
            self.address_memory,
            self.content_memory,
            config=self.config,
        )
        self.assertTrue(torch.allclose(output.alpha.sum(-1), torch.ones(1), atol=1e-6))
        conserved = output.read_mass.sum(-1) + output.total_residual
        self.assertTrue(torch.allclose(conserved, torch.ones(1), atol=1e-6))
        for gate in (
            output.alpha,
            output.mismatch,
            output.read_assent,
            output.read_mass,
            output.slot_residual,
            output.total_residual,
            output.write_assent,
            output.write_gate,
        ):
            self.assertTrue(torch.all(gate >= 0.0).item())
            self.assertTrue(torch.all(gate <= 1.0).item())

    def test_normalized_mismatch_has_declared_scale(self) -> None:
        evidence = torch.tensor([[1.0, 0.0]])
        retained = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])
        mismatch = normalized_cosine_mismatch(evidence, retained)
        expected = torch.tensor([[0.0, 0.5, 1.0]])
        self.assertTrue(torch.allclose(mismatch, expected, atol=1e-6))

    def test_positive_threshold_margin_is_enforced(self) -> None:
        with self.assertRaises(ValueError):
            ChevronAttentionConfig(read_threshold=0.1, write_threshold=0.1)
        with self.assertRaises(ValueError):
            ChevronAttentionConfig(read_threshold=0.1, write_threshold=0.2)
        with self.assertRaises(ValueError):
            ChevronAttentionConfig(read_slope=0.0)

    def test_eligibility_bounds_are_enforced(self) -> None:
        with self.assertRaises(ValueError):
            chevron_attention(
                torch.tensor([4.0, 0.0]),
                torch.tensor([1.0, 0.0]),
                self.address_memory,
                self.content_memory,
                eligibility=torch.tensor([1.0, -0.1, 1.0, 1.0]),
                config=self.config,
            )

    def test_top_k_preserves_dropped_mass_as_residual(self) -> None:
        config = ChevronAttentionConfig(top_k=1)
        output = chevron_attention(
            torch.tensor([4.0, 0.0]),
            torch.tensor([1.0, 0.0]),
            self.address_memory,
            self.content_memory,
            config=config,
        )
        self.assertEqual(int(output.selected_mask.sum().item()), 1)
        conserved = output.read_mass.sum(-1) + output.total_residual
        self.assertTrue(torch.allclose(conserved, torch.ones(1), atol=1e-6))
        dropped = output.alpha.masked_select(~output.selected_mask).sum()
        rejected_selected = (
            output.alpha
            * (1.0 - output.read_assent)
            * output.selected_mask.to(output.alpha.dtype)
        ).sum()
        self.assertTrue(
            torch.allclose(output.total_residual.squeeze(), dropped + rejected_selected, atol=1e-6)
        )

    def test_fixed_address_changed_content_separates_retrieval_and_assent(self) -> None:
        address = torch.tensor([4.0, 0.0])
        first = chevron_attention(
            address,
            torch.tensor([1.0, 0.0]),
            self.address_memory,
            self.content_memory,
            config=self.config,
        )
        second = chevron_attention(
            address,
            torch.tensor([0.0, 1.0]),
            self.address_memory,
            self.content_memory,
            config=self.config,
        )
        self.assertTrue(torch.equal(first.alpha, second.alpha))
        self.assertGreater(first.read_assent[0, 0].item(), 0.99)
        self.assertLess(first.read_assent[0, 1].item(), 0.01)
        self.assertGreater(second.read_assent[0, 1].item(), 0.99)
        self.assertLess(second.read_assent[0, 0].item(), 0.01)

    def test_write_permission_gates_content_and_address_updates(self) -> None:
        zero_gate = torch.zeros(4)
        updated_a, updated_n = apply_convex_write(
            self.address_memory,
            self.content_memory,
            torch.tensor([3.0, 1.0]),
            torch.tensor([0.7, 0.7]),
            zero_gate,
            eta_address=0.4,
            eta_content=0.2,
        )
        self.assertTrue(torch.equal(updated_a, self.address_memory))
        self.assertTrue(torch.equal(updated_n, self.content_memory))

    def test_convex_update_stays_between_old_and_target(self) -> None:
        gate = torch.tensor([1.0, 0.5, 0.0, 0.0])
        target_a = torch.tensor([2.0, 2.0])
        target_n = torch.tensor([0.5, 0.5])
        updated_a, updated_n = apply_convex_write(
            self.address_memory,
            self.content_memory,
            target_a,
            target_n,
            gate,
            eta_address=0.5,
            eta_content=0.25,
        )
        for old, target, updated in (
            (self.address_memory, target_a, updated_a),
            (self.content_memory, target_n, updated_n),
        ):
            lower = torch.minimum(old, target)
            upper = torch.maximum(old, target)
            self.assertTrue(torch.all(updated >= lower).item())
            self.assertTrue(torch.all(updated <= upper).item())

    def test_gradients_cross_retrieval_and_assent_paths(self) -> None:
        address = torch.tensor([4.0, 0.1], requires_grad=True)
        evidence = torch.tensor([0.9, 0.1], requires_grad=True)
        address_memory = self.address_memory.clone().requires_grad_()
        content_memory = self.content_memory.clone().requires_grad_()
        output = chevron_attention(
            address,
            evidence,
            address_memory,
            content_memory,
            config=self.config,
        )
        loss = output.read_output.square().sum() + output.total_residual.sum()
        loss.backward()
        for tensor in (address, evidence, address_memory, content_memory):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all().item())


if __name__ == "__main__":
    unittest.main()
