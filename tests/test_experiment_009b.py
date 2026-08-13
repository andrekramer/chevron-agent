from __future__ import annotations

import unittest

from experiments.experiment_009_dual_relation import ExperimentConfig
from experiments.experiment_009b_capacity_confirmation import (
    CONDITIONS,
    run_confirmation,
)


class CapacityConfirmationTests(unittest.TestCase):
    def test_confirmation_conditions_are_narrow(self) -> None:
        self.assertEqual(
            CONDITIONS,
            (
                "dual_shared_4",
                "dual_shared_8",
                "identity_only_shared_4",
            ),
        )

    def test_confirmation_seed_boundary_is_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "92,000,000"):
            run_confirmation(
                ExperimentConfig(stream_steps=10),
                seeds=1,
                seed_offset=91_999_999,
            )


if __name__ == "__main__":
    unittest.main()
