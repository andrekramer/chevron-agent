from __future__ import annotations

import unittest

from experiments.experiment_010_confirmation import run_confirmation
from experiments.experiment_010_retrospective_policy import (
    CONDITIONS,
    ExperimentConfig,
)


class RetrospectiveConfirmationTests(unittest.TestCase):
    def test_confirmation_keeps_all_frozen_conditions(self) -> None:
        self.assertEqual(
            CONDITIONS,
            (
                "direct_update",
                "retrospective_protected",
                "retrospective_fast_veto",
                "retrospective_immediate_write",
            ),
        )

    def test_confirmation_seed_boundary_is_enforced(self) -> None:
        config = ExperimentConfig(
            stream_steps=40,
            shift_step=10,
            retention_step=30,
        )
        with self.assertRaisesRegex(ValueError, "101,000,000"):
            run_confirmation(
                config,
                seeds=1,
                seed_offset=100_999_999,
            )


if __name__ == "__main__":
    unittest.main()
