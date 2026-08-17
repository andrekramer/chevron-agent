from __future__ import annotations

import unittest

from experiments.experiment_011_persistent_identity import ExperimentConfig
from experiments.experiment_011a_pairwise_confirmation import (
    CONDITIONS,
    run_confirmation,
)


class PairwiseIdentityConfirmationTests(unittest.TestCase):
    def test_conditions_are_the_frozen_narrow_comparison(self) -> None:
        self.assertEqual(
            CONDITIONS,
            (
                "oracle_protected",
                "raw_sensor_protected",
                "pairwise_temporal_protected",
                "pairwise_temporal_direct",
            ),
        )

    def test_encoder_seed_boundary_is_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "1200"):
            run_confirmation(
                ExperimentConfig(pretraining_steps=1),
                training_seeds=(1199,),
                lifetimes_per_encoder=1,
            )

    def test_lifetime_seed_boundary_is_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "112,000,000"):
            run_confirmation(
                ExperimentConfig(pretraining_steps=1),
                training_seeds=(1200,),
                lifetimes_per_encoder=1,
                lifetime_seed_offset=111_999_999,
            )


if __name__ == "__main__":
    unittest.main()
