from __future__ import annotations

import unittest

import torch

from experiments.experiment_010_retrospective_policy import Candidate, make_lifetime
from experiments.experiment_011_persistent_identity import ExperimentConfig
from experiments.experiment_012_empty_memory import (
    CONDITIONS,
    ColdStartAgent,
    _probe,
    run_bootstrap_lifetime,
)


class EmptyMemoryBootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ExperimentConfig(
            stream_steps=40,
            shift_step=10,
            retention_step=30,
            reward_flip_probability=0.0,
        )
        self.lifetime = make_lifetime(self.config, 113_000_000)

    def test_empty_memory_routes_all_mass_to_unresolved_identity(self) -> None:
        agent = ColdStartAgent(
            "retrospective_protected",
            self.config,
            self.lifetime,
            seed=1,
        )
        trace = agent.distribution(self.lifetime.events[0].observation)
        self.assertEqual(len(agent.slots), 0)
        self.assertEqual(trace.identity_mass.numel(), 0)
        self.assertEqual(float(trace.q_identity), 1.0)
        self.assertEqual(trace.candidate_kind, "new_identity")
        self.assertIsNone(trace.target_memory_id)

    def test_first_supported_candidate_can_allocate_into_empty_memory(self) -> None:
        agent = ColdStartAgent(
            "retrospective_protected",
            self.config,
            self.lifetime,
            seed=2,
        )
        event = self.lifetime.events[0]
        positives = torch.zeros(self.config.action_dim, dtype=torch.long)
        positives[event.correct_action] = self.config.identity_promotion_support
        candidate = Candidate(
            candidate_id=0,
            kind="new_identity",
            family=event.observation.family,
            target_memory_id=None,
            identity=event.observation.identity.clone(),
            observations=2,
            pending_events=set(),
            positive_actions=positives,
            last_seen=event.observation.event_id,
        )
        self.assertTrue(agent._allocate(candidate, event.correct_action))
        self.assertEqual(len(agent.slots), 1)
        self.assertEqual(agent.slots[0].origin_category, event.category)
        self.assertEqual(agent.counters.duplicate_allocations, 0)

    def test_short_cold_lifetime_runs_without_preloaded_slots(self) -> None:
        metrics = run_bootstrap_lifetime(
            "learned_cold_protected",
            "retrospective_protected",
            self.config,
            self.lifetime,
            seed=113_000_000,
            cold_start=True,
        )
        self.assertGreaterEqual(metrics.core_promotions_before_shift, 0.0)
        self.assertLessEqual(metrics.core_promotions_before_shift, 8.0)
        self.assertTrue(torch.isfinite(torch.tensor(metrics.return_per_decision)).item())

    def test_shift_probe_scores_the_initial_not_reversed_policy(self) -> None:
        from experiments.experiment_010_retrospective_policy import RetrospectiveAgent

        agent = RetrospectiveAgent(
            "retrospective_protected",
            self.config,
            self.lifetime,
            seed=3,
            revalidate_identity_promotion=True,
        )
        core = tuple(self.lifetime.stable_categories) + tuple(
            self.lifetime.reversed_categories
        )
        self.assertEqual(
            _probe(
                agent,
                self.lifetime,
                core,
                event_offset=100,
                actions=self.lifetime.initial_actions,
            ),
            1.0,
        )

    def test_conditions_include_bootstrap_and_preloaded_controls(self) -> None:
        self.assertEqual(
            CONDITIONS,
            (
                "oracle_preloaded_protected",
                "learned_preloaded_protected",
                "oracle_cold_protected",
                "raw_sensor_cold_protected",
                "learned_cold_protected",
                "learned_cold_direct",
            ),
        )


if __name__ == "__main__":
    unittest.main()
