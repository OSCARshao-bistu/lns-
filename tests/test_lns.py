import random
import unittest
from unittest.mock import patch

from mapf_lns import (
    Agent,
    AgentBasedDestroy,
    LNSConfig,
    MAPFInstance,
    NeighborhoodDestroy,
    PlanningFailure,
    RandomDestroy,
    Solution,
    find_all_conflicts,
    prioritized_planning,
    repair_prioritized,
    run_lns,
)


def _ten_agent_instance() -> MAPFInstance:
    return MAPFInstance(
        10,
        2,
        set(),
        [Agent((agent_id, 0), (agent_id, 0)) for agent_id in range(10)],
    )


def _four_agent_instance() -> MAPFInstance:
    starts = [(0, 0), (4, 0), (0, 4), (4, 4)]
    goals = [(4, 4), (0, 4), (4, 0), (0, 0)]
    return MAPFInstance(
        5, 5, set(), [Agent(start, goal) for start, goal in zip(starts, goals)]
    )


class DestroyTests(unittest.TestCase):
    def test_random_destroy_uses_ceil_neighborhood_size(self) -> None:
        instance = _ten_agent_instance()
        solution = prioritized_planning(instance)
        destroy = RandomDestroy()
        self.assertEqual(len(destroy.destroy(instance, solution, random.Random(0), 0.2)), 2)
        self.assertEqual(
            len(destroy.destroy(instance, solution, random.Random(0), 0.05)), 1
        )

    def test_agent_based_destroy_is_seed_deterministic(self) -> None:
        instance = MAPFInstance(
            4,
            2,
            set(),
            [Agent((agent_id, 0), (agent_id, 0)) for agent_id in range(4)],
        )
        # Conflict edges: 0--1 and 1--2; agent 3 is disconnected.
        solution = Solution(
            {
                0: [(0, 0), (1, 0)],
                1: [(2, 0), (1, 0)],
                2: [(3, 0), (2, 0), (1, 0)],
                3: [(3, 1)],
            }
        )
        destroy = AgentBasedDestroy()
        first = destroy.destroy(instance, solution, random.Random(42), 0.75)
        second = destroy.destroy(instance, solution, random.Random(42), 0.75)
        self.assertEqual(first, second)
        # Restricted walks may discover no usable neighborhood.
        self.assertIn(len(first), (0, 3))

    def test_neighborhood_destroy_picks_seed_uniformly_and_includes_it(self) -> None:
        instance = _ten_agent_instance()
        solution = prioritized_planning(instance)
        solution[9] = [(9, 0), (9, 0), (9, 0)]
        destroy = NeighborhoodDestroy()

        selections = {
            frozenset(destroy.destroy(instance, solution, random.Random(seed), 0.1))
            for seed in range(20)
        }
        self.assertTrue(all(len(selected) == 1 for selected in selections))
        self.assertGreater(len(selections), 1)
        self.assertNotEqual(selections, {frozenset({9})})

    def test_neighborhood_destroy_expands_by_map_hops(self) -> None:
        class SeedZeroRng:
            def choice(self, population: range) -> int:
                self.assert_population = population
                return 0

        instance = MAPFInstance(
            7,
            7,
            set(),
            [
                Agent((3, 3), (3, 3)),
                Agent((3, 4), (3, 4)),
                Agent((5, 5), (5, 5)),
                Agent((0, 0), (0, 0)),
                Agent((6, 6), (6, 6)),
            ],
        )
        solution = Solution(
            {
                agent_id: [agent.start]
                for agent_id, agent in enumerate(instance.agents)
            }
        )
        selected = NeighborhoodDestroy().destroy(
            instance, solution, SeedZeroRng(), 0.6  # type: ignore[arg-type]
        )
        self.assertEqual(selected, {0, 1, 2})


class RepairTests(unittest.TestCase):
    def test_repair_preserves_all_fixed_paths_exactly(self) -> None:
        instance = _four_agent_instance()
        original = prioritized_planning(instance)
        repaired = repair_prioritized(instance, original, {0}, random.Random(0))
        for agent_id in (1, 2, 3):
            self.assertEqual(repaired[agent_id], original[agent_id])
            self.assertIs(repaired[agent_id], original[agent_id])
        self.assertEqual(original, prioritized_planning(instance))

    def test_repair_failure_does_not_mutate_input(self) -> None:
        instance = MAPFInstance(
            3,
            1,
            set(),
            [Agent((0, 0), (2, 0)), Agent((2, 0), (2, 0))],
        )
        original = Solution({0: [(0, 0), (1, 0), (2, 0)], 1: [(2, 0)]})
        snapshot = Solution({agent_id: list(path) for agent_id, path in original.items()})
        with self.assertRaises(PlanningFailure):
            repair_prioritized(instance, original, {0}, random.Random(0))
        self.assertEqual(original, snapshot)

    def test_repair_order_is_randomized_by_rng(self) -> None:
        instance = MAPFInstance(
            3,
            2,
            set(),
            [Agent((0, 0), (2, 1)), Agent((1, 0), (2, 0))],
        )
        original = prioritized_planning(instance)
        order_01 = repair_prioritized(
            instance, original, {0, 1}, random.Random(0)
        )
        order_10 = repair_prioritized(
            instance, original, {0, 1}, random.Random(1)
        )
        self.assertNotEqual(order_01, order_10)
        self.assertNotEqual(order_01.compute_soc(), order_10.compute_soc())


class LNSTests(unittest.TestCase):
    def test_no_deterministic_fixed_point(self) -> None:
        instance = MAPFInstance(
            4,
            3,
            set(),
            [
                Agent((0, 0), (1, 2)),
                Agent((2, 1), (1, 1)),
                Agent((0, 2), (2, 0)),
            ],
        )
        _, logs = run_lns(
            instance, LNSConfig(max_iterations=200, gamma=0.5, seed=0)
        )
        has_improvement = any(
            entry.soc_candidate is not None
            and entry.soc_candidate < entry.soc_current_before
            for entry in logs
        )
        accepted_count = sum(entry.accepted for entry in logs)
        self.assertTrue(has_improvement or accepted_count < len(logs))

    def test_repair_failure_is_logged_and_keeps_current_and_best(self) -> None:
        instance = MAPFInstance(2, 2, set(), [Agent((0, 0), (1, 1))])
        initial = prioritized_planning(instance)
        with patch(
            "mapf_lns.lns.repair_prioritized",
            side_effect=PlanningFailure("forced repair failure"),
        ):
            best, logs = run_lns(instance, LNSConfig(max_iterations=1, seed=0))
        self.assertEqual(best, initial)
        self.assertEqual(len(logs), 1)
        self.assertFalse(logs[0].accepted)
        self.assertIsNone(logs[0].soc_candidate)
        self.assertEqual(logs[0].soc_current_before, logs[0].soc_best_after)

    def test_greedy_acceptance_condition_for_every_iteration(self) -> None:
        _, logs = run_lns(
            _four_agent_instance(), LNSConfig(max_iterations=30, gamma=0.3, seed=0)
        )
        self.assertEqual(len(logs), 30)
        for entry in logs:
            if entry.accepted:
                self.assertIsNotNone(entry.soc_candidate)
                self.assertLessEqual(entry.soc_candidate, entry.soc_current_before)
            elif entry.soc_candidate is not None:
                self.assertGreater(entry.soc_candidate, entry.soc_current_before)

    def test_end_to_end_best_solution_is_conflict_free(self) -> None:
        best, logs = run_lns(
            _four_agent_instance(), LNSConfig(max_iterations=50, gamma=0.3, seed=9)
        )
        self.assertEqual(len(logs), 50)
        self.assertEqual(find_all_conflicts(best), [])

    def test_seed_makes_runs_reproducible(self) -> None:
        config = LNSConfig(max_iterations=20, gamma=0.3, seed=42)
        best_a, logs_a = run_lns(_four_agent_instance(), config)
        best_b, logs_b = run_lns(_four_agent_instance(), config)
        self.assertEqual(best_a.compute_soc(), best_b.compute_soc())
        self.assertEqual(len(logs_a), len(logs_b))
        self.assertEqual(logs_a, logs_b)

    def test_initial_pp_failure_propagates_unchanged(self) -> None:
        instance = MAPFInstance(
            3,
            1,
            set(),
            [Agent((0, 0), (1, 0)), Agent((2, 0), (0, 0))],
        )
        with self.assertRaisesRegex(PlanningFailure, "agent 1"):
            run_lns(instance, LNSConfig(max_iterations=1))


if __name__ == "__main__":
    unittest.main()
