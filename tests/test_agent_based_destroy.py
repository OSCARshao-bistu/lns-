import copy
import importlib
import random
import unittest
from unittest.mock import patch

from mapf_lns import (Agent, AgentBasedDestroy, LNSConfig, MAPFInstance,
                      Solution, find_all_conflicts, prioritized_planning, run_lns)
from mapf_lns.destroy import _blocking_agents


def crossing():
    instance = MAPFInstance(3, 3, set(), [
        Agent((0, 0), (2, 1)), Agent((1, 0), (2, 0)),
        Agent((0, 2), (0, 2)),
    ])
    return instance, prioritized_planning(instance)


class AgentBasedRegressionTests(unittest.TestCase):
    def test_finds_blocker_in_a_collision_free_solution(self):
        instance, solution = crossing()
        self.assertEqual(find_all_conflicts(solution), [])
        self.assertEqual(solution.compute_soc(), 6)
        # Agent 1 has delay 2. Its shorter trial walk encounters agent 0.
        for seed in range(12):
            with self.subTest(seed=seed):
                selected = AgentBasedDestroy().destroy(instance, solution, random.Random(seed), .6)
                self.assertEqual(selected, {0, 1})

    def test_hypothetical_walk_does_not_modify_real_paths(self):
        instance, solution = crossing()
        snapshot = copy.deepcopy(solution)
        identities = {i: id(p) for i, p in solution.items()}
        AgentBasedDestroy().destroy(instance, solution, random.Random(1), .6)
        self.assertEqual(solution, snapshot)
        self.assertEqual({i: id(p) for i, p in solution.items()}, identities)

    def test_returns_discovered_subset_without_padding_to_target(self):
        instance, solution = crossing()
        instance = MAPFInstance(3, 3, set(), [
            *instance.agents, Agent((2, 2), (2, 2)),
        ])
        solution[3] = [(2, 2)]
        self.assertEqual(find_all_conflicts(solution), [])
        # Target is three; only seed 1 and blocker 0 are discovered.
        selected = AgentBasedDestroy().destroy(instance, solution, random.Random(0), .75)
        self.assertEqual(selected, {0, 1})

    def test_uses_obstacle_aware_shortest_distance(self):
        instance = MAPFInstance(3, 3, {(1, 0)}, [
            Agent((0, 0), (2, 0)), Agent((0, 2), (0, 2)), Agent((2, 2), (2, 2)),
        ])
        solution = prioritized_planning(instance)
        self.assertEqual(len(solution[0]) - 1, 4)
        self.assertEqual(AgentBasedDestroy().destroy(instance, solution, random.Random(1), .6), set())

    def test_no_random_padding_when_walk_finds_no_other_agent(self):
        instance = MAPFInstance(5, 2, set(), [
            Agent((0, 0), (1, 0)), Agent((3, 1), (3, 1)), Agent((4, 1), (4, 1)),
        ])
        solution = Solution({0: [(0, 0), (0, 0), (1, 0)], 1: [(3, 1)], 2: [(4, 1)]})
        self.assertEqual(find_all_conflicts(solution), [])
        self.assertEqual(AgentBasedDestroy().destroy(instance, solution, random.Random(1), .6), set())

    def test_tabu_chooses_next_delayed_agent_and_then_resets(self):
        instance = MAPFInstance(7, 1, set(), [
            Agent((0, 0), (0, 0)), Agent((3, 0), (3, 0)), Agent((6, 0), (6, 0)),
        ])
        solution = Solution({0: [(0, 0), (1, 0), (0, 0)],
                             1: [(3, 0), (4, 0), (3, 0)], 2: [(6, 0)]})
        destroy = AgentBasedDestroy()
        self.assertEqual([destroy._select_seed(instance, solution) for _ in range(4)], [0, 1, None, 0])

    def test_vertex_blocker(self):
        solution = Solution({0: [(0, 0)], 1: [(2, 0), (1, 0)]})
        self.assertEqual(_blocking_agents(solution, 0, (0, 0), (1, 0), 1), {1})

    def test_reverse_edge_blocker(self):
        solution = Solution({0: [(0, 0)], 1: [(1, 0), (0, 0)]})
        self.assertEqual(_blocking_agents(solution, 0, (0, 0), (1, 0), 1), {1})

    def test_permanent_goal_blocker(self):
        solution = Solution({0: [(0, 0)], 1: [(1, 0)]})
        self.assertEqual(_blocking_agents(solution, 0, (0, 0), (1, 0), 20), {1})

    def test_does_not_collect_walking_agent_itself(self):
        self.assertEqual(_blocking_agents(Solution({0: [(0, 0), (1, 0)]}), 0, (0, 0), (1, 0), 1), set())

    def test_reproducible_sequences_with_tabu_state(self):
        instance, solution = crossing()
        first, second = AgentBasedDestroy(), AgentBasedDestroy()
        rng_a, rng_b = random.Random(17), random.Random(17)
        a = [first.destroy(instance, solution, rng_a, .6) for _ in range(12)]
        b = [second.destroy(instance, solution, rng_b, .6) for _ in range(12)]
        self.assertEqual(a, b)
        self.assertIn(set(), a)
        self.assertIn({0, 1}, a)

    def test_empty_instance(self):
        self.assertEqual(AgentBasedDestroy().destroy(MAPFInstance(1, 1, set(), []), Solution(), random.Random(1), .5), set())

    def test_full_neighborhood_includes_all_agents(self):
        instance, solution = crossing()
        self.assertEqual(AgentBasedDestroy().destroy(instance, solution, random.Random(1), 1), {0, 1, 2})

    def test_size_one_is_unavailable_like_reference(self):
        instance, solution = crossing()
        self.assertEqual(AgentBasedDestroy().destroy(instance, solution, random.Random(1), .1), set())

    def test_instance_change_resets_cached_map_and_tabu(self):
        instance, solution = crossing()
        destroy = AgentBasedDestroy()
        self.assertEqual(destroy.destroy(instance, solution, random.Random(0), .6), {0, 1})
        other, other_solution = crossing()
        self.assertEqual(destroy.destroy(other, other_solution, random.Random(0), .6), {0, 1})

    def test_unavailable_neighborhood_skips_repair_and_logs_reason(self):
        instance, solution = crossing()
        module = importlib.import_module('mapf_lns.lns')
        with patch.object(module, 'RandomDestroy', side_effect=AgentBasedDestroy), \
             patch.object(module, 'NeighborhoodDestroy', side_effect=AgentBasedDestroy), \
             patch.object(module, 'repair_prioritized') as repair:
            best, logs = run_lns(instance, LNSConfig(max_iterations=3, gamma=.1))
        repair.assert_not_called()
        self.assertEqual(best, solution)
        self.assertTrue(all(log.failure_reason == 'neighborhood_unavailable' for log in logs))
        self.assertTrue(all(not log.accepted and log.soc_candidate is None for log in logs))

    def test_repair_failure_has_a_distinct_reason(self):
        from mapf_lns import PlanningFailure
        instance, _ = crossing()
        module = importlib.import_module('mapf_lns.lns')
        with patch.object(module, 'repair_prioritized', side_effect=PlanningFailure('forced')):
            _, logs = run_lns(instance, LNSConfig(max_iterations=3, gamma=1))
        self.assertTrue(all(log.failure_reason == 'repair_failed' for log in logs))

    def test_lns_remains_feasible_monotone_and_reproducible(self):
        instance, initial = crossing()
        for seed in range(5):
            config = LNSConfig(max_iterations=30, gamma=.6, seed=seed)
            best, logs = run_lns(instance, config)
            repeated_best, repeated_logs = run_lns(instance, config)
            self.assertEqual(find_all_conflicts(best), [])
            self.assertLessEqual(best.compute_soc(), initial.compute_soc())
            self.assertEqual((best, logs), (repeated_best, repeated_logs))
            self.assertTrue(all(b.soc_best_after <= a.soc_best_after for a, b in zip(logs, logs[1:])))
            self.assertTrue(all(log.soc_candidate <= log.soc_current_before for log in logs if log.accepted))


if __name__ == '__main__':
    unittest.main()
