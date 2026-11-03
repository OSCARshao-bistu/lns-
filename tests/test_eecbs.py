import math
import time
import unittest
from unittest.mock import patch

from mapf_lns.conflicts import find_all_conflicts
from mapf_lns.eecbs import (
    _constrained_astar, _CTNode, _EdgeConstraint, _FocalQueue,
    _VertexConstraint, solve_eecbs,
)
from mapf_lns.models import Agent, Conflict, MAPFInstance, Solution
from mapf_lns.prioritized_planning import PlanningFailure


class EECBSTests(unittest.TestCase):
    def assert_legal(self, instance, solution):
        self.assertIsInstance(solution, Solution)
        self.assertEqual(set(solution), set(range(len(instance.agents))))
        self.assertEqual(find_all_conflicts(solution), [])
        for agent_id, path in solution.items():
            self.assertEqual(path[0], instance.agents[agent_id].start)
            self.assertEqual(path[-1], instance.agents[agent_id].goal)
            self.assertTrue(all(instance.traversable(cell) for cell in path))
            for source, target in zip(path, path[1:]):
                self.assertLessEqual(abs(source[0] - target[0]) + abs(source[1] - target[1]), 1)

    def test_crossing_agents_have_theoretical_optimal_soc(self):
        instance = MAPFInstance(5, 5, set(), [
            Agent((0, 2), (4, 2)), Agent((2, 0), (2, 4)),
        ])
        solution = solve_eecbs(instance)
        self.assert_legal(instance, solution)
        # Independent lower bound 4+4=8 collides at t=2. One wait suffices.
        self.assertEqual(solution.compute_soc(), 9)

    def test_two_cell_swap_fails_within_five_seconds(self):
        instance = MAPFInstance(1, 2, set(), [
            Agent((0, 0), (0, 1)), Agent((0, 1), (0, 0)),
        ])
        start = time.perf_counter()
        with self.assertRaises(PlanningFailure):
            solve_eecbs(instance, time_limit_seconds=.05)
        self.assertLess(time.perf_counter() - start, 5.)

    def test_one_obstacle_requires_shortest_detour(self):
        instance = MAPFInstance(3, 3, {(1, 0)}, [Agent((0, 0), (2, 0))])
        solution = solve_eecbs(instance)
        self.assert_legal(instance, solution)
        self.assertEqual(solution.compute_soc(), 4)

    def test_edge_swap_conflict_is_split_and_resolved(self):
        instance = MAPFInstance(2, 2, set(), [
            Agent((0, 0), (1, 0)), Agent((1, 0), (0, 0)),
        ])
        solution = solve_eecbs(instance, weight=0)
        self.assert_legal(instance, solution)
        self.assertEqual(solution.compute_soc(), 4)

    def test_conflict_after_shorter_path_finishes_is_resolved(self):
        instance = MAPFInstance(3, 2, set(), [
            Agent((0, 0), (1, 0)), Agent((2, 0), (0, 0)),
        ])
        self.assert_legal(instance, solve_eecbs(instance))

    def test_disconnected_goal_is_proven_unreachable(self):
        instance = MAPFInstance(3, 1, {(1, 0)}, [Agent((0, 0), (2, 0))])
        with self.assertRaisesRegex(PlanningFailure, 'cannot reach goal'):
            solve_eecbs(instance)

    def test_zero_time_budget_raises(self):
        with self.assertRaisesRegex(PlanningFailure, 'time limit'):
            solve_eecbs(MAPFInstance(1, 1, set(), []), time_limit_seconds=0)

    def test_empty_and_already_solved_instances(self):
        for agents in [[], [Agent((0, 0), (0, 0))]]:
            instance = MAPFInstance(1, 1, set(), agents)
            solution = solve_eecbs(instance)
            self.assert_legal(instance, solution)
            self.assertEqual(solution.compute_soc(), 0)

    def test_duplicate_starts_and_permanent_goals_are_unsolvable(self):
        for agents in [
            [Agent((0, 0), (0, 0)), Agent((0, 0), (1, 0))],
            [Agent((0, 0), (0, 0)), Agent((1, 0), (0, 0))],
        ]:
            with self.subTest(agents=agents), self.assertRaisesRegex(PlanningFailure, 'no solution'):
                solve_eecbs(MAPFInstance(2, 1, set(), agents))

    def test_invalid_parameters(self):
        instance = MAPFInstance(1, 1, set(), [])
        for value in [-1., math.inf, math.nan]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    solve_eecbs(instance, weight=value)
                with self.assertRaises(ValueError):
                    solve_eecbs(instance, time_limit_seconds=value)

    def test_solver_is_independent_of_existing_astar(self):
        with patch('mapf_lns.prioritized_planning.space_time_astar',
                   side_effect=AssertionError('existing A* called')):
            instance = MAPFInstance(3, 1, set(), [Agent((0, 0), (2, 0))])
            self.assert_legal(instance, solve_eecbs(instance))


class LowLevelConstraintTests(unittest.TestCase):
    def setUp(self):
        self.instance = MAPFInstance(3, 1, set(), [Agent((0, 0), (2, 0))])

    def test_vertex_constraint_forces_wait(self):
        result = _constrained_astar(self.instance, 0,
                                   frozenset({_VertexConstraint((1, 0), 1)}))
        self.assertEqual(result, ([(0, 0), (0, 0), (1, 0), (2, 0)], 3))

    def test_directed_edge_constraint_forces_wait(self):
        result = _constrained_astar(self.instance, 0,
                                   frozenset({_EdgeConstraint((0, 0), (1, 0), 0)}))
        self.assertEqual(result, ([(0, 0), (0, 0), (1, 0), (2, 0)], 3))

    def test_reverse_edge_constraint_does_not_block_forward_move(self):
        result = _constrained_astar(self.instance, 0,
                                   frozenset({_EdgeConstraint((1, 0), (0, 0), 0)}))
        self.assertEqual(result, ([(0, 0), (1, 0), (2, 0)], 2))

    def test_forbidden_start_has_no_path(self):
        self.assertIsNone(_constrained_astar(self.instance, 0,
                          frozenset({_VertexConstraint((0, 0), 0)})))

    def test_future_goal_vertex_constraint_prevents_early_termination(self):
        instance = MAPFInstance(2, 1, set(), [Agent((0, 0), (0, 0))])
        path, cost = _constrained_astar(instance, 0,
                                      frozenset({_VertexConstraint((0, 0), 3)}))
        self.assertEqual(cost, 4)
        self.assertEqual(path[3], (1, 0))
        self.assertEqual(path[-1], (0, 0))

    def test_future_goal_wait_edge_constraint_prevents_early_termination(self):
        instance = MAPFInstance(2, 1, set(), [Agent((0, 0), (0, 0))])
        path, cost = _constrained_astar(instance, 0,
                                      frozenset({_EdgeConstraint((0, 0), (0, 0), 2)}))
        self.assertEqual(cost, 3)
        self.assertNotEqual((path[2], path[3]), ((0, 0), (0, 0)))

    def test_low_level_enforces_deadline(self):
        with self.assertRaisesRegex(PlanningFailure, 'time limit'):
            _constrained_astar(self.instance, 0, deadline=time.perf_counter() - 1)


class FocalQueueTests(unittest.TestCase):
    @staticmethod
    def node(cost, conflicts):
        conflict = Conflict(0, 1, 1, 'vertex')
        return _CTNode((), Solution(), cost, (conflict,) * conflicts)

    def test_threshold_then_conflicts_and_refresh_after_open_min_changes(self):
        queue = _FocalQueue(weight=1.5)  # 2.5 * minimum cost, inclusive.
        cheapest = self.node(10, 5)
        boundary = self.node(25, 1)
        excluded = self.node(26, 0)
        for node in [cheapest, boundary, excluded]:
            queue.push(node)
        self.assertIs(queue.pop(), boundary)
        self.assertIs(queue.pop(), cheapest)
        # Removing the cheapest changes min_cost and admits the excluded node.
        self.assertIs(queue.pop(), excluded)
        self.assertFalse(queue)

    def test_conflict_count_precedes_cost_within_focal(self):
        queue = _FocalQueue(weight=1)
        cheap = self.node(10, 2)
        fewer_conflicts = self.node(20, 1)
        queue.push(cheap)
        queue.push(fewer_conflicts)
        self.assertIs(queue.pop(), fewer_conflicts)

    def test_cost_breaks_equal_conflict_count_ties(self):
        queue = _FocalQueue(weight=1)
        expensive = self.node(15, 1)
        cheap = self.node(10, 1)
        queue.push(expensive)
        queue.push(cheap)
        self.assertIs(queue.pop(), cheap)

    def test_zero_weight_restricts_focal_to_minimum_cost(self):
        queue = _FocalQueue(weight=0)
        cheap = self.node(10, 8)
        queue.push(self.node(11, 0))
        queue.push(cheap)
        self.assertIs(queue.pop(), cheap)


if __name__ == '__main__':
    unittest.main()
