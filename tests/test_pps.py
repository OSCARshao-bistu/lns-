import unittest
from unittest.mock import patch

from mapf_lns.conflicts import find_all_conflicts
from mapf_lns.models import Agent, MAPFInstance, Solution
from mapf_lns.pps import solve_pps
from mapf_lns.prioritized_planning import PlanningFailure


class PPSTests(unittest.TestCase):
    def assert_legal(self, instance, solution, max_steps=1000):
        self.assertIsInstance(solution, Solution)
        self.assertEqual(set(solution), set(range(len(instance.agents))))
        self.assertEqual(find_all_conflicts(solution), [])
        self.assertLessEqual(len({len(path) for path in solution.values()}), 1)
        for agent_id, path in solution.items():
            self.assertEqual(path[0], instance.agents[agent_id].start)
            self.assertEqual(path[-1], instance.agents[agent_id].goal)
            self.assertLessEqual(len(path) - 1, max_steps)
            self.assertTrue(all(instance.traversable(cell) for cell in path))
            for source, target in zip(path, path[1:]):
                self.assertLessEqual(abs(source[0] - target[0]) + abs(source[1] - target[1]), 1)

    def test_crossing_agents_on_empty_5x5(self):
        instance = MAPFInstance(5, 5, set(), [
            Agent((0, 2), (4, 2)), Agent((2, 0), (2, 4)),
        ])
        self.assert_legal(instance, solve_pps(instance))

    def test_opposing_agents_in_single_width_corridor_fail(self):
        for width, height, far_end in [(5, 1, (4, 0)), (1, 5, (0, 4))]:
            with self.subTest(width=width, height=height):
                instance = MAPFInstance(width, height, set(), [
                    Agent((0, 0), far_end), Agent(far_end, (0, 0)),
                ])
                with self.assertRaisesRegex(PlanningFailure, 'PPS deadlock'):
                    solve_pps(instance)

    def test_push_moves_blocker_and_follower_in_one_tick(self):
        instance = MAPFInstance(3, 1, set(), [
            Agent((0, 0), (1, 0)), Agent((1, 0), (2, 0)),
        ])
        solution = solve_pps(instance, max_steps=1)
        self.assert_legal(instance, solution, max_steps=1)
        self.assertEqual(solution[0], [(0, 0), (1, 0)])
        self.assertEqual(solution[1], [(1, 0), (2, 0)])

    def test_swap_uses_side_cells_and_waits_without_edge_conflict(self):
        instance = MAPFInstance(2, 2, set(), [
            Agent((0, 0), (1, 0)), Agent((1, 0), (0, 0)),
        ])
        solution = solve_pps(instance, max_steps=3)
        self.assert_legal(instance, solution, max_steps=3)
        self.assertEqual(solution[0], [(0, 0), (0, 1), (1, 1), (1, 0)])
        self.assertEqual(solution[1], [(1, 0), (1, 0), (0, 0), (0, 0)])

    def test_pushed_goal_agent_returns_to_its_goal(self):
        instance = MAPFInstance(3, 2, set(), [
            Agent((0, 0), (2, 0)), Agent((1, 0), (1, 0)),
        ])
        solution = solve_pps(instance)
        self.assert_legal(instance, solution)
        self.assertIn((2, 0), solution[1])

    def test_obstacle_detour_uses_only_local_moves(self):
        instance = MAPFInstance(3, 3, {(1, 0)}, [Agent((0, 0), (2, 0))])
        self.assert_legal(instance, solve_pps(instance))

    def test_max_steps_includes_swap_substeps(self):
        instance = MAPFInstance(2, 2, set(), [
            Agent((0, 0), (1, 0)), Agent((1, 0), (0, 0)),
        ])
        with self.assertRaisesRegex(PlanningFailure, 'max_steps=2'):
            solve_pps(instance, max_steps=2)

    def test_physically_trapped_agent_reports_deadlock(self):
        instance = MAPFInstance(3, 1, {(1, 0)}, [Agent((0, 0), (2, 0))])
        with self.assertRaisesRegex(PlanningFailure, '^PPS deadlock$'):
            solve_pps(instance)

    def test_empty_and_already_solved_instances_need_no_steps(self):
        for agents in [[], [Agent((0, 0), (0, 0))]]:
            instance = MAPFInstance(1, 1, set(), agents)
            self.assert_legal(instance, solve_pps(instance, max_steps=0), max_steps=0)

    def test_zero_budget_fails_for_unsolved_instance(self):
        instance = MAPFInstance(2, 1, set(), [Agent((0, 0), (1, 0))])
        with self.assertRaisesRegex(PlanningFailure, 'max_steps=0'):
            solve_pps(instance, max_steps=0)

    def test_negative_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            solve_pps(MAPFInstance(1, 1, set(), []), max_steps=-1)

    def test_duplicate_starts_or_goals_are_rejected(self):
        for agents in [
            [Agent((0, 0), (0, 0)), Agent((0, 0), (1, 0))],
            [Agent((0, 0), (0, 0)), Agent((1, 0), (0, 0))],
        ]:
            with self.subTest(agents=agents), self.assertRaises(PlanningFailure):
                solve_pps(MAPFInstance(2, 1, set(), agents))

    def test_no_existing_astar_or_pp_solver_is_called(self):
        with patch('mapf_lns.prioritized_planning.space_time_astar',
                   side_effect=AssertionError('A* called')), \
             patch('mapf_lns.lra_star._space_only_astar',
                   side_effect=AssertionError('spatial A* called')), \
             patch('mapf_lns.prioritized_planning.prioritized_planning',
                   side_effect=AssertionError('PP called')):
            instance = MAPFInstance(2, 2, set(), [
                Agent((0, 0), (1, 0)), Agent((1, 0), (0, 0)),
            ])
            self.assert_legal(instance, solve_pps(instance))


if __name__ == '__main__':
    unittest.main()
