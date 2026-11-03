import copy
import unittest

from mapf_lns.models import Agent, MAPFInstance, Solution
from mapf_lns.validator import FAILURE_REASONS, validate_solution


class MakespanTests(unittest.TestCase):
    def test_longest_path_counts_moves_and_waits(self):
        solution = Solution({0: [(0, 0), (1, 0)],
                             1: [(0, 1), (0, 1), (1, 1), (2, 1)]})
        self.assertEqual(solution.compute_makespan(), 3)
        self.assertEqual(solution.compute_soc(), 4)

    def test_stationary_agent_has_zero_makespan(self):
        self.assertEqual(Solution({0: [(0, 0)]}).compute_makespan(), 0)

    def test_empty_solution_has_zero_makespan(self):
        self.assertEqual(Solution().compute_makespan(), 0)

    def test_empty_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'at least one position'):
            Solution({0: []}).compute_makespan()


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.instance = MAPFInstance(3, 2, set(), [
            Agent((0, 0), (2, 0)), Agent((0, 1), (1, 1)),
        ])
        self.solution = Solution({0: [(0, 0), (1, 0), (2, 0)],
                                  1: [(0, 1), (1, 1)]})

    def test_normal_solution_is_valid(self):
        self.assertEqual(validate_solution(self.instance, self.solution), (True, ''))

    def test_waiting_is_valid(self):
        self.solution[0].insert(0, (0, 0))
        self.assertEqual(validate_solution(self.instance, self.solution), (True, ''))

    def test_missing_agent_is_unfinished(self):
        del self.solution[1]
        self.assertEqual(validate_solution(self.instance, self.solution),
                         (False, 'agent_unfinished'))

    def test_empty_agent_path_is_unfinished(self):
        self.solution[1] = []
        self.assertEqual(validate_solution(self.instance, self.solution),
                         (False, 'agent_unfinished'))

    def test_wrong_goal_is_unfinished(self):
        self.solution[0].pop()
        self.assertEqual(validate_solution(self.instance, self.solution),
                         (False, 'agent_unfinished'))

    def test_wrong_start_is_invalid(self):
        self.solution[0] = [(1, 0), (2, 0)]
        self.assertEqual(validate_solution(self.instance, self.solution), (False, 'invalid_path'))

    def test_teleport_is_invalid(self):
        self.solution[0] = [(0, 0), (2, 0)]
        self.assertEqual(validate_solution(self.instance, self.solution), (False, 'invalid_path'))

    def test_diagonal_move_is_invalid(self):
        self.solution[0] = [(0, 0), (1, 1), (2, 1), (2, 0)]
        self.assertEqual(validate_solution(self.instance, self.solution), (False, 'invalid_path'))

    def test_obstacle_crossing_is_invalid(self):
        blocked = MAPFInstance(3, 2, {(1, 0)}, self.instance.agents)
        self.assertEqual(validate_solution(blocked, self.solution), (False, 'invalid_path'))

    def test_out_of_bounds_is_invalid(self):
        self.solution[0] = [(0, 0), (0, -1), (1, -1), (2, -1), (2, 0)]
        self.assertEqual(validate_solution(self.instance, self.solution), (False, 'invalid_path'))

    def test_extra_agent_is_invalid(self):
        self.solution[2] = [(0, 0)]
        self.assertEqual(validate_solution(self.instance, self.solution), (False, 'invalid_path'))

    def test_non_integer_agent_id_is_invalid(self):
        self.solution['1'] = self.solution.pop(1)
        self.assertEqual(validate_solution(self.instance, self.solution), (False, 'invalid_path'))

    def test_malformed_paths_and_coordinates_are_invalid(self):
        for path in [None, 'path', [None], [(0,)], [(0, 0, 0)],
                     [('0', 0)], [(0., 0)], [(False, 0)]]:
            with self.subTest(path=path):
                self.solution[0] = path
                self.assertEqual(validate_solution(self.instance, self.solution),
                                 (False, 'invalid_path'))

    def test_no_solver_result_is_solver_error(self):
        for result in [None, [], 'failed', 1]:
            with self.subTest(result=result):
                self.assertEqual(validate_solution(self.instance, result), (False, 'solver_error'))

    def test_vertex_conflict(self):
        instance = MAPFInstance(3, 3, set(), [
            Agent((0, 1), (2, 1)), Agent((1, 0), (1, 2)),
        ])
        solution = Solution({0: [(0, 1), (1, 1), (2, 1)],
                             1: [(1, 0), (1, 1), (1, 2)]})
        self.assertEqual(validate_solution(instance, solution), (False, 'has_conflicts'))

    def test_edge_swap_conflict(self):
        instance = MAPFInstance(2, 1, set(), [
            Agent((0, 0), (1, 0)), Agent((1, 0), (0, 0)),
        ])
        solution = Solution({0: [(0, 0), (1, 0)], 1: [(1, 0), (0, 0)]})
        self.assertEqual(validate_solution(instance, solution), (False, 'has_conflicts'))

    def test_conflict_with_permanent_goal_occupancy(self):
        instance = MAPFInstance(3, 2, set(), [
            Agent((1, 0), (1, 0)), Agent((2, 0), (0, 0)),
        ])
        solution = Solution({0: [(1, 0)], 1: [(2, 0), (1, 0), (0, 0)]})
        self.assertEqual(validate_solution(instance, solution), (False, 'has_conflicts'))

    def test_list_coordinates_are_supported_without_mutating_input(self):
        solution = {i: [list(cell) for cell in path] for i, path in self.solution.items()}
        snapshot = copy.deepcopy(solution)
        self.assertEqual(validate_solution(self.instance, solution), (True, ''))
        self.assertEqual(solution, snapshot)

    def test_empty_instance_is_valid(self):
        self.assertEqual(validate_solution(MAPFInstance(1, 1, set(), []), Solution()),
                         (True, ''))

    def test_exact_failure_reason_vocabulary(self):
        self.assertEqual(FAILURE_REASONS, {
            'time_limit', 'has_conflicts', 'agent_unfinished', 'invalid_path', 'solver_error',
        })


if __name__ == '__main__':
    unittest.main()
