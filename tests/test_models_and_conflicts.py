import unittest

from mapf_lns import (
    Agent,
    Conflict,
    MAPFInstance,
    Solution,
    find_all_conflicts,
    find_first_conflict,
)


class ModelsAndConflictsTests(unittest.TestCase):
    def test_soc_counts_moves_and_waits_until_arrival(self) -> None:
        solution = Solution({0: [(0, 0), (1, 0), (2, 0)], 1: [(3, 0)]})
        self.assertEqual(solution.compute_soc(), 2)

    def test_vertex_conflict_after_short_path_has_finished(self) -> None:
        solution = Solution({0: [(0, 0), (1, 0)], 1: [(2, 0), (2, 0), (1, 0)]})
        self.assertEqual(find_first_conflict(solution), Conflict(0, 1, 2, "vertex"))

    def test_edge_swap_conflict(self) -> None:
        solution = Solution({0: [(0, 0), (1, 0)], 1: [(1, 0), (0, 0)]})
        self.assertEqual(find_all_conflicts(solution), [Conflict(0, 1, 1, "edge")])

    def test_instance_rejects_blocked_endpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "lies on an obstacle"):
            MAPFInstance(2, 2, {(0, 0)}, [Agent((0, 0), (1, 1))])


if __name__ == "__main__":
    unittest.main()
