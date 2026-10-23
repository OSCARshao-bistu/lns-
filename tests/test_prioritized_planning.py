import unittest

from mapf_lns import (
    Agent,
    MAPFInstance,
    PlanningFailure,
    find_first_conflict,
    prioritized_planning,
)


class PrioritizedPlanningTests(unittest.TestCase):
    def test_pp_builds_conflict_free_solution_with_wait(self) -> None:
        instance = MAPFInstance(
            width=3,
            height=2,
            obstacles=set(),
            agents=[Agent((0, 0), (2, 0)), Agent((2, 0), (0, 0))],
        )
        solution = prioritized_planning(instance)
        self.assertEqual(solution[0], [(0, 0), (1, 0), (2, 0)])
        self.assertEqual(solution[1][0], (2, 0))
        self.assertEqual(solution[1][-1], (0, 0))
        self.assertIsNone(find_first_conflict(solution))
        self.assertEqual(
            solution.compute_soc(), sum(len(path) - 1 for path in solution.values())
        )

    def test_pp_raises_when_higher_priority_goal_blocks_corridor(self) -> None:
        instance = MAPFInstance(
            width=3,
            height=1,
            obstacles=set(),
            agents=[Agent((0, 0), (1, 0)), Agent((2, 0), (0, 0))],
        )
        with self.assertRaisesRegex(PlanningFailure, "agent 1"):
            prioritized_planning(instance)

    def test_custom_order_must_be_a_permutation(self) -> None:
        instance = MAPFInstance(2, 1, set(), [Agent((0, 0), (0, 0))])
        with self.assertRaisesRegex(ValueError, "permutation"):
            prioritized_planning(instance, [])


if __name__ == "__main__":
    unittest.main()
