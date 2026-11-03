import unittest

from mapf_lns.conflicts import find_all_conflicts
from mapf_lns.lra_star import solve_lra_star
from mapf_lns.models import Agent, MAPFInstance
from mapf_lns.prioritized_planning import PlanningFailure


class LRAStarTests(unittest.TestCase):
    def test_crossing_agents_are_repaired(self) -> None:
        instance = MAPFInstance(
            width=5,
            height=5,
            obstacles=set(),
            agents=[
                Agent(start=(0, 0), goal=(4, 4)),
                Agent(start=(4, 0), goal=(0, 4)),
            ],
        )

        solution = solve_lra_star(instance)

        self.assertEqual(find_all_conflicts(solution), [])
        self.assertEqual(solution[0][0], instance.agents[0].start)
        self.assertEqual(solution[1][0], instance.agents[1].start)
        self.assertEqual(solution[0][-1], instance.agents[0].goal)
        self.assertEqual(solution[1][-1], instance.agents[1].goal)

    def test_two_cell_swap_reports_deadlock(self) -> None:
        instance = MAPFInstance(
            width=1,
            height=2,
            obstacles=set(),
            agents=[
                Agent(start=(0, 0), goal=(0, 1)),
                Agent(start=(0, 1), goal=(0, 0)),
            ],
        )

        with self.assertRaisesRegex(PlanningFailure, "deadlock"):
            solve_lra_star(instance, max_iterations=20)

    def test_ten_agents_terminate_with_solution_or_failure(self) -> None:
        agents = [
            Agent((0, 0), (9, 9)),
            Agent((9, 0), (0, 9)),
            Agent((0, 9), (9, 0)),
            Agent((9, 9), (0, 0)),
            Agent((1, 2), (8, 7)),
            Agent((8, 2), (1, 7)),
            Agent((2, 1), (7, 8)),
            Agent((7, 1), (2, 8)),
            Agent((1, 5), (8, 5)),
            Agent((8, 4), (1, 4)),
        ]
        instance = MAPFInstance(10, 10, set(), agents)

        try:
            solution = solve_lra_star(instance, max_iterations=500)
        except PlanningFailure:
            return

        self.assertEqual(find_all_conflicts(solution), [])


if __name__ == "__main__":
    unittest.main()
