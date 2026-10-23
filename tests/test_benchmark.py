import unittest

from benchmark import (
    make_bottleneck,
    make_corridor_swap,
    make_dense_empty,
    make_empty,
    make_hub,
    make_random_obstacles,
)
from mapf_lns import PlanningFailure, prioritized_planning


class BenchmarkTests(unittest.TestCase):
    def assert_same_instance(self, first, second) -> None:
        self.assertEqual(first.width, second.width)
        self.assertEqual(first.height, second.height)
        self.assertEqual(first.obstacles, second.obstacles)
        self.assertEqual(first.agents, second.agents)

    def test_generators_are_reproducible(self) -> None:
        factories = (
            lambda: make_empty(8, 8, 17),
            lambda: make_random_obstacles(10, 8, 0.2, 17),
            lambda: make_corridor_swap(6, 17),
        )
        for factory in factories:
            with self.subTest(factory=factory):
                first = factory()
                second = factory()
                self.assert_same_instance(first, second)

    def test_dense_empty_is_reproducible(self) -> None:
        self.assert_same_instance(
            make_dense_empty(8, 20, 0), make_dense_empty(8, 20, 0)
        )

    def test_bottleneck_is_reproducible(self) -> None:
        self.assert_same_instance(
            make_bottleneck(16, 12, 0), make_bottleneck(16, 12, 0)
        )

    def test_hub_is_reproducible(self) -> None:
        self.assert_same_instance(make_hub(16, 12, 0), make_hub(16, 12, 0))

    def test_random_obstacle_instance_has_valid_free_endpoints(self) -> None:
        instance = make_random_obstacles(12, 10, 0.3, 5)
        for agent in instance.agents:
            self.assertNotIn(agent.start, instance.obstacles)
            self.assertNotIn(agent.goal, instance.obstacles)
            self.assertTrue(instance.in_bounds(agent.start))
            self.assertTrue(instance.in_bounds(agent.goal))

    def test_corridor_pp_fails_or_has_high_soc(self) -> None:
        instance = make_corridor_swap(6, 0)
        manhattan_lower_bound = sum(
            abs(agent.start[0] - agent.goal[0])
            + abs(agent.start[1] - agent.goal[1])
            for agent in instance.agents
        )
        try:
            solution = prioritized_planning(instance)
        except PlanningFailure:
            return  # PP is incomplete; failure is an allowed benchmark outcome.

        # Opposing traffic should cost at least the independent-path lower bound.
        self.assertGreaterEqual(solution.compute_soc(), manhattan_lower_bound)


if __name__ == "__main__":
    unittest.main()
