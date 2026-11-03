import random
import unittest
from collections import Counter
from unittest.mock import patch

from mapf_lns import Agent, LNSConfig, MAPFInstance, PlanningFailure, Solution, run_lns
from mapf_lns.lns import _AdaptiveSelector


def instance():
    return MAPFInstance(3, 3, set(), [
        Agent((0, 0), (2, 1)), Agent((1, 0), (2, 0)),
        Agent((0, 2), (0, 2)),
    ])


class NeighborhoodModeTests(unittest.TestCase):
    def test_agent_based_is_used_for_every_iteration(self):
        from mapf_lns.destroy import AgentBasedDestroy

        original = AgentBasedDestroy.destroy
        with patch.object(AgentBasedDestroy, 'destroy', autospec=True,
                          side_effect=original) as destroy:
            _, logs = run_lns(instance(), LNSConfig(
                max_iterations=100, gamma=.6, neighborhood_mode='agent_based'))
        self.assertEqual(destroy.call_count, 100)
        self.assertEqual(len(logs), 100)
        self.assertTrue(all(log.heuristic == 'agent_based' for log in logs))

    def test_map_based_is_used_for_every_iteration(self):
        from mapf_lns.destroy import NeighborhoodDestroy

        original = NeighborhoodDestroy.destroy
        with patch.object(NeighborhoodDestroy, 'destroy', autospec=True,
                          side_effect=original) as destroy:
            _, logs = run_lns(instance(), LNSConfig(
                max_iterations=100, gamma=.6, neighborhood_mode='map_based'))
        self.assertEqual(destroy.call_count, 100)
        self.assertEqual(len(logs), 100)
        self.assertTrue(all(log.heuristic == 'neighborhood' for log in logs))

    def test_default_keeps_the_original_choice_rng_sequence(self):
        names = ['random', 'agent_based', 'neighborhood']
        rng = random.Random(17)
        expected = [rng.choice(names) for _ in range(100)]
        # Empty neighborhoods isolate operator selection from repair RNG usage.
        with patch('mapf_lns.lns.RandomDestroy.destroy', return_value=set()), \
             patch('mapf_lns.lns.AgentBasedDestroy.destroy', return_value=set()), \
             patch('mapf_lns.lns.NeighborhoodDestroy.destroy', return_value=set()):
            _, logs = run_lns(instance(), LNSConfig(max_iterations=100, seed=17))
        self.assertEqual([log.heuristic for log in logs], expected)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'neighborhood_mode'):
            LNSConfig(neighborhood_mode='unknown')

    def test_alns_is_reproducible_and_logs_concrete_operator_names(self):
        config = LNSConfig(max_iterations=120, gamma=.6, seed=17,
                           neighborhood_mode='alns')
        first = run_lns(instance(), config)
        self.assertEqual(first, run_lns(instance(), config))
        self.assertEqual({log.heuristic for log in first[1]},
                         {'random', 'agent_based', 'neighborhood'})


class AdaptiveSelectorTests(unittest.TestCase):
    def test_segment_average_update_reset_and_unselected_operator(self):
        selector = _AdaptiveSelector(3)
        for _ in range(25):
            selector.record(0, 33)
        for _ in range(24):
            selector.record(1, 9)
        self.assertEqual(selector.weights, [1., 1., 1.])
        selector.record(1, 9)
        self.assertAlmostEqual(selector.weights[0], 4.2)
        self.assertAlmostEqual(selector.weights[1], 1.8)
        self.assertEqual(selector.weights[2], 1.)
        self.assertEqual(selector.scores, [0., 0., 0.])
        self.assertEqual(selector.counts, [0, 0, 0])
        for _ in range(50):
            selector.record(0, 0)
        self.assertAlmostEqual(selector.weights[0], 3.78)
        self.assertAlmostEqual(selector.weights[1], 1.8)
        self.assertEqual(selector.weights[2], 1.)

    def test_roulette_boundaries(self):
        selector = _AdaptiveSelector(3)
        selector.weights = [1., 2., 7.]
        rng = random.Random(0)
        for draw, expected in [(0., 0), (.099, 0), (.1, 1),
                               (.299, 1), (.3, 2), (.999, 2)]:
            with self.subTest(draw=draw), patch.object(rng, 'random', return_value=draw):
                self.assertEqual(selector.select(rng), expected)

    def test_roulette_distribution_matches_weights(self):
        selector = _AdaptiveSelector(3)
        selector.weights = [1., 2., 7.]
        rng = random.Random(42)
        counts = Counter(selector.select(rng) for _ in range(20000))
        for index, probability in enumerate([.1, .2, .7]):
            self.assertAlmostEqual(counts[index] / 20000, probability, delta=.015)

    def test_run_lns_rewards_best_accept_reject_and_repair_failure(self):
        # Valid one-agent paths consisting of waits; mock repair controls costs.
        single = MAPFInstance(1, 1, set(), [Agent((0, 0), (0, 0))])
        initial = Solution({0: [(0, 0)] * 11})
        improved = Solution({0: [(0, 0)] * 9})
        rejected = Solution({0: [(0, 0)] * 10})
        selector = _AdaptiveSelector(3)
        outcomes = [improved, improved, rejected] + [PlanningFailure('forced')] * 47
        with patch('mapf_lns.lns._AdaptiveSelector', return_value=selector), \
             patch.object(selector, 'select', return_value=0), \
             patch('mapf_lns.lns.prioritized_planning', return_value=initial), \
             patch('mapf_lns.lns.RandomDestroy.destroy', return_value={0}), \
             patch('mapf_lns.lns.repair_prioritized', side_effect=outcomes):
            best, logs = run_lns(single, LNSConfig(
                max_iterations=50, neighborhood_mode='alns'))
        self.assertEqual(best, improved)
        self.assertEqual([log.accepted for log in logs[:3]], [True, True, False])
        self.assertTrue(all(log.failure_reason == 'repair_failed' for log in logs[3:]))
        self.assertAlmostEqual(selector.weights[0], .9 + .1 * (33 + 9 + 1) / 50)
        self.assertEqual(selector.weights[1:], [1., 1.])
        self.assertEqual(selector.counts, [0, 0, 0])

    def test_empty_neighborhoods_count_toward_segment_with_zero_reward(self):
        selector = _AdaptiveSelector(3)
        with patch('mapf_lns.lns._AdaptiveSelector', return_value=selector), \
             patch.object(selector, 'select', return_value=1), \
             patch('mapf_lns.lns.AgentBasedDestroy.destroy', return_value=set()), \
             patch('mapf_lns.lns.repair_prioritized') as repair:
            _, logs = run_lns(instance(), LNSConfig(
                max_iterations=50, neighborhood_mode='alns'))
        repair.assert_not_called()
        self.assertTrue(all(log.failure_reason == 'neighborhood_unavailable' for log in logs))
        self.assertAlmostEqual(selector.weights[1], .9)
        self.assertEqual(selector.weights[0], 1.)
        self.assertEqual(selector.weights[2], 1.)


if __name__ == '__main__':
    unittest.main()
