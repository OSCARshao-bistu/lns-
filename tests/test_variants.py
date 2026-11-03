import unittest
from unittest.mock import patch

import experiments_variants as experiment
from mapf_lns import Agent, LNSConfig, MAPFInstance, PlanningFailure, Solution, run_lns
from mapf_lns.lns import LNSIterationLog


class InitializerTests(unittest.TestCase):
    def setUp(self):
        self.instance = MAPFInstance(2, 1, set(), [Agent((0, 0), (1, 0))])
        self.initial = Solution({0: [(0, 0), (1, 0)]})

    def test_each_initializer_is_called_once_and_others_are_not_called(self):
        for method, expected in [("pp", 0), ("pps", 1), ("eecbs", 2)]:
            with self.subTest(method=method), \
                 patch('mapf_lns.lns.prioritized_planning', return_value=self.initial) as pp, \
                 patch('mapf_lns.lns.pps.solve_pps', return_value=self.initial) as pps, \
                 patch('mapf_lns.lns.eecbs.solve_eecbs', return_value=self.initial) as eecbs:
                result, logs = run_lns(self.instance, LNSConfig(
                    init_method=method, max_iterations=0))
                self.assertEqual(result, self.initial)
                self.assertEqual(logs, [])
                for index, solver in enumerate([pp, pps, eecbs]):
                    self.assertEqual(solver.call_count, int(index == expected))
                [pp, pps, eecbs][expected].assert_called_once_with(self.instance)

    def test_initializer_failure_propagates_before_entering_lns(self):
        for method, target in [
            ('pp', 'prioritized_planning'), ('pps', 'pps.solve_pps'),
            ('eecbs', 'eecbs.solve_eecbs'),
        ]:
            failure = PlanningFailure('initialization failed')
            with self.subTest(method=method), \
                 patch('mapf_lns.lns.' + target, side_effect=failure), \
                 patch('mapf_lns.lns.RandomDestroy') as destroy, \
                 patch('mapf_lns.lns.repair_prioritized') as repair:
                with self.assertRaises(PlanningFailure) as caught:
                    run_lns(self.instance, LNSConfig(init_method=method))
                self.assertIs(caught.exception, failure)
                destroy.assert_not_called()
                repair.assert_not_called()

    def test_default_matches_explicit_pp_and_random(self):
        default = run_lns(self.instance, LNSConfig(max_iterations=20))
        explicit = run_lns(self.instance, LNSConfig(
            max_iterations=20, init_method='pp', neighborhood_mode='random'))
        self.assertEqual(default, explicit)

    def test_all_real_initializers_can_run_with_lns(self):
        for method in ['pp', 'pps', 'eecbs']:
            with self.subTest(method=method):
                result, logs = run_lns(self.instance, LNSConfig(
                    init_method=method, max_iterations=3))
                self.assertEqual(result, self.initial)
                self.assertEqual(len(logs), 3)

    def test_invalid_initializer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'init_method'):
            LNSConfig(init_method='unknown')


class ExperimentTests(unittest.TestCase):
    def test_local_benchmark_loader(self):
        instance = experiment.load_instance(
            experiment.ROOT / 'data/random-32-32-20.map',
            experiment.ROOT / 'data/random-32-32-20.scen', 32)
        self.assertEqual((instance.width, instance.height), (32, 32))
        self.assertEqual(len(instance.agents), 32)
        self.assertGreater(len(instance.obstacles), 0)

    def test_initial_soc_status_and_continuing_after_failure(self):
        instance = MAPFInstance(2, 1, set(), [Agent((0, 0), (1, 0))])
        solution = Solution({0: [(0, 0), (1, 0)]})
        log = LNSIterationLog(0, 'random', 3, 1, True, 1)
        completed = (solution, [log] * experiment.MAX_ITERATIONS)
        outcomes = [completed, PlanningFailure('initial failed'),
                    completed, (solution, [log]), (solution, [])]
        with patch.object(experiment, 'run_lns', side_effect=outcomes) as solver:
            rows = experiment.run_experiment(instance)
        self.assertEqual(solver.call_count, 5)
        self.assertEqual([row['status'] for row in rows],
                         ['completed', 'failed', 'completed', 'time_limit', 'time_limit'])
        self.assertEqual(rows[0]['initial_soc'], 3)
        self.assertEqual(rows[0]['final_soc'], 1)
        self.assertEqual(rows[1]['initial_soc'], 'Failed')
        self.assertEqual(rows[1]['final_soc'], '')
        self.assertEqual(rows[1]['error'], 'initial failed')
        self.assertEqual(rows[4]['initial_soc'], 1)
        self.assertEqual(
            [(call.args[1].init_method, call.args[1].neighborhood_mode)
             for call in solver.call_args_list],
            [(method, mode) for _, method, mode in experiment.VARIANTS])


if __name__ == '__main__':
    unittest.main()
