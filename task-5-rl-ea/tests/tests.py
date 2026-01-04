import unittest
from unittest import TestCase
import numpy as np

OK_MSG = 'Your data looks alright, you can continue with the workbook' 
ERR_MSG = 'Oops, there is still an issue. Please fix your implementation before continuing with the workbook'

class EA_RL_Tests(TestCase):
    def test_return_values(self, calculate_returns):
        rewards = [0, 0, 0, 1]
        actual_returns = calculate_returns(rewards, gamma=0.9)
        self.assertListEqual(actual_returns, sorted(actual_returns), msg=ERR_MSG)
        self.assertEqual(actual_returns[0], 0.729, msg=ERR_MSG)
        self.assertEqual(actual_returns[-1], 1.0, msg=ERR_MSG)

        actual_returns = calculate_returns(rewards, gamma=0.5)
        self.assertListEqual(actual_returns, sorted(actual_returns), msg=ERR_MSG)
        self.assertEqual(actual_returns[0], 0.125, msg=ERR_MSG)
        self.assertEqual(actual_returns[-1], 1.0, msg=ERR_MSG)
        print(OK_MSG)

    def test_value_function(self, value_func):
        self.assertIsInstance(value_func, np.ndarray, msg=ERR_MSG)
        self.assertEqual(value_func.shape, (9,), msg=ERR_MSG)
        self.assertEqual(value_func[5], 10.0, msg=ERR_MSG)
        self.assertEqual(value_func[7], 10.0, msg=ERR_MSG)
        self.assertEqual(value_func[8], 0.0, msg=ERR_MSG)
        self.assertEqual(np.argmin(value_func[:-1]), 0, msg=ERR_MSG)
        print(OK_MSG)

    def test_q_table_update(self, update_q_table):
        q_table = [[0.0, 0.0], [0.0, 1.0]]
        update_q_table(q_table, state=0, action=1, reward=1, next_state=1, alpha=0.1, gamma=0.9)
        expected_q_table = [[0.0, 0.19], [0.0, 1.0]]
        self.assertListEqual(q_table, expected_q_table, msg=ERR_MSG)
        print(OK_MSG)

    def test_q_learning(self, q_learning, frozenlake_env):
        q_table = q_learning(env=frozenlake_env, episodes=1000, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.999)
        self.assertIsInstance(q_table, np.ndarray, msg=ERR_MSG)
        self.assertEqual(q_table.shape, (16, 4), msg=ERR_MSG)

        num_zero_rows = np.sum(np.all(q_table == 0, axis=1))
        self.assertEqual(num_zero_rows, 5, msg=ERR_MSG)  # 4 holes + 1 goal state
        print(OK_MSG)

    def test_evaluate_random_policy(self, evaluate_random_policy, frozenlake_env):
        success_rate = evaluate_random_policy(env=frozenlake_env, episodes=1000)
        self.assertIsInstance(success_rate, float, msg=ERR_MSG)
        self.assertTrue(0.0 <= success_rate <= 0.3, msg=ERR_MSG)
        print(OK_MSG)

    def test_evaluate_q_learning(self, evaluate_q_learning, frozenlake_env, q_table):
        success_rate = evaluate_q_learning(env=frozenlake_env, q_table=q_table, episodes=1000)
        self.assertIsInstance(success_rate, float, msg=ERR_MSG)
        self.assertEqual(success_rate, 1.0, msg=ERR_MSG)
        print(OK_MSG)

    def test_selection(self, selection_step, population):
        selected = selection_step._do(population)
        self.assertIsInstance(selected, list, msg=ERR_MSG)
        self.assertGreaterEqual(len(selected), 5, msg=ERR_MSG)
        self.assertTrue(all(individual in population for individual in selected), msg=ERR_MSG)
        print(OK_MSG)

    def test_crossover(self, crossover_step, population):
        children = crossover_step._do(population)
        self.assertIsInstance(children, list, msg=ERR_MSG)
        self.assertEqual(len(children), 50, msg=ERR_MSG)
        self.assertEqual(len(children), len(set(children)), msg=ERR_MSG)
        self.assertTrue(all(children[i]._parameter.keys() == population[0]._parameter.keys() for i in range(len(children))), msg=ERR_MSG)
        print(OK_MSG)

    def test_survival(self, survival_step, population):
        survivors = survival_step._do(population)
        self.assertIsInstance(survivors, list, msg=ERR_MSG)
        self.assertEqual(len(survivors), survival_step._n, msg=ERR_MSG)
        self.assertTrue(all(individual in population for individual in survivors), msg=ERR_MSG)
        self.assertEqual(len(survivors), len(set(survivors)), msg=ERR_MSG)  # Ensure uniqueness
        print(OK_MSG)